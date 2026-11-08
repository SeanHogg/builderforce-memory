/**
 * knowledge/stores/rowSync.ts — what a ROW-persisted {@link LocalSnapshotStore} does with its {@link RowPersistence}: write the
 * dirty rows, reload when another writer changed the store, and read extract bodies lazily. The store owns the maps, the mutation
 * queue and the persist chain; this reaches them only through {@link RowHost}.
 *
 * Every step that talks to the host's connection about the whole store (an apply, a changed-elsewhere check with its load) runs on
 * the store's `persisting` chain, so a load never interleaves with an apply and no apply can commit between a load and the
 * `replaceWith` that installs it.
 */

import type { Extracted } from '../model.js';
import { RowDelta, type RowPersistence } from './rowDelta.js';
import { RowMemory } from './rowMemory.js';
import type { KnowledgeSnapshot } from './snapshotFormat.js';

export interface RowHost {
    mem(): RowMemory;
    setMem(next: RowMemory): void;
    /** Runs a task inside the store's mutation queue. */
    enqueue<T>(task: () => Promise<T>): Promise<T>;
    /** Runs a step on the store's persist chain (one at a time, never rejects the chain). */
    chain<T>(step: () => Promise<T>): Promise<T>;
    inBatch(): boolean;
    /** A step is in flight on the persist chain. */
    busy(): boolean;
}

const PRELOAD_CHUNK = 64;

export class RowSync {
    readonly delta = new RowDelta();
    /** The one changed-elsewhere check (with its reload) in flight. */
    private checking: Promise<void> | undefined;
    /** Hashes the host had no body for; cleared by a `putExtracted` of that hash and by every reload. */
    private misses = new Set<string>();
    /** Counts reloads; a miss recorded by a read that started before a reload is dropped. */
    private generation = 0;

    constructor(readonly rows: RowPersistence, private readonly host: RowHost) {}

    /**
     * Swaps the dirty marks out, writes those rows and, when the host answers with `reloaded`, replaces memory inside the queue. A
     * failed apply puts the marks back under any newer ones. Call it on the persist chain.
     */
    async apply(): Promise<void> {
        const taken = this.delta.take();
        if (taken.empty) { this.delta.settle(); return; }
        let reloaded: KnowledgeSnapshot | undefined;
        try {
            reloaded = (await this.rows.apply(this.delta.toChange(taken, this.host.mem()))).reloaded;
        } catch (err) {
            this.delta.restore(taken);
            throw err;
        }
        this.delta.settle();
        if (reloaded) await this.host.enqueue(async () => this.replaceWith(reloaded!));
    }

    /**
     * INSIDE the mutation queue: replaces memory with rows read from the host, keeps the extract bodies it already holds
     * (content-addressed, so never stale), lays this store's unwritten rows back on top and rebuilds the relations-by-path index.
     */
    private replaceWith(snap: KnowledgeSnapshot): void {
        const old = this.host.mem();
        const next = new RowMemory();
        next.fill(snap, true);
        for (const [h, e] of old.extracted) if (!next.extracted.has(h)) next.extracted.set(h, e);
        this.delta.overlay(old, next);
        next.reindex();
        this.host.setMem(next);
        this.misses.clear();
        this.generation += 1;
    }

    /** Before a read: when another writer changed the store, load and install its rows, as ONE step on the persist chain. None inside a `batched` run; a failed check keeps memory. */
    sync(): Promise<void> {
        const rows = this.rows;
        if (this.checking) return this.checking;
        // A read never waits on persistence (PRD 41 A10): while a persist step is in flight, answer from memory (the next read checks again; an apply's `reloaded` covers other writers).
        if (!rows.changedElsewhere || this.host.inBatch() || this.host.busy()) return Promise.resolve();
        this.checking =this.host.chain(async () => {
            if (!(await rows.changedElsewhere!())) return;
            const snap = await rows.load();
            if (snap) await this.host.enqueue(async () => this.replaceWith(snap));
        }).catch(() => undefined).finally(() => { this.checking = undefined; });
        return this.checking;
    }

    /** A body not in memory, read through `loadExtracted` once and kept; a miss is remembered until the hash is put or the store reloads. */
    async body(hash: string): Promise<Extracted[] | undefined> {
        const mem = this.host.mem();
        const have = mem.extracted.get(hash);
        if (have || !this.rows.loadExtracted || this.misses.has(hash)) return have;
        const startedIn = this.generation;
        const found = await this.rows.loadExtracted(hash);
        if (!found) { if (startedIn === this.generation) this.misses.add(hash); return undefined; }
        const now = this.host.mem();
        if (!now.extracted.has(hash)) now.extracted.set(hash, found);
        return now.extracted.get(hash);
    }

    /** Reads the bodies of these hashes (those not in memory and not known misses), a chunk at a time. */
    async preload(hashes: readonly string[]): Promise<void> {
        const want = [...new Set(hashes)].filter((h) => !this.host.mem().extracted.has(h) && !this.misses.has(h));
        for (let i = 0; i < want.length; i += PRELOAD_CHUNK) await Promise.all(want.slice(i, i + PRELOAD_CHUNK).map((h) => this.body(h)));
    }

    /** A `putExtracted` of this hash makes an earlier miss stale. */
    forget(hash: string): void { this.misses.delete(hash); }

    /** EVERY body: the host's, with the ones in memory winning. Never partial: without `loadAllExtracted` it throws. */
    async allBodies(): Promise<Map<string, Extracted[]>> {
        if (!this.rows.loadAllExtracted) throw new Error('knowledge_dump_needs_all_bodies');
        const all = new Map(Object.entries(await this.rows.loadAllExtracted()));
        for (const [h, e] of this.host.mem().extracted) all.set(h, e);
        return all;
    }
}
