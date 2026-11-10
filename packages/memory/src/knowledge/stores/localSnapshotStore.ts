/**
 * knowledge/stores/localSnapshotStore.ts — a {@link KnowledgeStore} over one JSON snapshot,
 * for a folder not linked to a project (PRD 38 §4.2, §5.1).
 *
 * Persistence is a port (`read`/`write` of the whole snapshot string), so this module has
 * no `fs`: the memory MCP host persists it beside its own snapshot through its shared JSON
 * file, a browser host in IndexedDB. Upserts use the platform's unique keys — a fact by
 * `key`, a relation by `(from, to, kind)`, a scope by `rootId` — so linking a folder later
 * uploads the same rows idempotently. The snapshot is written canonically (sorted), so
 * equal content is equal bytes.
 *
 * Every write serializes the WHOLE snapshot, so a many-mutation run (a directory review)
 * would cost quadratic bytes. {@link LocalSnapshotStore.batched} holds persistence back
 * inside a run; outside one, each awaited mutation has persisted when it resolves.
 *
 * PRD 41 A10: reads come from memory once loaded (never behind the mutation queue); rows live in
 * per-root maps (rowMemory.ts) updated incrementally; with the optional `extractedLog` port the
 * extract cache is an append-only JSON Lines file (extractLogWriter.ts) instead of part of the snapshot.
 *
 * PRD 41 B4 (G03): with `opts.rows` ({@link RowPersistence}) the store persists only the rows that changed
 * (dirty keys in rowDelta.ts) and `persistence` / `extractedLog` are not used. Construct it as
 * `new LocalSnapshotStore(undefined, { rows })`; a `persistence` given beside `rows` is ignored.
 */

import type { Extracted, KnowledgeStore, NodeRef, Scope, ScopeRecord, StoredFact, StoredRelation } from '../model.js';
import { canonicalJson, compareStr, pathOfRef } from '../refs.js';
import { ExtractLogWriter } from './extractLogWriter.js';
import type { RowDelta, RowPersistence } from './rowDelta.js';
import { RowMemory } from './rowMemory.js';
import { RowSync } from './rowSync.js';
import { emptySnapshot, parse, stripUndefined, type KnowledgeSnapshot } from './snapshotFormat.js';

export type { KnowledgeSnapshot } from './snapshotFormat.js';
export type { RowChange, RowPersistence } from './rowDelta.js';

export interface SnapshotPersistence {
    read(): Promise<string | undefined>;
    write(snapshot: string): Promise<void>;
    /** Keeps the bytes of a snapshot that did not parse (e.g. as `<snapshot>.corrupt`) before the store starts empty. Optional: without it they are lost. */
    keepAside?(text: string): Promise<void>;
}

/**
 * Append-only, content-addressed home of the extract cache (PRD 41 A10 rule 7). JSON Lines, one
 * `{"h":"<hash>","e":[...]}` per line; the LAST line for a hash wins, an unparseable (torn) line is ignored.
 */
export interface ExtractedLogPersistence {
    read(): Promise<string | undefined>;
    /** Appends `text` (whole lines, newline-terminated) to the end of the log. */
    append(text: string): Promise<void>;
    /** Replaces the whole log with `text` (atomically where the host can). Optional: without it the log is never compacted. */
    replace?(text: string): Promise<void>;
}

export interface LocalSnapshotStoreOptions {
    /** When given, the extract cache lives in this log and the snapshot omits `extracted`. Ignored with `rows`. */
    extractedLog?: ExtractedLogPersistence;
    /** Row-level persistence (a SQLite file): only changed rows are written; `SnapshotPersistence` and `extractedLog` are not used. */
    rows?: RowPersistence;
    /** ISO clock for `staleAt`. Default `new Date().toISOString()`. */
    now?(): string;
    /** Inside `batched`, a mutation persists when this many ms passed since the last write (a crash loses at most this much). Default 30_000. */
    checkpointMs?: number;
    /** Millisecond clock for the checkpoint. Default `Date.now`. */
    nowMs?(): number;
}

export class LocalSnapshotStore implements KnowledgeStore {
    private loaded = false;
    private loading: Promise<void> | undefined;
    /** Per-root rows and the extract cache (PRD 41 A10 rule 7). */
    private mem = new RowMemory();
    private readonly log: ExtractLogWriter | undefined;
    private readonly rows: RowPersistence | undefined;
    /** Row mode only: writes the changed rows, reloads, reads bodies lazily (rowSync.ts). */
    private readonly sync: RowSync | undefined;
    /** Row mode only: the keys changed since the last `apply`. */
    private readonly delta: RowDelta | undefined;
    /** Serializes mutations so two writers never interleave a read-modify-write. */
    private queue: Promise<unknown> = Promise.resolve();
    private readonly now: () => string;
    private readonly nowMs: () => number;
    private readonly checkpointMs: number;
    /** Open `batched` calls; persistence is held while it is above zero. */
    private depth = 0;
    /** Memory is ahead of the persisted snapshot. */
    private dirty = false;
    private persistedAt = 0;
    /** Serializes snapshot writes; never rejects. */
    private persisting: Promise<void> = Promise.resolve();
    /** Steps queued or running on the persist chain. */
    private steps = 0;
    /** A background checkpoint in flight (at most one). */
    private checkpointing: Promise<void> | undefined;

    /** Row-persisted: no whole-snapshot port; only changed rows are written through `opts.rows`. */
    constructor(persistence: undefined, opts: LocalSnapshotStoreOptions & { rows: RowPersistence });
    /** Snapshot-persisted (the default): the whole canonical snapshot is written through `persistence`. */
    constructor(persistence: SnapshotPersistence, opts?: LocalSnapshotStoreOptions);
    constructor(private readonly persistence: SnapshotPersistence | undefined, opts: LocalSnapshotStoreOptions = {}) {
        this.rows = opts.rows;
        if (!this.rows && !persistence) throw new Error('LocalSnapshotStore: give a SnapshotPersistence or opts.rows');
        this.sync = this.rows && new RowSync(this.rows, {
            mem: () => this.mem, setMem: (m) => { this.mem = m; }, enqueue: (t) => this.enqueue(t), chain: (s) => this.onChain(s), inBatch: () => this.depth > 0, busy: () => this.steps > 0,
        });
        this.delta = this.sync?.delta;
        this.now = opts.now ?? (() => new Date().toISOString());
        this.nowMs = opts.nowMs ?? Date.now;
        this.checkpointMs = opts.checkpointMs ?? 30_000;
        this.log = !this.rows && opts.extractedLog ? new ExtractLogWriter(opts.extractedLog, (h) => this.mem.extracted.has(h)) : undefined;
    }

    /** Loads once; every later call is a no-op, so reads never wait on the mutation queue after the first. */
    private load(): Promise<void> {
        if (this.loaded) return Promise.resolve();
        this.loading ??= this.doLoad().catch((err) => { this.loading = undefined; throw err; });
        return this.loading;
    }

    /** What a read waits for: the first load, then (row mode) a reload when another writer changed the store. */
    private async fresh(): Promise<void> {
        await this.load();
        await this.sync?.sync();
    }

    private async doLoad(): Promise<void> {
        let snap: KnowledgeSnapshot;
        let logText: string | undefined;
        if (this.rows) {
            snap = (await this.rows.load()) ?? emptySnapshot();
        } else {
            const snapText = await this.persistence!.read();
            try {
                snap = parse(snapText);
            } catch {
                // Unreadable: never silently replaced. Its bytes are kept aside, then the store starts empty (the next review rebuilds).
                await this.persistence!.keepAside?.(snapText!).catch(() => undefined);
                snap = emptySnapshot();
            }
            logText = this.log ? await this.log.read() : undefined;
        }
        this.mem.fill(snap, !this.log);
        // Row mode: bodies the host holds but never sent are tracked by key, so a later edit's eviction reaches them too.
        if (this.rows?.extractedSources) {
            for (const [h, refs] of Object.entries(await this.rows.extractedSources())) if (!this.mem.extracted.has(h)) this.mem.seedKey(h, refs);
        }
        if (this.log) {
            await this.log.load(logText, this.mem);
            // An old snapshot still carrying the cache: move it into the log once; the next snapshot write omits it.
            const legacy = Object.entries(snap.extracted ?? {});
            if (legacy.length) {
                for (const [h, e] of legacy) {
                    if (this.mem.extracted.has(h)) continue;
                    this.mem.seedBody(h, e);
                    this.log.queue(h, e);
                }
                this.dirty = true;
            }
        }
        this.loaded = true;
    }

    /** Runs a step after the previous persist step (one at a time); the chain itself never rejects. */
    private onChain<T>(step: () => Promise<T>): Promise<T> {
        const run = this.persisting.then(step);
        this.steps += 1;
        this.persisting = run.then(() => undefined, () => undefined).then(() => { this.steps -= 1; });
        return run;
    }

    private enqueue<T>(task: () => Promise<T>): Promise<T> {
        const run = this.queue.then(task);
        this.queue = run.catch(() => undefined);
        return run;
    }

    /**
     * Appends new extract entries, then writes the snapshot (or, in row mode, the changed rows) if memory is ahead of it,
     * one write at a time. It runs OUTSIDE the mutation queue: a write of a large snapshot (serialize + disk) must never
     * hold up reads or the next mutation's in-memory apply (PRD 41 A10 — it did: every `getExtracted` waited out each
     * checkpoint). The log goes first, so a snapshot that dropped a migrated cache never outlives its entries. A failed
     * write leaves the store dirty, so the next one retries it.
     */
    private persistIfDirty(): Promise<void> {
        return this.onChain(async () => {
            if (!this.loaded) return;
            await this.log?.flush();
            if (!this.dirty) return;
            this.dirty = false;
            try {
                await (this.sync ? this.sync.apply() : this.persistence!.write(this.mem.serialize(this.log ? undefined : this.mem.extracted)));
            } catch (err) {
                this.dirty = true;
                throw err;
            }
            this.persistedAt = this.nowMs();
        });
    }

    /**
     * Applies in memory (queued, so reads and writes never interleave), then persists: at once outside a
     * batch, else a checkpoint in the BACKGROUND once `checkpointMs` passed. A plain mutation from another
     * caller during a batch (the live file watcher shares this store) is held like the batch's own and made
     * durable by the next checkpoint, flush or batch end. `snapshotChanged: false` is for a change that only
     * touches the extract log.
     */
    private async mutate(fn: () => void, snapshotChanged = true): Promise<void> {
        await this.enqueue(async () => {
            await this.load();
            fn();
            if (snapshotChanged) this.dirty = true;
        });
        if (this.depth === 0) await this.persistIfDirty();
        else if (!this.checkpointing && this.nowMs() - this.persistedAt >= this.checkpointMs) {
            // A failed checkpoint is not lost: the snapshot stays dirty and the batch's end or flush writes (and reports) it.
            // It backs off by `checkpointMs` (advances `persistedAt`) so a failing disk is not retried on every mutation.
            this.checkpointing = this.persistIfDirty().catch(() => { this.persistedAt = this.nowMs(); }).finally(() => { this.checkpointing = undefined; });
        }
    }

    /**
     * Holds persistence while `work` runs. `flush` persists once if anything is unwritten. When `work`
     * settles the outermost batch persists what is left: `work`'s own error wins over a failed final
     * write, a failed final write fails a successful `work`. Batches may nest or overlap.
     */
    async batched<T>(work: (flush: () => Promise<void>) => Promise<T>): Promise<T> {
        if (this.depth === 0) this.persistedAt = this.nowMs();
        this.depth += 1;
        // The end goes through the queue so every mutation made during the batch is held, then written once.
        const end = async () => {
            await this.enqueue(async () => { this.depth -= 1; });
            if (this.depth === 0) await this.persistIfDirty();
        };
        let result: T;
        try {
            result = await work(async () => { await this.queue; await this.persistIfDirty(); });
        } catch (err) {
            await end().catch(() => undefined);
            throw err;
        }
        await end();
        return result;
    }

    async scopes(): Promise<ScopeRecord[]> {
        await this.fresh();
        return [...this.mem.scopes];
    }

    async facts(scope: Scope): Promise<StoredFact[]> {
        await this.fresh();
        return [...(this.mem.facts.get(scope.rootId)?.values() ?? [])];
    }

    async relations(scope: Scope): Promise<StoredRelation[]> {
        await this.fresh();
        return [...(this.mem.relations.get(scope.rootId)?.values() ?? [])];
    }

    write(batch: { scope: ScopeRecord; facts: StoredFact[]; relations: StoredRelation[] }): Promise<void> {
        const bad = [...batch.facts, ...batch.relations].find((x) => (x.origin as string) === 'extracted');
        if (bad) return Promise.reject(new Error('knowledge: extracted output is never stored'));
        return this.mutate(() => {
            const rootId = batch.scope.rootId;
            const prior = this.mem.scopes.find((x) => x.rootId === rootId);
            const merged: ScopeRecord = { ...(prior ?? {}), ...stripUndefined(batch.scope) } as ScopeRecord;
            this.mem.scopes = [...this.mem.scopes.filter((x) => x.rootId !== rootId), merged];
            this.delta?.mark('scopes', [rootId]);

            const facts = this.mem.factsOf(rootId);
            for (const f of batch.facts) { facts.set(f.key, { ...f }); this.delta?.mark('facts', [rootId, f.key]); }
            this.mem.relsOf(rootId);
            for (const r of batch.relations) { this.mem.putRelation(rootId, { ...r }); this.delta?.mark('relations', [rootId, r.from, r.to, r.kind]); }
        });
    }

    markStale(scope: Scope, sourceRef: NodeRef, currentHash: string): Promise<void> {
        const path = pathOfRef(sourceRef);
        return this.mutate(() => {
            const at = this.now();
            const rels = this.mem.relsOf(scope.rootId);
            const keys = path === undefined ? undefined : this.mem.relationKeysAt(scope.rootId, path);
            for (const key of keys ?? []) {
                const r = rels.get(key);
                if (r && r.origin === 'model' && !r.staleAt && r.sourceHash !== currentHash) {
                    rels.set(key, { ...r, staleAt: at });
                    this.delta?.mark('relations', [scope.rootId, r.from, r.to, r.kind]);
                }
            }
        });
    }

    /** The cached entries of a content hash. In row mode a hash not in memory is read through `rows.loadExtracted` and kept. */
    async getExtracted(hash: string): Promise<Extracted[] | undefined> {
        await this.fresh();
        return this.sync ? this.sync.body(hash) : this.mem.extracted.get(hash);
    }

    /** Row mode: reads the bodies of these hashes through `rows.loadExtracted`, a chunk at a time. Otherwise a no-op (everything is in memory). */
    async preloadExtracted(hashes: readonly string[]): Promise<void> {
        if (!this.sync) return;
        await this.fresh();
        await this.sync.preload(hashes);
    }

    /** Asks the next review to re-check the files with these content hashes (persisted until {@link clearRecheck}). */
    markRecheck(hashes: readonly string[]): Promise<void> {
        // The snapshot is dirty only when the set really changed (a repeated mark costs no write).
        return this.mutate(() => {
            for (const h of hashes) if (!this.mem.recheck.has(h)) { this.mem.recheck.add(h); this.dirty = true; this.delta?.mark('recheck', [h]); }
        }, false);
    }

    async hasRecheck(hash: string): Promise<boolean> {
        await this.fresh();
        return this.mem.recheck.has(hash);
    }

    async recheckHashes(): Promise<string[]> {
        await this.fresh();
        return [...this.mem.recheck].sort(compareStr);
    }

    /** Forgets hashes a review has handled. */
    clearRecheck(hashes: readonly string[]): Promise<void> {
        return this.mutate(() => {
            for (const h of hashes) if (this.mem.recheck.delete(h)) { this.dirty = true; this.delta?.mark('recheck', [h], true); }
        }, false);
    }

    /** Every cached entry by content hash (the log's, else the snapshot's `extracted`), for a host that packs the cache. Row mode reads ALL bodies through `loadAllExtracted` (memory wins) and throws without it: never a partial map. */
    async extractedEntries(): Promise<Record<string, Extracted[]>> {
        await this.fresh();
        return Object.fromEntries(this.sync ? await this.sync.allBodies() : this.mem.extracted);
    }

    putExtracted(hash: string, extracted: Extracted[]): Promise<void> {
        return this.mutate(() => {
            // The same entry again (a second pass re-puts what the first stored) is skipped: no append, no dirty snapshot, and no
            // claim — a re-put never makes the files of a shared body current at it again (extractRefs.ts).
            const prior = this.mem.extracted.get(hash);
            if (prior && canonicalJson(prior) === canonicalJson(extracted)) return;
            // Only the entries new or changed against `prior` become current; bodies no longer current or previous for any source
            // leave memory here, and the cache's persistence below.
            for (const h of this.mem.putBody(hash, extracted)) this.evicted(h);
            this.sync?.forget(hash);
            this.delta?.mark('extracted', [hash]);
            if (this.log) this.log.queue(hash, extracted);
            else this.dirty = true;
        }, false);
    }

    /**
     * A body {@link RowMemory.putBody} evicted: a row delete (row mode), never appended if still queued and dropped from the log at its
     * next compaction (log mode), or gone from the next snapshot write (snapshot mode).
     */
    private evicted(hash: string): void {
        this.delta?.mark('extracted', [hash], true);
        this.log?.unqueue(hash);
        if (!this.log) this.dirty = true;
    }

    /** See {@link canonicalDump}. Static so it can reach private state without widening the instance's public surface. */
    static async dumpOf(store: LocalSnapshotStore): Promise<string> {
        await store.fresh();
        if (!store.sync) return store.mem.serialize(store.log ? undefined : store.mem.extracted);
        const bodies = await store.sync.allBodies();
        return store.mem.serialize(bodies);
    }
}

/**
 * The canonical text of a store's rows: byte-equal to what a snapshot-persisted store writes (without `extractedLog`, the
 * extract cache included). The projection `packPush` / `portablePack` read. Row mode reads every body through `loadAllExtracted` and throws `knowledge_dump_needs_all_bodies` without it.
 */
export const canonicalDump = (store: LocalSnapshotStore): Promise<string> => LocalSnapshotStore.dumpOf(store);

export function localSnapshotStore(persistence: undefined, opts: LocalSnapshotStoreOptions & { rows: RowPersistence }): LocalSnapshotStore;
export function localSnapshotStore(persistence: SnapshotPersistence, opts?: LocalSnapshotStoreOptions): LocalSnapshotStore;
export function localSnapshotStore(persistence: SnapshotPersistence | undefined, opts?: LocalSnapshotStoreOptions): LocalSnapshotStore {
    return new LocalSnapshotStore(persistence as SnapshotPersistence, opts);
}
