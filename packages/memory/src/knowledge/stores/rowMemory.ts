/**
 * knowledge/stores/rowMemory.ts — the in-memory rows of a {@link LocalSnapshotStore}, moved out of it: per-root fact and
 * relation maps updated incrementally (PRD 41 A10 rule 7), the relations-by-source-path index, the extract cache and the
 * re-check set, with the snapshot read and write that go with them. No persistence and no queue: the store owns both.
 */

import type { Extracted, ScopeRecord, StoredFact, StoredRelation } from '../model.js';
import { pathOfRef } from '../refs.js';
import { ExtractRefs, sourceRefsOf } from './extractRefs.js';
import type { RowState } from './rowDelta.js';
import { relKey, serializeSnapshot, type KnowledgeSnapshot } from './snapshotFormat.js';

export class RowMemory implements RowState {
    scopes: ScopeRecord[] = [];
    readonly facts = new Map<string, Map<string, StoredFact>>();
    readonly relations = new Map<string, Map<string, StoredRelation>>();
    /** The extract cache by content hash. Written through {@link putBody} / {@link seedBody}, so {@link refs} stays in step. */
    readonly extracted = new Map<string, Extracted[]>();
    /** Which bodies still hold a current source (extractRefs.ts). */
    private refs = new ExtractRefs();
    readonly recheck = new Set<string>();
    /** Per root: source path → keys of the relations touching it (`from` or `to`). */
    private byPath = new Map<string, Map<string, Set<string>>>();

    factsOf(rootId: string): Map<string, StoredFact> {
        let m = this.facts.get(rootId);
        if (!m) this.facts.set(rootId, m = new Map());
        return m;
    }

    relsOf(rootId: string): Map<string, StoredRelation> {
        let m = this.relations.get(rootId);
        if (!m) this.relations.set(rootId, m = new Map());
        return m;
    }

    /** Upserts a relation and indexes it under the source paths of both ends. */
    putRelation(rootId: string, r: StoredRelation): void {
        this.relsOf(rootId).set(relKey(r), r);
        this.index(rootId, r);
    }

    /** Keys of the root's relations touching a source path. */
    relationKeysAt(rootId: string, path: string): Set<string> | undefined {
        return this.byPath.get(rootId)?.get(path);
    }

    /** Rebuilds the path index from the relation maps (after rows were edited directly). */
    reindex(): void {
        this.byPath = new Map();
        for (const [rootId, m] of this.relations) for (const r of m.values()) this.index(rootId, r);
    }

    /**
     * After a reload into this fresh memory: keeps `old`'s body references (their put order, and the bodies it knows only by key)
     * and adds the bodies this memory holds that `old` did not track, as seeds.
     */
    adoptRefs(old: RowMemory): void {
        this.refs = old.refs.clone();
        for (const [h, e] of this.extracted) if (!this.refs.has(h)) this.refs.seed(h, sourceRefsOf(e));
    }

    /**
     * Stores `entries` under `hash`; the sources of the entries new or changed against what `hash` held become current at it, and
     * the bodies no longer current or previous for any source are evicted (extractRefs.ts). Returns the evicted hashes, for the
     * caller's persistence (a row delete, a log entry not to append).
     */
    putBody(hash: string, entries: Extracted[]): string[] {
        const prior = this.extracted.get(hash);
        this.extracted.set(hash, entries);
        const evicted = this.refs.put(hash, entries, prior);
        for (const h of evicted) this.extracted.delete(h);
        return evicted;
    }

    /** Keeps a body whose place in time is unknown (a snapshot, a reload, a lazy read) without superseding any other. */
    seedBody(hash: string, entries: Extracted[]): void {
        this.extracted.set(hash, entries);
        this.refs.seed(hash, sourceRefsOf(entries));
    }

    /** Stores a body with the reference state a compacted log recorded for it (extractRefs.ts `stateOf`). */
    restoreBody(hash: string, entries: Extracted[], state: { current: readonly string[]; previous: readonly string[] }): void {
        this.extracted.set(hash, entries);
        this.refs.restore(hash, state);
    }

    /** What a body is to its sources (current / previous), for a compacted log line. */
    bodyState(hash: string): { current: string[]; previous: string[] } { return this.refs.stateOf(hash); }

    /** Tracks a body the host holds but memory does not (known by its key and source refs only), so eviction covers it. */
    seedKey(hash: string, sourceRefs: readonly string[]): void {
        this.refs.seed(hash, sourceRefs);
    }

    /** Bodies the reference index tracks (diagnostics and tests). */
    get trackedBodies(): number { return this.refs.size; }

    /** Puts a snapshot's rows into these (empty) maps; the extract cache only when `withExtracted` (a log owns it otherwise). */
    fill(snap: KnowledgeSnapshot, withExtracted: boolean): void {
        this.scopes = [...snap.scopes];
        for (const h of snap.recheck ?? []) this.recheck.add(h);
        for (const [rootId, rows] of Object.entries(snap.facts)) {
            const m = this.factsOf(rootId);
            if (Array.isArray(rows)) for (const f of rows) m.set(f.key, f);
        }
        for (const [rootId, rows] of Object.entries(snap.relations)) {
            this.relsOf(rootId);
            if (Array.isArray(rows)) for (const r of rows) this.putRelation(rootId, r);
        }
        if (withExtracted) for (const [h, e] of Object.entries(snap.extracted ?? {})) this.seedBody(h, e);
    }

    /** Canonical text of the rows with the given extract cache; `undefined` omits it (a log owns it). */
    serialize(extracted: Map<string, Extracted[]> | undefined): string {
        return serializeSnapshot({ scopes: this.scopes, facts: this.facts, relations: this.relations, extracted, recheck: this.recheck });
    }

    private index(rootId: string, r: StoredRelation): void {
        const key = relKey(r);
        let idx = this.byPath.get(rootId);
        if (!idx) this.byPath.set(rootId, idx = new Map());
        for (const ref of [r.from, r.to]) {
            const p = pathOfRef(ref);
            if (p === undefined) continue;
            let keys = idx.get(p);
            if (!keys) idx.set(p, keys = new Set());
            keys.add(key);
        }
    }
}
