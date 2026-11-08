/**
 * knowledge/stores/rowDelta.ts — the dirty tracking and the port types of a ROW-persisted {@link LocalSnapshotStore}
 * (PRD 41 B4, G03). A host that keeps rows (a SQLite file) gets only the rows that changed, not the whole snapshot.
 *
 * The delta records KEYS, never values: a mark says "this row was upserted" or "this row was deleted", and the value is
 * read from memory when the change is built. The last mark per key wins (an upsert after a delete is an upsert; a delete
 * after an upsert is a delete). `take()` swaps the marks out for one `apply`; `restore(taken)` merges them back UNDER any
 * newer marks when the apply failed, so nothing is lost and nothing older overrides something newer. While an apply is in
 * flight its marks stay visible to {@link RowDelta.overlay}, so a reload that races it cannot drop those rows from memory.
 *
 * No I/O and no store: the store only calls this.
 */

import type { Extracted, ScopeRecord, StoredFact, StoredRelation } from '../model.js';
import type { KnowledgeSnapshot } from './snapshotFormat.js';
import { relKey } from './snapshotFormat.js';

/** What changed since the last `apply`. Keys of `recheck` are content hashes (the snapshot keeps ONE global set, not one per root). */
export interface RowChange {
    upserts: {
        scopes: ScopeRecord[]; facts: Array<[rootId: string, fact: StoredFact]>; relations: Array<[rootId: string, relation: StoredRelation]>;
        extracted: Array<[hash: string, entries: Extracted[]]>; recheck: string[];
    };
    deletes: {
        scopes: string[]; facts: Array<[rootId: string, key: string]>;
        relations: Array<[rootId: string, from: string, to: string, kind: string]>;
        extracted: string[]; recheck: string[];
    };
}

/** A host that keeps knowledge as rows. With it, `SnapshotPersistence` and `extractedLog` are not used. */
export interface RowPersistence {
    /** Every row. `undefined` is an empty store (first open). `extracted` may be absent: bodies load lazily through `loadExtracted`. */
    load(): Promise<KnowledgeSnapshot | undefined>;
    /**
     * Writes the change in ONE transaction. When another writer changed the store since this one last saw it, return the
     * full rows as `reloaded`: the store replaces its memory with them and re-applies its own newer marks on top.
     * - `reloaded` must be read INSIDE that same transaction, after the change was applied, so it contains the change.
     * - `apply` may be called again with a SUPERSET of a change that already committed (the commit succeeded but returning or
     *   reading `reloaded` threw), so every upsert and delete must be idempotent.
     * - The store never runs `load`, `apply` and the changed-elsewhere check concurrently on one connection.
     * - `loadExtracted` and `loadAllExtracted` MAY run concurrently with an `apply` (reads never wait on persistence), so the host must allow concurrent reads.
     */
    apply(change: RowChange): Promise<{ generation: number; reloaded?: KnowledgeSnapshot }>;
    /** EVERY extract body, for a host that packs or dumps the cache. Without it, `extractedEntries()` and `canonicalDump` throw `knowledge_dump_needs_all_bodies` in row mode: they never return a partial map. Optional. */
    loadAllExtracted?(): Promise<Record<string, Extracted[]>>;
    /** The extract bodies of one content hash that are not in memory (reload returns rows, bodies load lazily). Optional. */
    loadExtracted?(hash: string): Promise<Extracted[] | undefined>;
    /** True when another writer changed the store since this store last saw it (e.g. `PRAGMA data_version`). Optional. */
    changedElsewhere?(): Promise<boolean>;
}

type Kind = 'scopes' | 'facts' | 'relations' | 'extracted' | 'recheck';
const KINDS: readonly Kind[] = ['scopes', 'facts', 'relations', 'extracted', 'recheck'];
type Marks = Record<Kind, Map<string, { key: readonly string[]; del: boolean }>>;
const fresh = (): Marks => ({ scopes: new Map(), facts: new Map(), relations: new Map(), extracted: new Map(), recheck: new Map() });

/** The in-memory rows of a store, by reference: what {@link RowDelta.toChange} reads and {@link RowDelta.overlay} edits. */
export interface RowState {
    scopes: ScopeRecord[];
    facts: Map<string, Map<string, StoredFact>>;
    relations: Map<string, Map<string, StoredRelation>>;
    extracted: Map<string, Extracted[]>;
    recheck: Set<string>;
}

/** The marks swapped out by {@link RowDelta.take}. */
export interface TakenRows { readonly marks: Marks; readonly empty: boolean }

export class RowDelta {
    private cur: Marks = fresh();
    private inflight: Marks | undefined;

    /** Marks a row. `key` is `[rootId]` (scope), `[rootId, key]` (fact), `[rootId, from, to, kind]` (relation), `[hash]` (extracted, recheck). */
    mark(kind: Kind, key: readonly string[], del = false): void {
        this.cur[kind].set(JSON.stringify(key), { key, del });
    }

    /** True when a mark is waiting for the next `apply`. */
    get pending(): boolean { return KINDS.some((k) => this.cur[k].size > 0); }

    /** Swaps the marks out for one `apply`. They stay visible to `overlay` until `settle` or `restore`. */
    take(): TakenRows {
        const marks = this.cur;
        this.cur = fresh();
        this.inflight = marks;
        return { marks, empty: KINDS.every((k) => marks[k].size === 0) };
    }

    /** The apply succeeded. */
    settle(): void { this.inflight = undefined; }

    /** The apply failed: the taken marks go back UNDER any newer ones (a key marked since keeps its newer mark). */
    restore(taken: TakenRows): void {
        for (const k of KINDS) for (const [id, m] of taken.marks[k]) if (!this.cur[k].has(id)) this.cur[k].set(id, m);
        if (this.inflight === taken.marks) this.inflight = undefined;
    }

    /** Reads the marked rows' CURRENT values from memory. A row marked as an upsert but gone from memory is skipped. */
    toChange(taken: TakenRows, s: RowState): RowChange {
        const out: RowChange = {
            upserts: { scopes: [], facts: [], relations: [], extracted: [], recheck: [] },
            deletes: { scopes: [], facts: [], relations: [], extracted: [], recheck: [] },
        };
        for (const { key, del } of taken.marks.scopes.values()) {
            const v = s.scopes.find((x) => x.rootId === key[0]);
            if (del) out.deletes.scopes.push(key[0]!); else if (v) out.upserts.scopes.push(v);
        }
        for (const { key, del } of taken.marks.facts.values()) {
            const v = s.facts.get(key[0]!)?.get(key[1]!);
            if (del) out.deletes.facts.push([key[0]!, key[1]!]); else if (v) out.upserts.facts.push([key[0]!, v]);
        }
        for (const { key, del } of taken.marks.relations.values()) {
            const v = s.relations.get(key[0]!)?.get(relKey({ from: key[1]!, to: key[2]!, kind: key[3]! }));
            if (del) out.deletes.relations.push([key[0]!, key[1]!, key[2]!, key[3]!]); else if (v) out.upserts.relations.push([key[0]!, v]);
        }
        for (const { key, del } of taken.marks.extracted.values()) {
            const v = s.extracted.get(key[0]!);
            if (del) out.deletes.extracted.push(key[0]!); else if (v) out.upserts.extracted.push([key[0]!, v]);
        }
        for (const { key, del } of taken.marks.recheck.values()) {
            if (del) out.deletes.recheck.push(key[0]!); else if (s.recheck.has(key[0]!)) out.upserts.recheck.push(key[0]!);
        }
        return out;
    }

    /**
     * Lays this store's unwritten rows (the in-flight marks, then the current ones, the current winning) from `old` over `onto`,
     * a state freshly read from the host: an upserted row keeps its in-memory value, a deleted row is removed. `onto.scopes`
     * is edited in place. The caller rebuilds any index derived from the relations.
     */
    overlay(old: RowState, onto: RowState): void {
        for (const marks of [this.inflight, this.cur]) {
            if (!marks) continue;
            for (const { key, del } of marks.scopes.values()) {
                const at = onto.scopes.findIndex((x) => x.rootId === key[0]);
                const v = del ? undefined : old.scopes.find((x) => x.rootId === key[0]);
                if (!del && !v) continue; // an upsert whose row is gone from memory is skipped, as in toChange
                if (at >= 0) onto.scopes.splice(at, 1);
                if (v) onto.scopes.push(v);
            }
            for (const { key, del } of marks.facts.values()) {
                const v = old.facts.get(key[0]!)?.get(key[1]!);
                if (del) onto.facts.get(key[0]!)?.delete(key[1]!); else if (v) rowsOf(onto.facts, key[0]!).set(key[1]!, v);
            }
            for (const { key, del } of marks.relations.values()) {
                const rk = relKey({ from: key[1]!, to: key[2]!, kind: key[3]! });
                const v = old.relations.get(key[0]!)?.get(rk);
                if (del) onto.relations.get(key[0]!)?.delete(rk); else if (v) rowsOf(onto.relations, key[0]!).set(rk, v);
            }
            for (const { key, del } of marks.extracted.values()) {
                const v = old.extracted.get(key[0]!);
                if (del) onto.extracted.delete(key[0]!); else if (v) onto.extracted.set(key[0]!, v);
            }
            for (const { key, del } of marks.recheck.values()) {
                if (del) onto.recheck.delete(key[0]!); else if (old.recheck.has(key[0]!)) onto.recheck.add(key[0]!);
            }
        }
    }
}

function rowsOf<T>(maps: Map<string, Map<string, T>>, rootId: string): Map<string, T> {
    let m = maps.get(rootId);
    if (!m) maps.set(rootId, m = new Map());
    return m;
}
