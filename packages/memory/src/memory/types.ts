/**
 * memory/types.ts – The public shapes of {@link MemoryStore}: entries, recall
 * results, write options and store options.
 */

import type { FactProvenance, FactStoragePort, MemoryScope } from '../storage/types.js';

export type FactType = 'text' | 'json' | 'number' | 'boolean';

export interface MemoryEntry {
    key        : string;
    content    : string;
    timestamp  : number;
    /**
     * Monotonic write sequence, used only to break `timestamp` ties so that
     * ordering stays deterministic when several entries are written within the
     * same millisecond (Date.now() resolution). Higher = written later.
     * Optional for backward compatibility with entries persisted/imported
     * before this field existed (those sort as seq 0).
     */
    seq?       : number;
    /** Optional time-to-live in milliseconds. */
    ttlMs?     : number;
    /** Semantic type of the stored value. Default: 'text'. */
    type?      : FactType;
    /** Tags for grouping and filtering. */
    tags?      : string[];
    /** Importance weight in the range 0–1. Default: 0.5. */
    importance?: number;
    /**
     * Durable revision of this record (the store generation that wrote it).
     * Present on every record written through the repository; absent only on
     * entries fabricated by callers for import.
     */
    revision?  : number;
    /** Durable identity of the operation that wrote this record. */
    opId?      : string;
    /** Ownership partition this entry was recalled from. */
    scope?     : MemoryScope;
    /** Where the fact came from; preserved across export/import (M06). */
    provenance?: FactProvenance;
    /** True while a sync acknowledgement for this record is outstanding. */
    pendingSync?: boolean;
}

/**
 * Which ranker produced a recall ordering. Matches the vocabulary the gateway's
 * Evermind recall/validate surfaces already report, so a host can render one chip
 * ("Semantic recall" / "Lexical recall (fallback)") regardless of which layer
 * answered.
 */
export type RecallMethod = 'embedding' | 'lexical';

/** A recalled entry with its relevance score (higher = closer). */
export interface ScoredMemory {
    entry: MemoryEntry;
    score: number;
}

/** A ranked recall: the hits plus the ranker that ordered them. */
export interface RankedRecall {
    hits: ScoredMemory[];
    method: RecallMethod;
}

export interface RememberOptions {
    /** Override the store-level defaultTtlMs for this entry. */
    ttlMs?     : number;
    type?      : FactType;
    tags?      : string[];
    importance?: number;
    /** Provenance to record with the fact (surface, importer, source time). */
    provenance?: FactProvenance;
    /** Explicit source timestamp (import/replay). Default: now. */
    timestamp? : number;
}

export interface MemoryStoreOptions {
    /** IndexedDB database name. Default: 'ssmjs'. Ignored when `storage` is supplied. */
    dbName?       : string;
    /** Key used for weight storage within the 'weights' object store. Default: 'ssmjs-weights'. */
    weightsKey?   : string;
    /**
     * IDBFactory to use instead of the global `indexedDB` (browser path only).
     * fake-indexeddb is supported for tests, but it is IN-MEMORY: on Node.js,
     * pass `storage` from `NodeFileStorageAdapter.open(...)` for durable facts.
     */
    idbFactory?   : IDBFactory;
    /**
     * Durable storage port backing this store (M01). When omitted, an
     * IndexedDB adapter over `dbName` is used.
     */
    storage?      : FactStoragePort;
    /**
     * Explicit ownership partition for every fact in this store (M03).
     * Default: the unbound scope — personal, never-connected memory.
     */
    scope?        : MemoryScope;
    /**
     * Enable the sync journal, tombstones and pendingSync tracking (M04/M06).
     * Unsynced facts are then never evicted by `maxEntries`. Default: false
     * (legacy retention semantics — nothing to sync, nothing pending).
     */
    sync?         : boolean;
    /** Identity of this context for operation ids and cross-context notes. */
    contextId?    : string;
    /**
     * Default TTL applied to new entries when no per-entry ttlMs is provided.
     * Entries with an expired TTL are filtered from recallAll() and related methods.
     */
    defaultTtlMs? : number;
    /**
     * Fact count at/above which semantic recall switches from an exact O(N)
     * cosine scan to an O(log N) HNSW ANN index. Default 256 — small stores stay
     * exact (simpler + faster), large stores stay fast. (EVM-1)
     */
    annThreshold? : number;
    /**
     * Hard cap on stored facts. When set, each write evicts the lowest-value
     * SYNCED entries (expired first, then lowest importance, then oldest) so the
     * store stays bounded. Facts with unsynced operations are never evicted
     * (M06). Unset = unbounded (legacy). (EVM-7)
     */
    maxEntries?   : number;
    /**
     * Max content→embedding vectors held in the LRU recall cache. Default 2000.
     * (EVM-8 — bounded LRU instead of clear-on-overflow.)
     */
    embedCacheMax?: number;
    /**
     * Minimum relevance (0..1) a hit must reach to be recalled by
     * `recallSimilarScored` / `recallRanked` (and their wrappers). Below it, a
     * weak match returns nothing rather than noise. Default 0 (off): recall
     * returns its best-ranked hits exactly as before this option existed. Opt in
     * with e.g. 0.05 to drop zero-overlap matches (a query sharing no token with
     * any fact scores 0 on the lexical path).
     */
    minRelevance? : number;
}

// Minimal interface to avoid importing SSMRuntime (circular dep)
export interface SaveLoadRuntime {
    save(opts?: { storage: 'indexedDB'; key: string }): Promise<void>;
    load(opts?: { key: string }): Promise<boolean>;
}

// Forward-declared to avoid circular dependency at import time
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type SSMRuntimeRef = any;
