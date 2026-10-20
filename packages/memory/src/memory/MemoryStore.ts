/**
 * MemoryStore – persistent key-value fact store and weight checkpoint helper.
 *
 * Persistence is delegated to a narrow {@link FactStoragePort} (M01) fronted by
 * a {@link ScopedFactRepository} unit-of-work layer (M02–M06):
 *   - Browser default: IndexedDB via {@link IdbStorageAdapter}; every write
 *     resolves only after the transaction commits (M02).
 *   - Node.js: pass `storage: await NodeFileStorageAdapter.open(path)` for real
 *     durable storage. fake-indexeddb via `idbFactory` remains supported for
 *     TESTS, but is never durable and is not presented as such.
 *   - Facts live in an explicit {@link MemoryScope} partition (default: the
 *     unbound, never-connected scope) with durable revisions, operation
 *     identity, provenance and — when `sync: true` — a journaled tombstone
 *     trail that caps/TTL will not sacrifice (M03/M04/M06).
 *
 * Weight save/load is delegated to the SSMRuntime passed to saveWeights/loadWeights.
 */

import { SSMError } from '../errors/SSMError.js';
import { tokenize, jaccardSimilarity } from '../similarity/index.js';
import { hybridRetrieve, denseSearch, HnswIndex, type RetrievalCandidate, type HybridRetrieveOptions } from '../retrieval/index.js';
import { IdbStorageAdapter } from '../storage/idbAdapter.js';
import { ScopedFactRepository } from '../storage/ScopedFactRepository.js';
import {
    MemoryStorageError,
    type ConsistencyNote,
    type FactProvenance,
    type FactStoragePort,
    type MemoryScope,
    type StoredFactRecord,
    type TombstoneRecord,
    type JournalEntry,
} from '../storage/types.js';

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
     * {@link recallSimilarScored} or {@link recallRanked}. Below it, a weak match
     * returns nothing rather than noise — recall that always returns something is
     * how an unrelated fact gets surfaced as if it were relevant. Default 0.05:
     * an empty query or a query sharing no token with any fact scores 0 and is
     * dropped, while a genuine overlap clears it. Pass 0 to disable.
     */
    minRelevance? : number;
}

// Minimal interface to avoid importing SSMRuntime (circular dep)
interface SaveLoadRuntime {
    save(opts?: { storage: 'indexedDB'; key: string }): Promise<void>;
    load(opts?: { key: string }): Promise<boolean>;
}

// Forward-declared to avoid circular dependency at import time
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type SSMRuntimeRef = any;

/** Max content→embedding vectors retained (LRU-evicted past this). (EVM-8) */
const EMBED_CACHE_MAX = 2000;

export class MemoryStore {
    private readonly _weightsKey : string;
    private readonly _defaultTtl : number | undefined;
    private readonly _annThreshold : number;
    private readonly _maxEntries : number | undefined;
    private readonly _embedCacheMax : number;
    /** Below this a recall hit is noise, not a match. 0 disables the floor. */
    private readonly _minRelevance : number;
    private readonly _scope: MemoryScope;
    /** The unit-of-work layer every durable operation flows through. */
    readonly repository: ScopedFactRepository;
    private readonly _storage: FactStoragePort;
    private readonly _unsubscribe: (() => void) | undefined;
    private _sweepTimer: ReturnType<typeof setInterval> | null = null;
    /** Content → L2-normalised embedding cache, used by recallSimilar. */
    private readonly _embedCache = new Map<string, Float32Array>();
    /** Runtime whose embeddings populate `_embedCache`; a change invalidates it. */
    private _embedRuntime: unknown = null;
    /**
     * Last store generation observed from a FOREIGN context (another tab,
     * window or process). Monotonic, in-memory only; lets readers notice
     * cross-context writes without a storage round-trip per recall (M05).
     */
    private _foreignRevision = 0;

    // ── Persistent dense index (EVM-1b) ────────────────────────────────────────
    // Fact embeddings keyed by `${key}@${seq}` so an EDIT (seq bump) is a new key
    // and an unchanged fact is never re-embedded. The HNSW index is derived from
    // these vectors and reused across recalls until the fact set changes — so
    // steady-state recall is O(1) query-embed + O(log N) search, not O(N) embed.
    private readonly _denseVectors = new Map<string, Float32Array>();
    private _denseIndex: HnswIndex | null = null;
    /** Signature of the fact set the current index was built from. */
    private _denseSig = '';
    /** Identity of the runtime whose embeddings populated the vectors. */
    private _denseRuntime: unknown = null;

    constructor(opts: MemoryStoreOptions = {}) {
        this._weightsKey = opts.weightsKey ?? 'ssmjs-weights';
        this._defaultTtl = opts.defaultTtlMs;
        this._annThreshold = opts.annThreshold ?? 256;
        this._maxEntries = opts.maxEntries;
        this._embedCacheMax = opts.embedCacheMax ?? EMBED_CACHE_MAX;
        this._minRelevance = opts.minRelevance ?? 0.05;
        this._scope = opts.scope ? { ...opts.scope } : {};
        this._storage = opts.storage ?? new IdbStorageAdapter(opts.dbName ?? 'ssmjs', {
            idbFactory: opts.idbFactory,
            contextId : opts.contextId,
        });
        this.repository = new ScopedFactRepository(this._storage, {
            syncEnabled: opts.sync ?? false,
            contextId  : opts.contextId,
        });

        // Cross-context invalidation (M05): a sibling tab/window/process that
        // commits against the same storage invalidates this instance's derived
        // caches — in-process serialization alone is not the consistency boundary.
        this._unsubscribe = this.repository.subscribe((note) => this._onForeignChange(note));
    }

    /** The scope partition this store is bound to (never widened implicitly). */
    get scope(): MemoryScope {
        return { ...this._scope };
    }

    /** The durable storage backend kind in use ('indexedDB' | 'node-file'). */
    get storageKind(): FactStoragePort['kind'] {
        return this._storage.kind;
    }

    /**
     * Best-effort in-memory token that changes when another context writes to
     * this store's database. Readers (e.g. EvermindCognition's recall cache)
     * can fold it into their cache key so cross-context writes invalidate
     * without a storage hit per recall.
     */
    get foreignRevision(): number {
        return this._foreignRevision;
    }

    // ── TTL helpers ───────────────────────────────────────────────────────────

    private _isExpired(entry: { timestamp: number; ttlMs?: number }): boolean {
        if (entry.ttlMs == null) return false;
        return Date.now() > entry.timestamp + entry.ttlMs;
    }

    private _toEntry(record: StoredFactRecord): MemoryEntry {
        return {
            key        : record.key,
            content    : record.content,
            timestamp  : record.timestamp,
            seq        : record.seq,
            ttlMs      : record.ttlMs,
            type       : record.type,
            tags       : record.tags,
            importance : record.importance,
            revision   : record.revision,
            opId       : record.opId,
            scope      : { ...record.scope },
            provenance : record.provenance,
            pendingSync: record.pendingSync,
        };
    }

    private _onForeignChange(note: ConsistencyNote): void {
        if (note.revision <= this._foreignRevision) return;
        this._foreignRevision = note.revision;
        // Derived caches describe a fact set another context just changed.
        this._embedCache.clear();
        this._denseVectors.clear();
        this._denseIndex = null;
        this._denseSig = '';
    }

    // ── Semantic facts ────────────────────────────────────────────────────────

    /**
     * Stores or overwrites a fact. Resolves only after the underlying
     * transaction commits durably (M02); the fact, its provenance and its sync
     * journal entry commit atomically.
     */
    async remember(key: string, content: string, opts?: RememberOptions): Promise<void> {
        await this.repository.remember(this._scope, key, {
            content,
            timestamp  : opts?.timestamp,
            ttlMs      : opts?.ttlMs ?? this._defaultTtl,
            type       : opts?.type,
            tags       : opts?.tags,
            importance : opts?.importance,
            provenance : opts?.provenance,
        });
        // EVM-7: keep the store bounded — evict lowest-value SYNCED entries past
        // the cap; unsynced knowledge is never sacrificed to a cache cap (M06).
        if (this._maxEntries != null) {
            await this.repository.evictForCap(this._scope, this._maxEntries, (r) => this._isExpired(r));
        }
    }

    /**
     * Compare-and-set write (M04/M05): commits only if the record's current
     * revision equals `expectedRevision` (`null` asserts the key is absent).
     * Throws a STORAGE_CONFLICT MemoryStorageError otherwise — callers reconcile
     * (re-read, re-evaluate evidence) instead of blindly overwriting a
     * concurrent writer. Returns the committed entry.
     */
    async rememberCas(key: string, content: string, expectedRevision: number | null, opts?: RememberOptions): Promise<MemoryEntry> {
        const { record } = await this.repository.remember(this._scope, key, {
            content,
            timestamp       : opts?.timestamp,
            ttlMs           : opts?.ttlMs ?? this._defaultTtl,
            type            : opts?.type,
            tags            : opts?.tags,
            importance      : opts?.importance,
            provenance      : opts?.provenance,
            expectedRevision,
        });
        return this._toEntry(record);
    }

    /** Fetches an entry WITH its durable revision, for CAS-based reconciliation. */
    async recallVersioned(key: string): Promise<MemoryEntry | undefined> {
        return this.recall(key);
    }

    /**
     * Starts a background timer that periodically deletes expired entries —
     * TTL is otherwise only evaluated lazily on read, so an idle store would keep
     * dead entries forever. Idempotent; pair with {@link stopTtlSweeper}. (EVM-7)
     *
     * With sync enabled each expiry is a scoped deletion (tombstone + journal),
     * not a silent drop (M06).
     */
    startTtlSweeper(intervalMs: number): void {
        if (this._sweepTimer) return;
        const timer = setInterval(() => { void this.purgeExpired(); }, intervalMs);
        // Don't keep a Node process alive just for the sweeper.
        const maybeUnref = timer as unknown as { unref?: () => void };
        /* istanbul ignore else -- unref exists in Node; absent in the browser */
        if (typeof maybeUnref.unref === 'function') {
            maybeUnref.unref();
        }
        this._sweepTimer = timer;
    }

    /** Stops the background TTL sweeper started by {@link startTtlSweeper}. */
    stopTtlSweeper(): void {
        if (this._sweepTimer) {
            clearInterval(this._sweepTimer);
            this._sweepTimer = null;
        }
    }

    /**
     * Retrieves a fact by key from this store's scope partition.
     * Returns `undefined` if the key does not exist or the entry has expired.
     */
    async recall(key: string): Promise<MemoryEntry | undefined> {
        const record = await this.repository.get(this._scope, key);
        if (!record || this._isExpired(record)) return undefined;
        return this._toEntry(record);
    }

    /** Returns all non-expired stored facts in this store's scope, newest first. */
    async recallAll(): Promise<MemoryEntry[]> {
        const records = await this.repository.getAll(this._scope);
        return records
            .filter(e => !this._isExpired(e))
            // Newest first; break same-millisecond ties by the durable write
            // sequence so ordering is deterministic (and survives restarts).
            .sort((a, b) => b.timestamp - a.timestamp || b.seq - a.seq)
            .map((r) => this._toEntry(r));
    }

    /**
     * Returns the N most recently updated non-expired facts.
     * Equivalent to `recallAll()` truncated to `n` entries.
     */
    async recallRecent(n: number): Promise<MemoryEntry[]> {
        const all = await this.recallAll();
        return all.slice(0, n);
    }

    /**
     * Returns all non-expired entries that contain the given tag.
     * Looks the tag up on the facts `byTag` index when the storage adapter has
     * one; a full scan is used only when that index is missing.
     */
    async recallByTag(tag: string): Promise<MemoryEntry[]> {
        const records = await this.repository.getByTag(this._scope, tag);
        return records
            .filter(e => !this._isExpired(e))
            .sort((a, b) => b.timestamp - a.timestamp || b.seq - a.seq)
            .map((r) => this._toEntry(r));
    }

    /**
     * Finds the top-K semantically similar entries to `query`.
     *
     * When `runtime` exposes an `embed()` method (the SSMRuntime does), similarity
     * is computed as cosine distance between SSM hidden-state embeddings — i.e. the
     * memory layer uses the very model it is attached to, and recall quality
     * improves automatically as that model is adapted/distilled. Embeddings are
     * cached per content string to avoid recomputing across calls.
     *
     * If no embedding-capable runtime is provided, or embedding fails for any
     * reason, it transparently falls back to Jaccard word-overlap similarity.
     */
    async recallSimilar(query: string, topK: number, runtime?: SSMRuntimeRef): Promise<MemoryEntry[]> {
        return (await this.recallSimilarScored(query, topK, runtime)).map(s => s.entry);
    }

    /**
     * Like {@link recallSimilar} but returns each hit's real 0..1 relevance score
     * (SSM-embedding cosine, or Jaccard on the fallback path) alongside the entry —
     * so callers can rank/threshold by TRUE similarity instead of list position.
     * {@link recallSimilar} is the score-dropping convenience wrapper over this.
     */
    async recallSimilarScored(query: string, topK: number, runtime?: SSMRuntimeRef): Promise<Array<{ entry: MemoryEntry; score: number }>> {
        const all = await this.recallAll();
        if (all.length === 0) return [];

        // ── Preferred path: SSM-embedding cosine similarity ───────────────────
        if (runtime != null && typeof runtime.embed === 'function') {
            const queryVec = await this._embedWithCache(runtime, query);
            if (queryVec) {
                // EVM-1b: sync the persistent vector store + index (embeds only the
                // delta), then query it — no per-call re-embed of every fact.
                const synced = await this._syncDense(runtime, all);
                if (synced) {
                    const hits = synced.index
                        // EVM-1: above the threshold, query the persistent HNSW (O(log N)).
                        ? synced.index.search(queryVec, topK)
                        // Below it, an exact scan over the (already-embedded) vectors.
                        : denseSearch(queryVec, synced.items, topK, this._annThreshold);
                    return hits
                        .map(h => { const entry = synced.byKey.get(h.id); return entry ? { entry, score: h.score } : null; })
                        .filter((e): e is { entry: MemoryEntry; score: number } => !!e)
                        .filter(e => e.score >= this._minRelevance);
                }
            }
            // Any failure falls through to the Jaccard path below.
        }

        // ── Fallback: Jaccard word-overlap similarity ─────────────────────────
        const queryTokens = new Set(tokenize(query));
        const scored = all.map(entry => {
            const entryTokens = new Set(tokenize(entry.content));
            const score       = jaccardSimilarity(queryTokens, entryTokens);
            return { entry, score };
        });

        scored.sort((a, b) => b.score - a.score);
        return scored.filter(s => s.score >= this._minRelevance).slice(0, topK);
    }

    /**
     * Hybrid recall: fuses dense (SSM-embedding cosine) and sparse (BM25 lexical)
     * rankings via Reciprocal Rank Fusion, then applies an MMR diversity rerank.
     *
     * This is the production RAG retrieval path — it catches both semantic matches
     * (embeddings) and exact-token matches (BM25 — identifiers, codes, rare names)
     * that cosine-only `recallSimilar` misses, and avoids returning near-duplicate
     * facts. Degrades to BM25-only when no embedding-capable runtime is available,
     * so it is always strictly at least as good as the lexical fallback.
     */
    async recallHybrid(
        query: string,
        topK: number,
        runtime?: SSMRuntimeRef,
        opts?: HybridRetrieveOptions,
    ): Promise<MemoryEntry[]> {
        return (await this.recallRanked(query, topK, runtime, opts)).hits.map(h => h.entry);
    }

    /**
     * The hybrid retrieval path with its evidence attached: each hit's fused score,
     * plus WHICH ranker produced the ordering.
     *
     * {@link recallHybrid} is the convenience wrapper that throws both away, so
     * there is exactly one implementation of the blend — and it is the shared
     * `hybridRetrieve` → `reciprocalRankFusion` primitive, never a second hand-
     * rolled mix of cosine and word overlap.
     *
     * `method` is honest rather than aspirational: `'embedding'` means the query
     * really did embed and the dense ranking really did participate in the fusion;
     * `'lexical'` means no model was available (or embedding failed) and the
     * ordering is BM25 alone. Callers surface it so a user can tell semantic
     * recall from the fallback instead of guessing.
     */
    async recallRanked(
        query: string,
        topK: number,
        runtime?: SSMRuntimeRef,
        opts?: HybridRetrieveOptions,
    ): Promise<RankedRecall> {
        const all = await this.recallAll();
        if (all.length === 0) return { hits: [], method: 'lexical' };

        // Embed candidates + query where a runtime is available; null vectors are
        // fine — hybridRetrieve degrades that candidate to BM25-only.
        const canEmbed = runtime != null && typeof runtime.embed === 'function';
        const queryVec = canEmbed ? (await this._embedWithCache(runtime, query)) ?? undefined : undefined;

        const candidates: RetrievalCandidate[] = [];
        for (const entry of all) {
            const vector = queryVec ? (await this._embedWithCache(runtime, entry.content)) ?? undefined : undefined;
            candidates.push({ id: entry.key, text: entry.content, vector });
        }

        // The dense ranking only exists when the query embedded AND at least one
        // candidate did — anything less is a lexical ordering wearing a label.
        const dense = !!queryVec && candidates.some(c => !!c.vector);
        const hits = hybridRetrieve({ text: query, vector: queryVec }, candidates, { topK, annThreshold: this._annThreshold, ...opts });
        const byKey = new Map(all.map(e => [e.key, e]));
        return {
            hits: hits
                .map(h => { const entry = byKey.get(h.id); return entry ? { entry, score: h.score } : null; })
                .filter((h): h is ScoredMemory => !!h)
                .filter(h => h.score >= this._minRelevance),
            method: dense ? 'embedding' : 'lexical',
        };
    }

    /**
     * Returns a cached embedding for `text`, computing it via `runtime.embed()`
     * on a cache miss. Returns `null` (never throws) when embedding is
     * unavailable so callers can fall back to lexical similarity.
     */
    private async _embedWithCache(runtime: SSMRuntimeRef, text: string): Promise<Float32Array | null> {
        // Embeddings are model-specific: if the runtime instance changed, the cached
        // vectors are from a different model and must be dropped. (EVM-1b)
        if (runtime !== this._embedRuntime) {
            this._embedCache.clear();
            this._embedRuntime = runtime;
        }
        const cached = this._embedCache.get(text);
        if (cached) {
            // LRU touch: re-insert so this entry becomes most-recently-used. (EVM-8)
            this._embedCache.delete(text);
            this._embedCache.set(text, cached);
            return cached;
        }
        try {
            const vec = await runtime.embed(text);
            if (vec instanceof Float32Array && vec.length > 0) {
                // EVM-8: bound cache growth by evicting the LEAST-recently-used entry
                // (Map keeps insertion order; the first key is the LRU), instead of
                // dumping the whole cache on every overflow.
                if (this._embedCache.size >= this._embedCacheMax) {
                    const lru = this._embedCache.keys().next().value;
                    if (lru !== undefined) this._embedCache.delete(lru);
                }
                this._embedCache.set(text, vec);
                return vec;
            }
        } catch {
            // Embedding unavailable (no GPU, destroyed runtime, etc.) — signal fallback.
        }
        return null;
    }

    /**
     * Synchronise the persistent dense vector store + HNSW index to `all` (EVM-1b).
     * Embeds ONLY new/changed facts (keyed by `key@seq`), prunes removed ones, and
     * rebuilds the index only when the fact set actually changed — so a read-heavy
     * workload reuses both the embeddings and the index across recalls. Returns
     * `null` (caller falls back to lexical) if any embedding is unavailable, or
     * resets everything when the runtime identity changes (embeddings are
     * model-specific).
     */
    private async _syncDense(
        runtime: SSMRuntimeRef,
        all: MemoryEntry[],
    ): Promise<{ byKey: Map<string, MemoryEntry>; items: Array<{ id: string; vector: Float32Array }>; index: HnswIndex | null } | null> {
        if (runtime !== this._denseRuntime) {
            this._denseVectors.clear();
            this._denseIndex = null;
            this._denseSig = '';
            this._denseRuntime = runtime;
        }

        const vkeyOf = (e: MemoryEntry): string => `${e.key}@${e.seq ?? 0}`;
        const currentVkeys = new Set<string>();
        let sig = `${all.length}`;
        for (const e of all) { const k = vkeyOf(e); currentVkeys.add(k); sig += '|' + k; }

        const byKey = new Map<string, MemoryEntry>();
        const items: Array<{ id: string; vector: Float32Array }> = [];
        for (const entry of all) {
            const vk = vkeyOf(entry);
            let vec = this._denseVectors.get(vk);
            if (!vec) {
                const v = await this._embedWithCache(runtime, entry.content);
                if (!v) return null; // embedding failed → lexical fallback
                vec = v;
                this._denseVectors.set(vk, vec);
            }
            items.push({ id: entry.key, vector: vec });
            byKey.set(entry.key, entry);
        }

        // Drop vectors for facts that are gone or were edited (stale vkeys).
        if (this._denseVectors.size > currentVkeys.size) {
            for (const k of this._denseVectors.keys()) {
                if (!currentVkeys.has(k)) this._denseVectors.delete(k);
            }
        }

        // Maintain the ANN index only above the threshold; (re)build it only when
        // the fact set changed (no embedding involved — purely in-memory).
        if (all.length >= this._annThreshold) {
            if (sig !== this._denseSig || this._denseIndex === null) {
                const index = new HnswIndex();
                for (const it of items) index.add(it.id, it.vector);
                this._denseIndex = index;
            }
        } else {
            this._denseIndex = null;
        }
        this._denseSig = sig;
        return { byKey, items, index: this._denseIndex };
    }

    /**
     * Deletes all entries whose TTL has expired. With sync enabled each expiry
     * commits a tombstone + journal entry in the same transaction, so expiries
     * propagate instead of resurrecting from a replica (M06).
     * @returns The number of entries deleted.
     */
    async purgeExpired(): Promise<number> {
        return this.repository.purgeExpired(this._scope, (r) => this._isExpired(r));
    }

    /**
     * Deletes a single fact. No-op if key does not exist. With sync enabled the
     * delete is a durable tombstone + journal entry committed atomically (M04).
     */
    async forget(key: string): Promise<void> {
        await this.repository.forget(this._scope, key);
    }

    /** Deletes all facts in this store's scope. Does not affect saved weights. */
    async clear(): Promise<void> {
        // Drop the persistent dense index/vectors too (EVM-1b) — they describe a
        // fact set that no longer exists.
        this._denseVectors.clear();
        this._denseIndex = null;
        this._denseSig = '';
        await this.repository.clear(this._scope);
    }

    // ── Sync protocol surface (M04) ───────────────────────────────────────────

    /** Operations not yet acknowledged by sync — the pending knowledge set. */
    async pendingOps(): Promise<JournalEntry[]> {
        return this.repository.pendingOps();
    }

    /** Delete tombstones in this store's scope (or all scopes when omitted). */
    async tombstones(scope?: MemoryScope): Promise<TombstoneRecord[]> {
        return this.repository.tombstones(scope);
    }

    /**
     * Acknowledges that the given operations reached the sync target. Their
     * facts/tombstones stop being `pendingSync`, so retention may touch them
     * again. Retention NEVER touches unacknowledged knowledge (M06).
     */
    async markSynced(opIds: string[]): Promise<void> {
        await this.repository.markSynced(opIds);
    }

    /** The current durable store generation. */
    async revision(): Promise<number> {
        return this.repository.revision();
    }

    /** Releases cross-context subscriptions and the underlying connection. */
    close(): void {
        this.stopTtlSweeper();
        this._unsubscribe?.();
        this._storage.close?.();
    }

    // ── Cross-session memory merge ────────────────────────────────────────────

    /**
     * Returns all non-expired facts as a plain array for export.
     * Scope, revision and provenance travel with each entry (M03).
     */
    async exportAll(): Promise<MemoryEntry[]> {
        return this.recallAll();
    }

    /**
     * Imports entries from an external array INTO this store's scope partition.
     * Source timestamps, sequences, tags, importance, TTL and provenance are
     * preserved — import never rewrites source time (M06).
     *
     * - `'merge'`     : only writes an entry if no existing entry with the same
     *                   key exists, or if the incoming entry has a newer timestamp.
     * - `'overwrite'` : writes all entries unconditionally.
     */
    async importAll(entries: MemoryEntry[], strategy: 'merge' | 'overwrite', importedFrom?: string): Promise<void> {
        const records = entries.map((e): StoredFactRecord => ({
            id         : '', // reassigned by the repository for this scope
            key        : e.key,
            scope      : this._scope,
            content    : e.content,
            timestamp  : e.timestamp,
            seq        : e.seq ?? 0,
            revision   : 0,
            opId       : e.opId ?? '',
            ttlMs      : e.ttlMs,
            type       : e.type,
            tags       : e.tags,
            importance : e.importance,
            provenance : e.provenance,
            pendingSync: false,
        }));
        await this.repository.importRecords(this._scope, records, strategy, importedFrom);
    }

    // ── Weight persistence ────────────────────────────────────────────────────

    /**
     * Saves SSM weights via `runtime.save()`.
     * The weights are stored under `weightsKey` in the session's IndexedDB,
     * separate from MambaSession's own key.
     */
    async saveWeights(runtime: SaveLoadRuntime): Promise<void> {
        await runtime.save({ storage: 'indexedDB', key: this._weightsKey });
    }

    /**
     * Loads SSM weights via `runtime.load()`.
     * Returns `false` when no saved weights exist under `weightsKey`.
     */
    async loadWeights(runtime: SaveLoadRuntime): Promise<boolean> {
        return runtime.load({ key: this._weightsKey });
    }
}

// Re-export the storage seam so consumers reach it through the store's module.
export {
    IdbStorageAdapter,
    ScopedFactRepository,
    MemoryStorageError,
};
export type {
    ConsistencyNote,
    FactProvenance,
    FactStoragePort,
    JournalEntry,
    MemoryScope,
    StoredFactRecord,
    TombstoneRecord,
};
