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
 * Recall ranking and its derived embedding/index caches live in
 * {@link SemanticRecall}; the public shapes in `./types.ts`.
 *
 * Weight save/load is delegated to the SSMRuntime passed to saveWeights/loadWeights.
 */

import type { HybridRetrieveOptions } from '../retrieval/index.js';
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
import { SemanticRecall } from './SemanticRecall.js';
import { isExpired, liveEntries, rememberInput, toEntry, toImportRecord } from './records.js';
import { TtlSweeper } from './ttlSweeper.js';
import type {
    MemoryEntry,
    MemoryStoreOptions,
    RankedRecall,
    RememberOptions,
    SaveLoadRuntime,
    ScoredMemory,
    SSMRuntimeRef,
} from './types.js';

export type {
    FactType,
    MemoryEntry,
    MemoryStoreOptions,
    RankedRecall,
    RecallMethod,
    RememberOptions,
    ScoredMemory,
} from './types.js';

/** Max content→embedding vectors retained (LRU-evicted past this). (EVM-8) */
const EMBED_CACHE_MAX = 2000;

export class MemoryStore {
    private readonly _weightsKey : string;
    private readonly _defaultTtl : number | undefined;
    private readonly _maxEntries : number | undefined;
    private readonly _scope: MemoryScope;
    /** The unit-of-work layer every durable operation flows through. */
    readonly repository: ScopedFactRepository;
    private readonly _storage: FactStoragePort;
    /** Cross-context subscription, opened on the first operation (see `_ready`). */
    private _unsubscribe: (() => void) | undefined;
    private _subscribed = false;
    private readonly _sweeper = new TtlSweeper();
    /** Ranking + its embedding / dense-index caches (EVM-1, EVM-1b, EVM-8). */
    private readonly _recall: SemanticRecall;
    /**
     * Last store generation observed from a FOREIGN context (another tab,
     * window or process). Monotonic, in-memory only; lets readers notice
     * cross-context writes without a storage round-trip per recall (M05).
     */
    private _foreignRevision = 0;

    constructor(opts: MemoryStoreOptions = {}) {
        this._weightsKey = opts.weightsKey ?? 'ssmjs-weights';
        this._defaultTtl = opts.defaultTtlMs;
        this._maxEntries = opts.maxEntries;
        this._recall = new SemanticRecall({
            annThreshold : opts.annThreshold ?? 256,
            embedCacheMax: opts.embedCacheMax ?? EMBED_CACHE_MAX,
            // Opt-in floor: 0 keeps recall returning its best hits, as it always did.
            minRelevance : opts.minRelevance ?? 0,
        });
        this._scope = opts.scope ? { ...opts.scope } : {};
        this._storage = opts.storage ?? new IdbStorageAdapter(opts.dbName ?? 'ssmjs', {
            idbFactory: opts.idbFactory,
            contextId : opts.contextId,
        });
        this.repository = new ScopedFactRepository(this._storage, {
            syncEnabled: opts.sync ?? false,
            contextId  : opts.contextId,
        });
    }

    /**
     * Cross-context invalidation (M05): a sibling tab/window/process that
     * commits against the same storage invalidates this instance's derived
     * caches. Subscribed lazily on the first operation: before it there are no
     * caches to invalidate, and an eager subscription opened a global
     * BroadcastChannel in the constructor that kept Node processes alive.
     */
    private _ready(): void {
        if (this._subscribed) return;
        this._subscribed = true;
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

    private _onForeignChange(note: ConsistencyNote): void {
        if (note.revision <= this._foreignRevision) return;
        this._foreignRevision = note.revision;
        // Derived caches describe a fact set another context just changed.
        this._recall.invalidate();
    }

    // ── Semantic facts ────────────────────────────────────────────────────────

    /**
     * Stores or overwrites a fact. Resolves only after the underlying
     * transaction commits durably (M02); the fact, its provenance and its sync
     * journal entry commit atomically.
     */
    async remember(key: string, content: string, opts?: RememberOptions): Promise<void> {
        this._ready();
        await this.repository.remember(this._scope, key, rememberInput(content, opts, this._defaultTtl));
        // EVM-7: keep the store bounded — evict lowest-value SYNCED entries past
        // the cap; unsynced knowledge is never sacrificed to a cache cap (M06).
        if (this._maxEntries != null) {
            await this.repository.evictForCap(this._scope, this._maxEntries, isExpired);
        }
    }

    /**
     * Compare-and-set write (M04/M05): commits only if the record's current
     * revision equals `expectedRevision` (`null` asserts the key is absent).
     * Throws a MemoryStorageError otherwise (`code` 'MEMORY_CONFLICT',
     * `storageCode` 'STORAGE_CONFLICT'; see `isStorageConflict`) — callers reconcile
     * (re-read, re-evaluate evidence) instead of blindly overwriting a
     * concurrent writer. Returns the committed entry.
     */
    async rememberCas(key: string, content: string, expectedRevision: number | null, opts?: RememberOptions): Promise<MemoryEntry> {
        this._ready();
        const { record } = await this.repository.remember(this._scope, key, {
            ...rememberInput(content, opts, this._defaultTtl),
            expectedRevision,
        });
        return toEntry(record);
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
        this._sweeper.start(intervalMs, () => this.purgeExpired());
    }

    /** Stops the background TTL sweeper started by {@link startTtlSweeper}. */
    stopTtlSweeper(): void {
        this._sweeper.stop();
    }

    /**
     * Retrieves a fact by key from this store's scope partition.
     * Returns `undefined` if the key does not exist or the entry has expired.
     */
    async recall(key: string): Promise<MemoryEntry | undefined> {
        this._ready();
        const record = await this.repository.get(this._scope, key);
        if (!record || isExpired(record)) return undefined;
        return toEntry(record);
    }

    /** Returns all non-expired stored facts in this store's scope, newest first. */
    async recallAll(): Promise<MemoryEntry[]> {
        this._ready();
        return liveEntries(await this.repository.getAll(this._scope));
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
        this._ready();
        return liveEntries(await this.repository.getByTag(this._scope, tag));
    }

    /**
     * Finds the top-K semantically similar entries to `query`: cosine over the
     * attached SSM runtime's embeddings (so recall improves as that model is
     * adapted), or Jaccard word overlap when no runtime can embed. See
     * {@link SemanticRecall}.
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
    async recallSimilarScored(query: string, topK: number, runtime?: SSMRuntimeRef): Promise<ScoredMemory[]> {
        return this._recall.similar(await this.recallAll(), query, topK, runtime);
    }

    /**
     * Hybrid recall: dense (SSM-embedding cosine) and sparse (BM25) rankings
     * fused by Reciprocal Rank Fusion, then an MMR diversity rerank — catches
     * exact-token matches cosine misses, and degrades to BM25 alone without a
     * runtime. The production RAG path.
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
     * {@link recallHybrid} with its evidence: each hit's fused score, and WHICH
     * ranker produced the ordering — `'embedding'` only when the query and the
     * candidates really embedded, `'lexical'` when the ordering is BM25 alone.
     * One blend implementation: the shared `hybridRetrieve` primitive.
     *
     * Each fact's importance and write time are handed to that primitive, so
     * `opts.importanceWeight` / `opts.recencyHalfLifeDays` rank by them (both
     * off by default here; the memory MCP server turns them on for recall).
     */
    async recallRanked(
        query: string,
        topK: number,
        runtime?: SSMRuntimeRef,
        opts?: HybridRetrieveOptions,
    ): Promise<RankedRecall> {
        return this._recall.ranked(await this.recallAll(), query, topK, runtime, opts);
    }

    /**
     * Deletes all entries whose TTL has expired. With sync enabled each expiry
     * commits a tombstone + journal entry in the same transaction, so expiries
     * propagate instead of resurrecting from a replica (M06).
     * @returns The number of entries deleted.
     */
    async purgeExpired(): Promise<number> {
        this._ready();
        return this.repository.purgeExpired(this._scope, isExpired);
    }

    /**
     * Deletes a single fact. No-op if key does not exist. With sync enabled the
     * delete is a durable tombstone + journal entry committed atomically (M04).
     */
    async forget(key: string): Promise<void> {
        this._ready();
        await this.repository.forget(this._scope, key);
    }

    /** Deletes all facts in this store's scope. Does not affect saved weights. */
    async clear(): Promise<void> {
        // Drop the persistent dense index/vectors too (EVM-1b) — they describe a
        // fact set that no longer exists.
        this._ready();
        this._recall.resetDense();
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
        this._unsubscribe = undefined;
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
        this._ready();
        const records = entries.map((e) => toImportRecord(e, this._scope));
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
