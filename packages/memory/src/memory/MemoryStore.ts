/**
 * MemoryStore – persistent key-value fact store and weight checkpoint helper.
 *
 * Uses IndexedDB with a dedicated 'ssmjs' database containing two object stores:
 *   - 'facts'   : MemoryEntry records keyed by the fact key string
 *   - 'weights' : a single ArrayBuffer keyed by `weightsKey`
 *
 * Weight save/load is delegated to the SSMRuntime passed to saveWeights/loadWeights.
 */

import { SSMError } from '../errors/SSMError.js';
import { tokenize, jaccardSimilarity, cosineSimilarity } from '../similarity/index.js';

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
}

export interface RememberOptions {
    /** Override the store-level defaultTtlMs for this entry. */
    ttlMs?     : number;
    type?      : FactType;
    tags?      : string[];
    importance?: number;
}

export interface MemoryStoreOptions {
    /** IndexedDB database name. Default: 'ssmjs'. */
    dbName?       : string;
    /** Key used for weight storage within the 'weights' object store. Default: 'ssmjs-weights'. */
    weightsKey?   : string;
    /**
     * IDBFactory to use instead of the global `indexedDB`.
     * Use this in Node.js environments with fake-indexeddb:
     *   import { IDBFactory } from 'fake-indexeddb';
     *   const idbFactory = new IDBFactory();
     */
    idbFactory?   : IDBFactory;
    /**
     * Default TTL applied to new entries when no per-entry ttlMs is provided.
     * Entries with an expired TTL are filtered from recallAll() and related methods.
     */
    defaultTtlMs? : number;
}

const FACTS_STORE   = 'facts';
const WEIGHTS_STORE = 'weights';
const DB_VERSION    = 1;

/**
 * Process-wide monotonic counter stamped onto each remembered entry. Strictly
 * increasing per write regardless of store instance, so it reliably breaks
 * `timestamp` ties (same-millisecond writes) in newest-first ordering.
 */
let _writeSeq = 0;

// Minimal interface to avoid importing SSMRuntime (circular dep)
interface SaveLoadRuntime {
    save(opts?: { storage: 'indexedDB'; key: string }): Promise<void>;
    load(opts?: { key: string }): Promise<boolean>;
}

// Forward-declared to avoid circular dependency at import time
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type SSMRuntimeRef = any;

/** Max number of cached content→embedding vectors retained in memory. */
const EMBED_CACHE_MAX = 2000;

export class MemoryStore {
    private readonly _dbName     : string;
    private readonly _weightsKey : string;
    private readonly _idb        : IDBFactory | undefined;
    private readonly _defaultTtl : number | undefined;
    private _db: IDBDatabase | null = null;
    /** Content → L2-normalised embedding cache, used by recallSimilar. */
    private readonly _embedCache = new Map<string, Float32Array>();

    constructor(opts: MemoryStoreOptions = {}) {
        this._dbName     = opts.dbName     ?? 'ssmjs';
        this._weightsKey = opts.weightsKey ?? 'ssmjs-weights';
        this._idb        = opts.idbFactory;
        this._defaultTtl = opts.defaultTtlMs;
    }

    // ── Internal DB open ──────────────────────────────────────────────────────

    private _open(): Promise<IDBDatabase> {
        if (this._db) return Promise.resolve(this._db);

        return new Promise((resolve, reject) => {
            const factory = this._idb ?? (typeof indexedDB !== 'undefined' ? indexedDB : undefined);
            if (!factory) {
                reject(new SSMError(
                    'MEMORY_UNAVAILABLE',
                    'IndexedDB is not available in this environment. Pass an idbFactory option (e.g. from fake-indexeddb) for Node.js support.',
                ));
                return;
            }

            const req = factory.open(this._dbName, DB_VERSION);

            req.onupgradeneeded = (e) => {
                const db = (e.target as IDBOpenDBRequest).result;
                if (!db.objectStoreNames.contains(FACTS_STORE)) {
                    db.createObjectStore(FACTS_STORE, { keyPath: 'key' });
                }
                if (!db.objectStoreNames.contains(WEIGHTS_STORE)) {
                    db.createObjectStore(WEIGHTS_STORE);
                }
            };

            req.onsuccess = () => {
                this._db = req.result;
                resolve(req.result);
            };
            /* istanbul ignore next -- IDB open onerror fires only on storage faults; not reproducible with fake-indexeddb */
            req.onerror = () => reject(new SSMError(
                'MEMORY_UNAVAILABLE',
                `Failed to open IndexedDB "${this._dbName}": ${req.error?.message ?? 'unknown'}`,
                req.error,
            ));
        });
    }

    // ── TTL helpers ───────────────────────────────────────────────────────────

    private _isExpired(entry: MemoryEntry): boolean {
        if (entry.ttlMs == null) return false;
        return Date.now() > entry.timestamp + entry.ttlMs;
    }

    // ── Semantic facts ────────────────────────────────────────────────────────

    /** Stores or overwrites a fact. */
    async remember(key: string, content: string, opts?: RememberOptions): Promise<void> {
        const db = await this._open();
        const entry: MemoryEntry = {
            key,
            content,
            timestamp  : Date.now(),
            seq        : ++_writeSeq,
            ttlMs      : opts?.ttlMs ?? this._defaultTtl,
            type       : opts?.type,
            tags       : opts?.tags,
            importance : opts?.importance,
        };

        const tx = db.transaction(FACTS_STORE, 'readwrite');
        return requestToPromise(tx.objectStore(FACTS_STORE).put(entry), `Failed to store fact "${key}"`, () => undefined);
    }

    /**
     * Retrieves a fact by key.
     * Returns `undefined` if the key does not exist or the entry has expired.
     */
    async recall(key: string): Promise<MemoryEntry | undefined> {
        const db = await this._open();

        const tx = db.transaction(FACTS_STORE, 'readonly');
        return requestToPromise(
            tx.objectStore(FACTS_STORE).get(key) as IDBRequest<MemoryEntry | undefined>,
            `Failed to recall fact "${key}"`,
            (entry) => (entry && this._isExpired(entry) ? undefined : entry),
        );
    }

    /** Returns all non-expired stored facts, newest first. */
    async recallAll(): Promise<MemoryEntry[]> {
        const db = await this._open();

        const tx = db.transaction(FACTS_STORE, 'readonly');
        return requestToPromise(
            tx.objectStore(FACTS_STORE).getAll() as IDBRequest<MemoryEntry[]>,
            'Failed to recall all facts',
            (entries) => entries
                .filter(e => !this._isExpired(e))
                // Newest first; break same-millisecond ties by write sequence so
                // ordering is deterministic (IndexedDB getAll() returns key order,
                // which would otherwise surface same-ms writes oldest-first).
                .sort((a, b) => b.timestamp - a.timestamp || (b.seq ?? 0) - (a.seq ?? 0)),
        );
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
     */
    async recallByTag(tag: string): Promise<MemoryEntry[]> {
        const all = await this.recallAll();
        return all.filter(e => e.tags?.includes(tag) ?? false);
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
        const all = await this.recallAll();
        if (all.length === 0) return [];

        // ── Preferred path: SSM-embedding cosine similarity ───────────────────
        if (runtime != null && typeof runtime.embed === 'function') {
            const queryVec = await this._embedWithCache(runtime, query);
            if (queryVec) {
                const scored: { entry: MemoryEntry; score: number }[] = [];
                let embeddedAll = true;
                for (const entry of all) {
                    const entryVec = await this._embedWithCache(runtime, entry.content);
                    if (!entryVec) { embeddedAll = false; break; }
                    scored.push({ entry, score: cosineSimilarity(queryVec, entryVec) });
                }
                if (embeddedAll) {
                    scored.sort((a, b) => b.score - a.score);
                    return scored.slice(0, topK).map(s => s.entry);
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
        return scored.slice(0, topK).map(s => s.entry);
    }

    /**
     * Returns a cached embedding for `text`, computing it via `runtime.embed()`
     * on a cache miss. Returns `null` (never throws) when embedding is
     * unavailable so callers can fall back to lexical similarity.
     */
    private async _embedWithCache(runtime: SSMRuntimeRef, text: string): Promise<Float32Array | null> {
        const cached = this._embedCache.get(text);
        if (cached) return cached;
        try {
            const vec = await runtime.embed(text);
            if (vec instanceof Float32Array && vec.length > 0) {
                // Bound cache growth — simplest eviction is a full clear.
                if (this._embedCache.size >= EMBED_CACHE_MAX) this._embedCache.clear();
                this._embedCache.set(text, vec);
                return vec;
            }
        } catch {
            // Embedding unavailable (no GPU, destroyed runtime, etc.) — signal fallback.
        }
        return null;
    }

    /**
     * Hard-deletes all entries whose TTL has expired.
     * @returns The number of entries deleted.
     */
    async purgeExpired(): Promise<number> {
        const db = await this._open();

        // Load all raw entries (including expired ones) to check each
        const tx = db.transaction(FACTS_STORE, 'readonly');
        const all = await requestToPromise(
            tx.objectStore(FACTS_STORE).getAll() as IDBRequest<MemoryEntry[]>,
            'Failed to scan facts for purge',
            (r) => r,
        );

        const expired = all.filter(e => this._isExpired(e));
        if (expired.length === 0) return 0;

        await Promise.all(expired.map(e => this.forget(e.key)));
        return expired.length;
    }

    /** Deletes a single fact. No-op if key does not exist. */
    async forget(key: string): Promise<void> {
        const db = await this._open();

        const tx = db.transaction(FACTS_STORE, 'readwrite');
        return requestToPromise(tx.objectStore(FACTS_STORE).delete(key), `Failed to forget fact "${key}"`, () => undefined);
    }

    /** Deletes all facts. Does not affect saved weights. */
    async clear(): Promise<void> {
        const db = await this._open();

        const tx = db.transaction(FACTS_STORE, 'readwrite');
        return requestToPromise(tx.objectStore(FACTS_STORE).clear(), 'Failed to clear facts', () => undefined);
    }

    // ── Cross-session memory merge ────────────────────────────────────────────

    /**
     * Returns all non-expired facts as a plain array for export.
     * Suitable for serialisation and import into another MemoryStore instance.
     */
    async exportAll(): Promise<MemoryEntry[]> {
        return this.recallAll();
    }

    /**
     * Imports entries from an external array.
     *
     * - `'merge'`     : only writes an entry if no existing entry with the same
     *                   key exists, or if the incoming entry has a newer timestamp.
     * - `'overwrite'` : writes all entries unconditionally.
     */
    async importAll(entries: MemoryEntry[], strategy: 'merge' | 'overwrite'): Promise<void> {
        for (const entry of entries) {
            if (strategy === 'overwrite') {
                await this._putRaw(entry);
            } else {
                const existing = await this.recall(entry.key);
                if (existing == null || entry.timestamp > existing.timestamp) {
                    await this._putRaw(entry);
                }
            }
        }
    }

    /** Writes a raw MemoryEntry directly (preserves original timestamp / metadata). */
    private async _putRaw(entry: MemoryEntry): Promise<void> {
        const db = await this._open();
        const tx = db.transaction(FACTS_STORE, 'readwrite');
        return requestToPromise(tx.objectStore(FACTS_STORE).put(entry), `Failed to import fact "${entry.key}"`, () => undefined);
    }

    // ── Weight persistence ────────────────────────────────────────────────────

    /**
     * Saves SSM weights via `runtime.save()`.
     * The weights are stored under `weightsKey` in this store's IndexedDB,
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

// ── IndexedDB helper ──────────────────────────────────────────────────────────

/**
 * Wires an IDBRequest to a promise: resolves with `map(result)` on success,
 * rejects with a normalised MEMORY_UNAVAILABLE error on failure. Extracted so
 * the success/error wiring lives in one place instead of being duplicated across
 * every store method.
 *
 * The onerror branch fires only on IndexedDB storage faults (quota exceeded,
 * disk failure, corruption), which the in-memory fake-indexeddb used in tests
 * cannot reproduce — hence the istanbul ignore on that single line.
 */
function requestToPromise<R, T>(req: IDBRequest<R>, failMsg: string, map: (result: R) => T): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        req.onsuccess = () => resolve(map(req.result));
        /* istanbul ignore next -- IDB storage-fault path; not reproducible with fake-indexeddb */
        req.onerror = () => reject(new SSMError(
            'MEMORY_UNAVAILABLE',
            `${failMsg}: ${req.error?.message ?? 'unknown'}`,
            req.error ?? undefined,
        ));
    });
}

