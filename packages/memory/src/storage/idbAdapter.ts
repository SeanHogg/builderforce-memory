/**
 * storage/idbAdapter.ts – Durable browser/IndexedDB implementation of the
 * storage port.
 *
 * Two properties the legacy MemoryStore write helpers did NOT have (M02):
 *   1. `transact()` resolves only on `tx.oncomplete` — the durable commit point —
 *      and rejects on `tx.onabort`. Request-level `onsuccess` is used internally
 *      to chain operations, never to signal durability.
 *   2. One `transact()` call is one unit of work spanning facts, tombstones,
 *      journal and meta, so a fact and its sync journal entry commit together.
 *
 * Schema (database version 3):
 *   - 'facts'      keyPath 'id'    — StoredFactRecord (scoped composite id)
 *       index 'byKey' on 'key'           — key lookup without a scan
 *       index 'byTag' on 'tags' multiEntry — tag lookup without a scan
 *   - 'tombstones' keyPath 'id'    — TombstoneRecord
 *   - 'journal'    keyPath 'opId'  — JournalEntry (the sync journal)
 *   - 'meta'       keyPath 'key'   — MetaCounters (durable seq / revision / opSeq)
 *   - 'weights'    (out-of-line)   — legacy weight blob store, retained
 *
 * Version 3 adds the key and tag indexes only. Existing fact records, including
 * their timestamps and TTLs, are not rewritten.
 *
 * Version-1 databases (facts keyed by the raw key, no journal) are migrated in
 * the upgrade transaction: records are copied to the new 'id'-keyed store with
 * the unbound scope, their seq preserved, revision 1, and no pending sync state.
 *
 * Cross-context consistency (M05): when BroadcastChannel is available the
 * adapter publishes a change note after every committed write and delivers
 * foreign notes to subscribers, so sibling tabs/windows invalidate their caches
 * instead of serving stale recalls. The channel opens lazily — on the first
 * publish, or once the database is open for a subscriber — and is `unref()`ed
 * where supported (Node), so a store that is never closed cannot keep the
 * process alive.
 */

import {
    FACT_KEY_INDEX,
    FACT_TAG_INDEX,
    MemoryStorageError,
    scopedRecordId,
    type ConsistencyNote,
    type FactStoragePort,
    type MetaCounters,
    type StorageStoreName,
    type StorageTransaction,
    type StoredFactRecord,
} from './types.js';

const FACTS_STORE      = 'facts';
const TOMBSTONES_STORE = 'tombstones';
const JOURNAL_STORE    = 'journal';
const META_STORE       = 'meta';
const WEIGHTS_STORE    = 'weights';
const DB_VERSION       = 3;

const ALL_STORES: StorageStoreName[] = [FACTS_STORE, TOMBSTONES_STORE, JOURNAL_STORE, META_STORE];

export interface IdbStorageAdapterOptions {
    /** IDBFactory override (tests, non-standard hosts). NOT a durability shim. */
    idbFactory?: IDBFactory;
    /** Identity of this context for cross-context notes. Default: random. */
    contextId?: string;
    /** BroadcastChannel override (tests). Default: global, when present. */
    channelFactory?: (name: string) => BroadcastChannelLike | undefined;
}

/** Minimal BroadcastChannel surface (Node's worker_threads channel satisfies it too). */
export interface BroadcastChannelLike {
    postMessage(message: unknown): void;
    close(): void;
    onmessage: ((event: { data: unknown }) => void) | null;
    /** Node only: stop this channel from keeping the event loop alive. */
    unref?(): void;
}

interface LegacyFactRecord {
    key: string;
    content: string;
    timestamp: number;
    seq?: number;
    ttlMs?: number;
    type?: StoredFactRecord['type'];
    tags?: string[];
    importance?: number;
}

function reqToPromise<R>(req: IDBRequest<R>): Promise<R> {
    return new Promise<R>((resolve, reject) => {
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error ?? new Error('IndexedDB request failed'));
    });
}

class IdbStorageTransaction implements StorageTransaction {
    constructor(private readonly _tx: IDBTransaction) {}

    async get<T>(store: StorageStoreName, id: string): Promise<T | undefined> {
        return reqToPromise(this._tx.objectStore(store).get(id) as IDBRequest<T | undefined>);
    }
    async getAll<T>(store: StorageStoreName): Promise<T[]> {
        return reqToPromise(this._tx.objectStore(store).getAll() as IDBRequest<T[]>);
    }
    async put(store: StorageStoreName, value: unknown): Promise<void> {
        await reqToPromise(this._tx.objectStore(store).put(value));
    }
    async delete(store: StorageStoreName, id: string): Promise<void> {
        await reqToPromise(this._tx.objectStore(store).delete(id));
    }

    async getByIndex<T>(store: StorageStoreName, indexName: string, query: string): Promise<T[] | undefined> {
        const objectStore = this._tx.objectStore(store);
        // Legacy databases opened before the index existed fall through to a scan.
        if (!objectStore.indexNames.contains(indexName)) return undefined;
        return reqToPromise(objectStore.index(indexName).getAll(query) as IDBRequest<T[]>);
    }
}

export class IdbStorageAdapter implements FactStoragePort {
    readonly kind = 'indexedDB' as const;

    private readonly _idb: IDBFactory | undefined;
    private readonly _contextId: string;
    private readonly _channelFactory: ((name: string) => BroadcastChannelLike | undefined) | undefined;
    private _db: IDBDatabase | null = null;
    private _channel: BroadcastChannelLike | null = null;
    private _channelFailed = false;
    private readonly _listeners = new Set<(note: ConsistencyNote) => void>();

    constructor(
        private readonly _dbName: string,
        opts: IdbStorageAdapterOptions = {},
    ) {
        this._idb = opts.idbFactory;
        this._contextId = opts.contextId ?? `ctx-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
        this._channelFactory = opts.channelFactory;
    }

    /** Identity of this context, stamped onto change notes and opIds. */
    get contextId(): string {
        return this._contextId;
    }

    private _open(): Promise<IDBDatabase> {
        if (this._db) return Promise.resolve(this._db);

        return new Promise((resolve, reject) => {
            const factory = this._idb ?? (typeof indexedDB !== 'undefined' ? indexedDB : undefined);
            if (!factory) {
                reject(new MemoryStorageError(
                    'STORAGE_UNAVAILABLE',
                    'IndexedDB is not available in this environment. For durable Node.js storage use NodeFileStorageAdapter; fake-indexeddb is a test double, not durable storage.',
                ));
                return;
            }

            const req = factory.open(this._dbName, DB_VERSION);

            req.onupgradeneeded = (e) => {
                const db = (e.target as IDBOpenDBRequest).result;
                const tx = (e.target as IDBOpenDBRequest).transaction!;
                migrateSchema(db, tx, e.oldVersion);
            };

            req.onsuccess = () => {
                this._db = req.result;
                // Subscribers registered before the first open start listening now.
                if (this._listeners.size > 0) this._ensureChannel();
                resolve(req.result);
            };
            /* istanbul ignore next -- IDB open onerror fires only on storage faults; not reproducible with fake-indexeddb */
            req.onerror = () => reject(new MemoryStorageError(
                'STORAGE_UNAVAILABLE',
                `Failed to open IndexedDB "${this._dbName}": ${req.error?.message ?? 'unknown'}`,
                req.error,
            ));
        });
    }

    /**
     * One unit of work: runs `fn` inside a single IndexedDB transaction and
     * resolves ONLY on `tx.oncomplete`. If `fn` throws, the transaction is
     * aborted so no partial writes survive; if the browser aborts the
     * transaction (quota, corruption, crash mid-commit) the promise rejects.
     */
    async transact<T>(
        stores: StorageStoreName[],
        mode: 'readonly' | 'readwrite',
        fn: (tx: StorageTransaction) => Promise<T> | T,
    ): Promise<T> {
        const db = await this._open();
        return new Promise<T>((resolve, reject) => {
            let settled = false;
            const fail = (err: unknown) => {
                if (settled) return;
                settled = true;
                reject(err instanceof MemoryStorageError ? err : new MemoryStorageError(
                    'STORAGE_ABORTED',
                    `Transaction aborted on "${this._dbName}": ${err instanceof Error ? err.message : String(err)}`,
                    err,
                ));
            };

            let tx: IDBTransaction;
            try {
                tx = db.transaction(stores, mode);
            } catch (err) {
                fail(err);
                return;
            }

            let result: T;
            tx.oncomplete = () => {
                if (settled) return;
                settled = true;
                resolve(result);
            };
            /* istanbul ignore next -- abort fires only on storage faults or explicit abort; fault path not reproducible with fake-indexeddb */
            tx.onabort = () => fail(tx.error ?? new Error('transaction aborted'));
            /* istanbul ignore next -- request errors surface through the per-request rejections and the abort handler */
            tx.onerror = () => { /* handled via request promises + onabort */ };

            const stx = new IdbStorageTransaction(tx);
            Promise.resolve()
                .then(() => fn(stx))
                .then((r) => { result = r; })
                .catch((err: unknown) => {
                    // The unit of work failed — abort so facts/journal/tombstones
                    // never commit partially (M02), then surface the ORIGINAL error.
                    try { tx.abort(); } catch { /* already finished */ }
                    if (!settled) {
                        settled = true;
                        reject(err);
                    }
                });
        });
    }

    // ── Cross-context consistency seam (M05) ──────────────────────────────────

    publish(note: ConsistencyNote): void {
        const channel = this._ensureChannel();
        if (!channel) return;
        try {
            channel.postMessage(note);
        } catch {
            // A dead channel must never fail a committed write.
        }
    }

    /**
     * Registers `listener` for foreign change notes. The channel itself opens
     * when the database does (or on the first publish), not here: subscribing
     * from a constructor must not open a global channel.
     */
    subscribe(listener: (note: ConsistencyNote) => void): () => void {
        this._listeners.add(listener);
        if (this._db) this._ensureChannel();
        return () => this._listeners.delete(listener);
    }

    private _ensureChannel(): BroadcastChannelLike | null {
        if (this._channel || this._channelFailed) return this._channel;
        try {
            const factory = this._channelFactory ?? defaultChannelFactory;
            const channel = factory(`bf-memory:${this._dbName}`);
            if (channel) {
                channel.onmessage = (e) => {
                    const note = e.data as ConsistencyNote;
                    if (!note || note.kind !== 'facts-changed') return;
                    if (note.origin === this._contextId) return; // our own write
                    for (const l of this._listeners) l(note);
                };
                channel.unref?.();
                this._channel = channel;
            }
        } catch {
            this._channelFailed = true;
        }
        return this._channel;
    }

    close(): void {
        this._channel?.close();
        this._channel = null;
        this._db?.close();
        this._db = null;
    }
}

function defaultChannelFactory(name: string): BroadcastChannelLike | undefined {
    if (typeof BroadcastChannel === 'undefined') return undefined;
    return new BroadcastChannel(name) as unknown as BroadcastChannelLike;
}

// ── Schema migration ──────────────────────────────────────────────────────────

function ensureFactIndexes(store: IDBObjectStore): void {
    if (!store.indexNames.contains(FACT_KEY_INDEX)) {
        store.createIndex(FACT_KEY_INDEX, 'key', { unique: false });
    }
    if (!store.indexNames.contains(FACT_TAG_INDEX)) {
        // One index row per tag, so recallByTag is a lookup rather than a scan.
        store.createIndex(FACT_TAG_INDEX, 'tags', { unique: false, multiEntry: true });
    }
}

function migrateSchema(db: IDBDatabase, tx: IDBTransaction, oldVersion: number): void {
    if (oldVersion < 1) {
        // Fresh database.
        const facts = db.createObjectStore(FACTS_STORE, { keyPath: 'id' });
        ensureFactIndexes(facts);
        db.createObjectStore(WEIGHTS_STORE);
    } else if (oldVersion < 2 && db.objectStoreNames.contains(FACTS_STORE)) {
        // v1 → v2: facts were keyed by the raw key; move them onto the scoped
        // record shape in the unbound partition. Copy out, drop, recreate, refill —
        // all inside the upgrade transaction, so a crash here rolls everything back.
        const legacy: LegacyFactRecord[] = [];
        const cursorReq = tx.objectStore(FACTS_STORE).openCursor();
        cursorReq.onsuccess = () => {
            const cursor = cursorReq.result;
            if (cursor) {
                legacy.push(cursor.value as LegacyFactRecord);
                cursor.continue();
            } else {
                db.deleteObjectStore(FACTS_STORE);
                const store = db.createObjectStore(FACTS_STORE, { keyPath: 'id' });
                ensureFactIndexes(store);
                for (const e of legacy) {
                    const record: StoredFactRecord = {
                        id          : scopedRecordId(undefined, e.key),
                        key         : e.key,
                        scope       : {},
                        content     : e.content,
                        timestamp   : e.timestamp,
                        seq         : e.seq ?? 0,
                        revision    : 1,
                        opId        : 'migrated:v1',
                        ttlMs       : e.ttlMs,
                        type        : e.type,
                        tags        : e.tags,
                        importance  : e.importance,
                        pendingSync : false,
                    };
                    store.put(record);
                }
            }
        };
    } else if (!db.objectStoreNames.contains(FACTS_STORE)) {
        const facts = db.createObjectStore(FACTS_STORE, { keyPath: 'id' });
        ensureFactIndexes(facts);
    } else if (oldVersion < 3) {
        // v2 → v3: add key/tag indexes. Do not rewrite records — timestamps,
        // TTLs, tags and content already on disk must survive the upgrade.
        ensureFactIndexes(tx.objectStore(FACTS_STORE));
    }

    if (!db.objectStoreNames.contains(WEIGHTS_STORE)) {
        db.createObjectStore(WEIGHTS_STORE);
    }
    if (!db.objectStoreNames.contains(TOMBSTONES_STORE)) {
        db.createObjectStore(TOMBSTONES_STORE, { keyPath: 'id' });
    }
    if (!db.objectStoreNames.contains(JOURNAL_STORE)) {
        db.createObjectStore(JOURNAL_STORE, { keyPath: 'opId' });
    }
    if (!db.objectStoreNames.contains(META_STORE)) {
        db.createObjectStore(META_STORE, { keyPath: 'key' });
    }
}

export { ALL_STORES, FACTS_STORE, TOMBSTONES_STORE, JOURNAL_STORE, META_STORE, WEIGHTS_STORE, DB_VERSION };
export type { MetaCounters };
