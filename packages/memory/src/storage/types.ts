/**
 * storage/types.ts – Narrow storage/transaction ports for durable memory.
 *
 * This module is the seam the offline-memory scope (M01–M05) requires:
 *   - MemoryStore no longer owns IndexedDB directly; it speaks to a
 *     {@link FactStoragePort} with REAL transaction semantics (a write resolves
 *     only after the transaction commits, and rejects on abort — M02).
 *   - Facts carry explicit scope partitions (account/tenant/project/repository/
 *     chat — or the explicit unbound scope for personal, never-connected
 *     memory), durable revisions, operation identity and provenance (M03/M04).
 *   - Deletes write tombstones, and every mutation appends to a sync journal in
 *     the SAME transaction as the fact change, so a crash can never acknowledge
 *     a fact whose journal entry was lost (M02/M04).
 *
 * Two durable adapters implement the port:
 *   - `IdbStorageAdapter`       (browser / VSIX webview — IndexedDB)
 *   - `NodeFileStorageAdapter`  (VSIX extension host / MCP — atomic JSON file)
 * An in-memory IndexedDB shim (fake-indexeddb) is a TEST double, never a
 * durability claim.
 */

import { SSMError, type SSMErrorCode } from '../errors/SSMError.js';

/** Explicit ownership partition for a fact. All fields optional; absent = unbound. */
export interface MemoryScope {
    account?: string;
    tenant?: string;
    project?: string;
    repository?: string;
    chat?: string;
}

/** The explicit "no account, never connected" partition — NOT the absence of a scope. */
export const UNBOUND_SCOPE: MemoryScope = Object.freeze({});

const SCOPE_FIELDS = ['account', 'tenant', 'project', 'repository', 'chat'] as const;

/**
 * Canonical partition string for a scope. The unbound scope partitions to the
 * empty string, so legacy records (written before scopes existed) land in the
 * unbound partition and stay readable. Two scopes that differ in any field are
 * different partitions — callers can never widen a recall by omitting a field
 * unless they genuinely target the unbound partition.
 */
export function scopePartitionKey(scope: MemoryScope | undefined): string {
    if (!scope) return '';
    const parts: string[] = [];
    for (const f of SCOPE_FIELDS) {
        const v = scope[f];
        if (v != null && v !== '') parts.push(`${f}=${encodeURIComponent(v)}`);
    }
    return parts.join('|');
}

/**
 * Physical record id for a (scope, key) pair. Unbound-scope ids are the bare
 * key — byte-identical to pre-scope records, so existing databases migrate
 * without rewriting keys.
 */
export function scopedRecordId(scope: MemoryScope | undefined, key: string): string {
    const partition = scopePartitionKey(scope);
    return partition === '' ? key : `${partition}::${key}`;
}

/** Where a fact came from. Survives export/import so source time is never rewritten. */
export interface FactProvenance {
    /** Origin surface/agent, e.g. 'vscode', 'mcp', 'web', 'cognition'. */
    source?: string;
    /** Snapshot/store this record was imported from. */
    importedFrom?: string;
    /** The timestamp the source store recorded; preserved across import (M06). */
    originalTimestamp?: number;
}

export type StorageStoreName = 'facts' | 'tombstones' | 'journal' | 'meta';

/**
 * Secondary index on {@link StoredFactRecord.key}. Key lookup uses this (or the
 * primary key) — never a full scan — when the adapter has built it.
 */
export const FACT_KEY_INDEX = 'byKey';

/**
 * Secondary multi-entry index on {@link StoredFactRecord.tags}. Tag lookup uses
 * this when present and falls back to a scan only when the index is missing.
 */
export const FACT_TAG_INDEX = 'byTag';

/**
 * The durable fact record. Extends the legacy MemoryEntry shape with identity,
 * revision and sync state. `pendingSync` is only ever true when the owning
 * store was created with sync enabled — without a sync target there is nothing
 * to be pending FOR, and eviction must behave exactly as before (M06).
 */
export interface StoredFactRecord {
    /** Physical primary key: {@link scopedRecordId}. */
    id: string;
    /** Caller-facing key within the scope. */
    key: string;
    /** Ownership partition. */
    scope: MemoryScope;
    content: string;
    /** Wall-clock write time (source time; preserved by import). */
    timestamp: number;
    /** Durable monotonic tiebreaker for same-millisecond writes. */
    seq: number;
    /** Durable revision: the store generation that last wrote this record. */
    revision: number;
    /** Durable identity of the operation that produced this record. */
    opId: string;
    ttlMs?: number;
    type?: 'text' | 'json' | 'number' | 'boolean';
    tags?: string[];
    importance?: number;
    provenance?: FactProvenance;
    /** True until a sync acknowledgement clears it; such facts are NEVER evicted by caps. */
    pendingSync: boolean;
}

/** A durable deletion marker. Deletes propagate through sync; tombstones prevent resurrection. */
export interface TombstoneRecord {
    id: string;
    key: string;
    scope: MemoryScope;
    deletedAt: number;
    /** Store generation of the delete; always greater than the fact's last revision. */
    revision: number;
    opId: string;
    pendingSync: boolean;
}

export type JournalOpKind = 'remember' | 'forget';

/** One durable operation in the sync journal — written atomically with its fact change. */
export interface JournalEntry {
    opId: string;
    kind: JournalOpKind;
    id: string;
    key: string;
    scope: MemoryScope;
    /** Store generation after this op committed. */
    revision: number;
    /** The revision the writer based its write on (CAS); undefined = blind write. */
    baseRevision?: number;
    timestamp: number;
    synced: boolean;
}

/** Counters kept durably in the meta store. */
export interface MetaCounters {
    key: 'counters';
    /** Durable write sequence (replaces the process-local `_writeSeq`). */
    seq: number;
    /** Durable store generation, bumped by every committed write transaction. */
    revision: number;
    /** Durable operation counter; opIds are `${contextId}:${opSeq}`. */
    opSeq: number;
}

/**
 * The minimal surface a transaction offers. Adapters MUST keep the underlying
 * transaction alive across the awaited operations (request-chained awaits only)
 * and MUST NOT resolve {@link FactStoragePort.transact} before commit.
 */
export interface StorageTransaction {
    get<T>(store: StorageStoreName, id: string): Promise<T | undefined>;
    getAll<T>(store: StorageStoreName): Promise<T[]>;
    put(store: StorageStoreName, value: unknown): Promise<void>;
    delete(store: StorageStoreName, id: string): Promise<void>;
    /**
     * Lookup via a named index. `undefined` means this store has no such index
     * (legacy schema, or an adapter that has not built it) — the caller MUST
     * fall back to a primary-key get or a scan. An empty array means the index
     * exists and matched nothing; do not scan in that case.
     */
    getByIndex?<T>(store: StorageStoreName, indexName: string, query: string): Promise<T[] | undefined>;
}

/** A change notice for cross-context invalidation (M05). */
export interface ConsistencyNote {
    kind: 'facts-changed';
    /** Scope partition that changed; undefined = whole database. */
    scope?: MemoryScope;
    /** Durable generation after the change. */
    revision: number;
    /** Identity of the committing context; a context ignores its own notes. */
    origin: string;
}

/**
 * The narrow durable-storage port. One transaction = one unit of work:
 * fact mutation + provenance + tombstone/event + journal commit together or
 * not at all.
 */
export interface FactStoragePort {
    /** Which durable backend this is (for honest capability reporting). */
    readonly kind: 'indexedDB' | 'node-file';
    /**
     * Runs `fn` inside one real transaction over the named stores. Resolves with
     * `fn`'s result ONLY after the transaction has committed durably; rejects if
     * the transaction aborts, `fn` throws, or the persist fails. A rejected
     * transact leaves NO partial writes behind.
     */
    transact<T>(
        stores: StorageStoreName[],
        mode: 'readonly' | 'readwrite',
        fn: (tx: StorageTransaction) => Promise<T> | T,
    ): Promise<T>;
    /**
     * Publishes a change notice to other contexts sharing this storage, when the
     * adapter supports cross-context notification. Optional.
     */
    publish?(note: ConsistencyNote): void;
    /**
     * Subscribes to change notices from OTHER contexts (tabs, extension windows,
     * MCP processes) sharing this storage. Returns an unsubscribe function.
     * Optional: an adapter without it simply offers no cross-context seam.
     */
    subscribe?(listener: (note: ConsistencyNote) => void): () => void;
    close?(): void | Promise<void>;
}

/** Detailed error codes surfaced by the storage layer (`MemoryStorageError.storageCode`). */
export type StorageErrorCode =
    /** The durable backend is unavailable or faulted. */
    | 'STORAGE_UNAVAILABLE'
    /** A compare-and-set write found a newer revision than its base. */
    | 'STORAGE_CONFLICT'
    /** A transaction aborted; no partial writes were committed. */
    | 'STORAGE_ABORTED';

/**
 * The public {@link SSMError} code for each storage code. Callers that predate
 * the storage layer branch on `code === 'MEMORY_UNAVAILABLE'` (what MemoryStore
 * always rejected with when IndexedDB was missing or faulted), so unavailable
 * and aborted storage keep that code; a CAS conflict is the new `MEMORY_CONFLICT`.
 */
const PUBLIC_CODE: Record<StorageErrorCode, SSMErrorCode> = {
    STORAGE_UNAVAILABLE: 'MEMORY_UNAVAILABLE',
    STORAGE_ABORTED    : 'MEMORY_UNAVAILABLE',
    STORAGE_CONFLICT   : 'MEMORY_CONFLICT',
};

/**
 * A storage-layer failure. It IS an {@link SSMError} whose `code` is the public
 * contract (`MEMORY_UNAVAILABLE` / `MEMORY_CONFLICT`); `storageCode` keeps the
 * detailed storage code for callers that need it.
 */
export class MemoryStorageError extends SSMError {
    readonly storageCode: StorageErrorCode;

    constructor(storageCode: StorageErrorCode, message: string, cause?: unknown) {
        super(PUBLIC_CODE[storageCode], message, cause);
        this.storageCode = storageCode;
        this.name = 'MemoryStorageError';
    }
}

/** Type guard for conflict errors so callers can reconcile instead of blindly retrying. */
export function isStorageConflict(err: unknown): err is MemoryStorageError {
    return err instanceof MemoryStorageError && err.storageCode === 'STORAGE_CONFLICT';
}
