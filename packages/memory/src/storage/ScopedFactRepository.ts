/**
 * storage/ScopedFactRepository.ts – Scoped, journaled, revision-tracked fact
 * repository over a {@link FactStoragePort}.
 *
 * This is the unit-of-work owner the offline-memory scope requires:
 *
 *   - Scope partitions (M03): every operation takes an explicit
 *     {@link MemoryScope}; records are physically partitioned by scoped record
 *     id and logically filtered by partition, so no code path can recall across
 *     account/tenant/project/repository/chat boundaries. The unbound scope is
 *     an explicit partition for personal, never-connected memory.
 *   - Atomic commits (M02): fact mutation + provenance + tombstone + sync
 *     journal + durable counters commit in ONE storage transaction.
 *   - Durable identity (M04): every record carries a durable `revision` (the
 *     store generation that wrote it) and `opId`; deletes write tombstones;
 *     writes may be compare-and-set against a base revision, so concurrent
 *     evidence conflicts surface as explicit conflicts instead of unconditional
 *     last-write-wins.
 *   - Sync-aware retention (M06): facts with `pendingSync` are never evicted by
 *     caps and are never dropped by import; TTL expiry is an intentional
 *     deletion and writes a tombstone when sync is enabled.
 *
 * The shared write rules live in `factMutations.ts`, retention in
 * `factRetention.ts`, import in `factImport.ts`, sync acknowledgement in
 * `syncJournal.ts`.
 */

import { factsInScope, indexedFacts } from './factQueries.js';
import {
    WRITE_STORES,
    bumpOp,
    clearTombstone,
    deleteWithTombstone,
    journal,
    readCounters,
    writeCounters,
    type MutationContext,
} from './factMutations.js';
import { importInto, type ImportResult } from './factImport.js';
import * as retention from './factRetention.js';
import * as syncJournal from './syncJournal.js';
import {
    FACT_KEY_INDEX,
    FACT_TAG_INDEX,
    MemoryStorageError,
    scopedRecordId,
    scopePartitionKey,
    type ConsistencyNote,
    type FactProvenance,
    type FactStoragePort,
    type JournalEntry,
    type MemoryScope,
    type StoredFactRecord,
    type TombstoneRecord,
} from './types.js';

export interface ScopedFactRepositoryOptions {
    /**
     * When true, mutations are journaled (unsynced) until {@link markSynced}
     * acknowledges and prunes them, deletes write tombstones, and cap eviction
     * skips unsynced facts. When false (default — no sync target configured)
     * nothing is journaled and retention behaves exactly like the legacy store.
     */
    syncEnabled?: boolean;
    /** Identity of this repository context for opIds and change notes. */
    contextId?: string;
    /** Injectable clock (ms). Default: Date.now. */
    now?: () => number;
}

export interface RememberRecordInput {
    content: string;
    /** Explicit source time (import). Default: now. */
    timestamp?: number;
    /** Explicit source sequence (import). Default: next durable sequence. */
    seq?: number;
    ttlMs?: number;
    type?: StoredFactRecord['type'];
    tags?: string[];
    importance?: number;
    provenance?: FactProvenance;
    /**
     * Compare-and-set base: `null` asserts the record does not exist; a number
     * asserts the current revision; undefined is a blind write (legacy).
     */
    expectedRevision?: number | null;
    /** Explicit op identity (import/replay). Default: generated from the durable op counter. */
    opId?: string;
}

export interface WriteOutcome {
    record: StoredFactRecord;
    /** True when an existing tombstone for this key was cleared by the write. */
    resurrected: boolean;
}

export type { ImportResult };

export class ScopedFactRepository {
    private readonly _storage: FactStoragePort;
    private readonly _ctx: MutationContext;

    constructor(storage: FactStoragePort, opts: ScopedFactRepositoryOptions = {}) {
        this._storage = storage;
        this._ctx = {
            syncEnabled: opts.syncEnabled ?? false,
            contextId  : opts.contextId ?? 'repo',
            now        : opts.now ?? (() => Date.now()),
        };
    }

    get storage(): FactStoragePort {
        return this._storage;
    }

    get contextId(): string {
        return this._ctx.contextId;
    }

    get syncEnabled(): boolean {
        return this._ctx.syncEnabled;
    }

    // ── Reads ─────────────────────────────────────────────────────────────────

    /** Current durable store generation (0 on a fresh store). */
    async revision(): Promise<number> {
        return this._storage.transact(['meta'], 'readonly', async (tx) => (await readCounters(tx)).revision);
    }

    /**
     * Fetches one record by (scope, key), or undefined. Tombstoned = absent.
     * Uses the `byKey` index when the adapter has it; falls back to the primary
     * key (itself an index on the scoped id) only when that index is missing.
     */
    async get(scope: MemoryScope | undefined, key: string): Promise<StoredFactRecord | undefined> {
        const id = scopedRecordId(scope, key);
        return this._storage.transact(['facts'], 'readonly', async (tx) => {
            const indexed = await indexedFacts(tx, FACT_KEY_INDEX, key);
            if (indexed !== undefined) return indexed.find((r) => r.id === id);
            return tx.get<StoredFactRecord>('facts', id);
        });
    }

    /**
     * Facts in `scope` carrying `tag`. Uses the multi-entry `byTag` index when
     * present. A full scan runs only when the index is missing (legacy schema).
     */
    async getByTag(scope: MemoryScope | undefined, tag: string): Promise<StoredFactRecord[]> {
        const partition = scopePartitionKey(scope);
        return this._storage.transact(['facts'], 'readonly', async (tx) => {
            const indexed = await indexedFacts(tx, FACT_TAG_INDEX, tag);
            if (indexed !== undefined) {
                return indexed.filter((r) => scopePartitionKey(r.scope) === partition);
            }
            return (await factsInScope(tx, scope)).filter((r) => r.tags?.includes(tag) ?? false);
        });
    }

    /**
     * All records in exactly this scope partition — never a wider or narrower
     * set. Expired records are included; TTL filtering is the caller's policy.
     */
    async getAll(scope: MemoryScope | undefined): Promise<StoredFactRecord[]> {
        return this._storage.transact(['facts'], 'readonly', (tx) => factsInScope(tx, scope));
    }

    /** Tombstones, optionally narrowed to one scope partition. */
    async tombstones(scope?: MemoryScope): Promise<TombstoneRecord[]> {
        return this._storage.transact(['tombstones'], 'readonly', async (tx) => {
            const all = await tx.getAll<TombstoneRecord>('tombstones');
            if (scope === undefined) return all;
            const partition = scopePartitionKey(scope);
            return all.filter((t) => scopePartitionKey(t.scope) === partition);
        });
    }

    /** Journal entries not yet acknowledged by sync — the pending operation set. */
    async pendingOps(): Promise<JournalEntry[]> {
        return syncJournal.pendingOps(this._storage);
    }

    // ── Writes (unit of work: fact + journal + counters in one transaction) ───

    /**
     * Stores or overwrites a fact in `scope`. With `expectedRevision` set this
     * is a compare-and-set: a mismatch throws a STORAGE_CONFLICT error carrying
     * the current record, so a concurrent writer's evidence is never silently
     * overwritten (M04/M05).
     */
    async remember(scope: MemoryScope | undefined, key: string, input: RememberRecordInput): Promise<WriteOutcome> {
        const id = scopedRecordId(scope, key);
        const outcome = await this._storage.transact(WRITE_STORES, 'readwrite', async (tx): Promise<WriteOutcome> => {
            const counters = await readCounters(tx);
            const existing = await tx.get<StoredFactRecord>('facts', id);
            if (input.expectedRevision !== undefined) {
                const current = existing?.revision ?? null;
                if (current !== input.expectedRevision) {
                    throw new MemoryStorageError(
                        'STORAGE_CONFLICT',
                        `Conflict writing "${key}": expected revision ${String(input.expectedRevision)}, found ${String(current)}. Re-read and reconcile instead of overwriting.`,
                    );
                }
            }
            const bumped = bumpOp(counters, this._ctx);
            const opId = input.opId ?? bumped.opId;
            const next = { ...bumped.counters, seq: Math.max(counters.seq + 1, input.seq ?? 0) };
            const record: StoredFactRecord = {
                id,
                key,
                scope      : { ...(scope ?? {}) },
                content    : input.content,
                timestamp  : input.timestamp ?? this._ctx.now(),
                seq        : next.seq,
                revision   : next.revision,
                opId,
                ttlMs      : input.ttlMs,
                type       : input.type,
                tags       : input.tags,
                importance : input.importance,
                provenance : input.provenance,
                pendingSync: this._ctx.syncEnabled,
            };
            await tx.put('facts', record);
            const resurrected = await clearTombstone(tx, id);
            await journal(tx, this._ctx, {
                opId,
                kind         : 'remember',
                id,
                key,
                scope        : record.scope,
                revision     : next.revision,
                baseRevision : input.expectedRevision ?? undefined,
                timestamp    : record.timestamp,
            });
            await writeCounters(tx, next);
            return { record, resurrected };
        });
        this._publish(scope, outcome.record.revision);
        return outcome;
    }

    /**
     * Deletes a fact. With sync enabled the delete is a durable tombstone +
     * journal entry committed in the same transaction, so the deletion
     * propagates and a stale replica cannot resurrect the fact (M04). Without
     * sync this is the legacy hard delete. Deleting an absent key is a no-op.
     */
    async forget(scope: MemoryScope | undefined, key: string, opts: { expectedRevision?: number } = {}): Promise<boolean> {
        const id = scopedRecordId(scope, key);
        const revision = await this._storage.transact(WRITE_STORES, 'readwrite', async (tx): Promise<number> => {
            const existing = await tx.get<StoredFactRecord>('facts', id);
            if (!existing) return 0;
            if (opts.expectedRevision !== undefined && existing.revision !== opts.expectedRevision) {
                throw new MemoryStorageError(
                    'STORAGE_CONFLICT',
                    `Conflict deleting "${key}": expected revision ${opts.expectedRevision}, found ${existing.revision}.`,
                );
            }
            const next = await deleteWithTombstone(tx, this._ctx, existing, await readCounters(tx), opts.expectedRevision);
            await writeCounters(tx, next);
            return next.revision;
        });
        if (revision > 0) this._publish(scope, revision);
        return revision > 0;
    }

    /** Removes every fact in the scope. Tombstones each deletion when sync is enabled. */
    async clear(scope: MemoryScope | undefined): Promise<number> {
        const { removed, revision } = await this._storage.transact(WRITE_STORES, 'readwrite', async (tx) => {
            const mine = await factsInScope(tx, scope);
            if (mine.length === 0) return { removed: 0, revision: 0 };
            let counters = await readCounters(tx);
            for (const record of mine) counters = await deleteWithTombstone(tx, this._ctx, record, counters);
            await writeCounters(tx, counters);
            return { removed: mine.length, revision: counters.revision };
        });
        if (removed > 0) this._publish(scope, revision);
        return removed;
    }

    // ── Import / export (M03/M06: scope + source time + provenance preserved) ──

    /**
     * Imports records INTO `scope`, preserving each record's source timestamp,
     * sequence, tags, importance, TTL and provenance. `importedFrom` is stamped
     * onto the provenance so the snapshot origin survives. 'merge' keeps the
     * record with the newer source timestamp; 'overwrite' always writes.
     * Imported facts are pendingSync when sync is enabled — they are new to
     * THIS replica even if another store already had them.
     */
    async importRecords(
        scope: MemoryScope | undefined,
        records: StoredFactRecord[],
        strategy: 'merge' | 'overwrite',
        importedFrom?: string,
    ): Promise<ImportResult> {
        const { written, skipped, revision } = await importInto(this._storage, this._ctx, scope, records, strategy, importedFrom);
        if (written > 0) this._publish(scope, revision);
        return { written, skipped };
    }

    /** Exports every record in the scope, with scope/revision/provenance intact (M03). */
    async exportRecords(scope: MemoryScope | undefined): Promise<StoredFactRecord[]> {
        return this.getAll(scope);
    }

    // ── Sync acknowledgement (M04/M06) ────────────────────────────────────────

    /**
     * Acknowledges synced operations: their journal entries are pruned and the
     * facts/tombstones they produced clear `pendingSync`, making cap eviction
     * eligible again. Only after this may retention touch those records.
     */
    async markSynced(opIds: string[]): Promise<void> {
        await syncJournal.markSynced(this._storage, opIds);
    }

    // ── Retention (M06) ───────────────────────────────────────────────────────

    /**
     * Local-only cap eviction: removes the lowest-value records past
     * `maxEntries`, EXCLUDING records with unsynced operations — pending
     * knowledge is never evicted to satisfy a cache cap. Eviction of synced
     * records writes no tombstone and no journal entry by design: it is local
     * cache management, not a deletion to propagate. Returns the evicted keys.
     */
    async evictForCap(
        scope: MemoryScope | undefined,
        maxEntries: number,
        isExpired: (record: StoredFactRecord) => boolean,
    ): Promise<string[]> {
        return retention.evictForCap(this._storage, scope, maxEntries, isExpired);
    }

    /**
     * Intentional retention: removes expired records. With sync enabled each
     * expiry is a scoped deletion — tombstone + journal in one transaction — so
     * the expiry propagates instead of resurrecting from a replica. Returns the
     * number of records purged.
     */
    async purgeExpired(scope: MemoryScope | undefined, isExpired: (record: StoredFactRecord) => boolean): Promise<number> {
        const { purged, revision } = await retention.purgeExpired(this._storage, this._ctx, scope, isExpired);
        if (purged > 0) this._publish(scope, revision);
        return purged;
    }

    // ── Cross-context consistency seam (M05) ──────────────────────────────────

    /**
     * Subscribes to change notes committed by OTHER contexts sharing this
     * storage (other tabs, extension windows, MCP processes). In-process
     * serialization alone cannot be the distributed consistency boundary; this
     * is the invalidation half of the seam, and compare-and-set is the
     * write-conflict half.
     */
    subscribe(listener: (note: ConsistencyNote) => void): () => void {
        if (!this._storage.subscribe) return () => undefined;
        return this._storage.subscribe((note) => {
            if (note.origin === this._ctx.contextId) return;
            listener(note);
        });
    }

    private _publish(scope: MemoryScope | undefined, revision: number): void {
        this._storage.publish?.({
            kind    : 'facts-changed',
            scope   : scope ? { ...scope } : undefined,
            revision,
            origin  : this._ctx.contextId,
        });
    }
}
