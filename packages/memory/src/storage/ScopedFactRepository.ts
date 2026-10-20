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
 */

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
    type MetaCounters,
    type StorageTransaction,
    type StoredFactRecord,
    type TombstoneRecord,
} from './types.js';

export interface ScopedFactRepositoryOptions {
    /**
     * When true, mutations are journaled with `pendingSync`/`synced=false`
     * until {@link markSynced} acknowledges them, deletes write tombstones, and
     * cap eviction skips unsynced facts. When false (default — no sync target
     * configured) retention behaves exactly like the legacy store.
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

export interface ImportResult {
    written: number;
    skipped: number;
}

const COUNTERS_ID = 'counters';

export class ScopedFactRepository {
    private readonly _storage: FactStoragePort;
    private readonly _syncEnabled: boolean;
    private readonly _contextId: string;
    private readonly _now: () => number;

    constructor(storage: FactStoragePort, opts: ScopedFactRepositoryOptions = {}) {
        this._storage = storage;
        this._syncEnabled = opts.syncEnabled ?? false;
        this._contextId = opts.contextId ?? 'repo';
        this._now = opts.now ?? (() => Date.now());
    }

    get storage(): FactStoragePort {
        return this._storage;
    }

    get contextId(): string {
        return this._contextId;
    }

    get syncEnabled(): boolean {
        return this._syncEnabled;
    }

    // ── Reads ─────────────────────────────────────────────────────────────────

    /** Current durable store generation (0 on a fresh store). */
    async revision(): Promise<number> {
        return this._storage.transact(['meta'], 'readonly', async (tx) => {
            const counters = await this._counters(tx);
            return counters.revision;
        });
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
            const all = await tx.getAll<StoredFactRecord>('facts');
            return all.filter((r) => scopePartitionKey(r.scope) === partition && (r.tags?.includes(tag) ?? false));
        });
    }

    /**
     * All records in exactly this scope partition — never a wider or narrower
     * set. Expired records are included; TTL filtering is the caller's policy.
     */
    async getAll(scope: MemoryScope | undefined): Promise<StoredFactRecord[]> {
        const partition = scopePartitionKey(scope);
        return this._storage.transact(['facts'], 'readonly', async (tx) => {
            const all = await tx.getAll<StoredFactRecord>('facts');
            return all.filter((r) => scopePartitionKey(r.scope) === partition);
        });
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
        return this._storage.transact(['journal'], 'readonly', async (tx) => {
            const all = await tx.getAll<JournalEntry>('journal');
            return all.filter((j) => !j.synced);
        });
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
        const outcome = await this._storage.transact(
            ['facts', 'tombstones', 'journal', 'meta'],
            'readwrite',
            async (tx): Promise<WriteOutcome> => {
                const counters = await this._counters(tx);
                const existing = await tx.get<StoredFactRecord>('facts', id);

                // ── Compare-and-set gate ────────────────────────────────────
                if (input.expectedRevision !== undefined) {
                    const current = existing?.revision ?? null;
                    if (current !== input.expectedRevision) {
                        throw new MemoryStorageError(
                            'STORAGE_CONFLICT',
                            `Conflict writing "${key}": expected revision ${String(input.expectedRevision)}, found ${String(current)}. Re-read and reconcile instead of overwriting.`,
                        );
                    }
                }

                const revision = counters.revision + 1;
                const opSeq = counters.opSeq + 1;
                const opId = input.opId ?? `${this._contextId}:${opSeq}`;
                const seq = Math.max(counters.seq + 1, input.seq ?? 0);

                const record: StoredFactRecord = {
                    id,
                    key,
                    scope      : { ...(scope ?? {}) },
                    content    : input.content,
                    timestamp  : input.timestamp ?? this._now(),
                    seq,
                    revision,
                    opId,
                    ttlMs      : input.ttlMs,
                    type       : input.type,
                    tags       : input.tags,
                    importance : input.importance,
                    provenance : input.provenance,
                    pendingSync: this._syncEnabled,
                };
                await tx.put('facts', record);

                // A write resurrects the key: clear any tombstone in the same tx.
                const tombstone = await tx.get<TombstoneRecord>('tombstones', id);
                if (tombstone) await tx.delete('tombstones', id);

                await tx.put('journal', {
                    opId,
                    kind         : 'remember',
                    id,
                    key,
                    scope        : record.scope,
                    revision,
                    baseRevision : input.expectedRevision ?? undefined,
                    timestamp    : record.timestamp,
                    synced       : !this._syncEnabled,
                } satisfies JournalEntry);

                await tx.put('meta', { key: COUNTERS_ID, seq, revision, opSeq } satisfies MetaCounters);
                return { record, resurrected: !!tombstone };
            },
        );
        this._publish(scope, outcome.record.revision);
        return outcome;
    }

    /**
     * Deletes a fact. With sync enabled the delete is a durable tombstone +
     * journal entry committed in the same transaction, so the deletion
     * propagates and a stale replica cannot resurrect the fact (M04). Without
     * sync this is the legacy hard delete.
     */
    async forget(scope: MemoryScope | undefined, key: string, opts: { expectedRevision?: number } = {}): Promise<boolean> {
        const id = scopedRecordId(scope, key);
        let revision = 0;
        const deleted = await this._storage.transact(
            ['facts', 'tombstones', 'journal', 'meta'],
            'readwrite',
            async (tx): Promise<boolean> => {
                const counters = await this._counters(tx);
                const existing = await tx.get<StoredFactRecord>('facts', id);
                if (!existing) {
                    const hadTombstone = !!(await tx.get<TombstoneRecord>('tombstones', id));
                    return hadTombstone ? false : false; // idempotent no-op
                }
                if (opts.expectedRevision !== undefined && existing.revision !== opts.expectedRevision) {
                    throw new MemoryStorageError(
                        'STORAGE_CONFLICT',
                        `Conflict deleting "${key}": expected revision ${opts.expectedRevision}, found ${existing.revision}.`,
                    );
                }

                revision = counters.revision + 1;
                const opSeq = counters.opSeq + 1;
                const opId = `${this._contextId}:${opSeq}`;
                await tx.delete('facts', id);
                if (this._syncEnabled) {
                    await tx.put('tombstones', {
                        id,
                        key,
                        scope      : existing.scope,
                        deletedAt  : this._now(),
                        revision,
                        opId,
                        pendingSync: true,
                    } satisfies TombstoneRecord);
                }
                await tx.put('journal', {
                    opId,
                    kind         : 'forget',
                    id,
                    key,
                    scope        : existing.scope,
                    revision,
                    baseRevision : opts.expectedRevision,
                    timestamp    : this._now(),
                    synced       : !this._syncEnabled,
                } satisfies JournalEntry);
                await tx.put('meta', { key: COUNTERS_ID, seq: counters.seq, revision, opSeq } satisfies MetaCounters);
                return true;
            },
        );
        if (deleted) this._publish(scope, revision);
        return deleted;
    }

    /** Removes every fact in the scope. Tombstones each deletion when sync is enabled. */
    async clear(scope: MemoryScope | undefined): Promise<number> {
        const partition = scopePartitionKey(scope);
        let revision = 0;
        const removed = await this._storage.transact(
            ['facts', 'tombstones', 'journal', 'meta'],
            'readwrite',
            async (tx): Promise<number> => {
                const all = await tx.getAll<StoredFactRecord>('facts');
                const mine = all.filter((r) => scopePartitionKey(r.scope) === partition);
                if (mine.length === 0) return 0;
                let counters = await this._counters(tx);
                for (const record of mine) {
                    counters = { ...counters, revision: counters.revision + 1, opSeq: counters.opSeq + 1 };
                    const opId = `${this._contextId}:${counters.opSeq}`;
                    await tx.delete('facts', record.id);
                    if (this._syncEnabled) {
                        await tx.put('tombstones', {
                            id         : record.id,
                            key        : record.key,
                            scope      : record.scope,
                            deletedAt  : this._now(),
                            revision   : counters.revision,
                            opId,
                            pendingSync: true,
                        } satisfies TombstoneRecord);
                    }
                    await tx.put('journal', {
                        opId,
                        kind     : 'forget',
                        id       : record.id,
                        key      : record.key,
                        scope    : record.scope,
                        revision : counters.revision,
                        timestamp: this._now(),
                        synced   : !this._syncEnabled,
                    } satisfies JournalEntry);
                }
                revision = counters.revision;
                await tx.put('meta', { key: COUNTERS_ID, ...counters } satisfies MetaCounters);
                return mine.length;
            },
        );
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
        let revision = 0;
        const result = await this._storage.transact(
            ['facts', 'tombstones', 'journal', 'meta'],
            'readwrite',
            async (tx): Promise<ImportResult> => {
                let counters = await this._counters(tx);
                let written = 0;
                let skipped = 0;
                for (const incoming of records) {
                    const id = scopedRecordId(scope, incoming.key);
                    const existing = await tx.get<StoredFactRecord>('facts', id);
                    if (strategy === 'merge' && existing && existing.timestamp >= incoming.timestamp) {
                        skipped++;
                        continue;
                    }
                    counters = {
                        ...counters,
                        revision : counters.revision + 1,
                        opSeq    : counters.opSeq + 1,
                        seq      : Math.max(counters.seq + 1, incoming.seq ?? 0),
                    };
                    const opId = `${this._contextId}:${counters.opSeq}`;
                    const record: StoredFactRecord = {
                        ...incoming,
                        id,
                        key         : incoming.key,
                        scope       : { ...(scope ?? {}) },
                        seq         : incoming.seq ?? counters.seq,
                        revision    : counters.revision,
                        opId,
                        provenance  : {
                            ...incoming.provenance,
                            ...(importedFrom ? { importedFrom } : {}),
                            originalTimestamp: incoming.provenance?.originalTimestamp ?? incoming.timestamp,
                        },
                        pendingSync : this._syncEnabled,
                    };
                    await tx.put('facts', record);
                    const tombstone = await tx.get<TombstoneRecord>('tombstones', id);
                    if (tombstone) await tx.delete('tombstones', id);
                    await tx.put('journal', {
                        opId,
                        kind     : 'remember',
                        id,
                        key      : incoming.key,
                        scope    : record.scope,
                        revision : counters.revision,
                        timestamp: record.timestamp,
                        synced   : !this._syncEnabled,
                    } satisfies JournalEntry);
                    written++;
                }
                if (written > 0) {
                    revision = counters.revision;
                    await tx.put('meta', { key: COUNTERS_ID, ...counters } satisfies MetaCounters);
                }
                return { written, skipped };
            },
        );
        if (result.written > 0) this._publish(scope, revision);
        return result;
    }

    /** Exports every record in the scope, with scope/revision/provenance intact (M03). */
    async exportRecords(scope: MemoryScope | undefined): Promise<StoredFactRecord[]> {
        return this.getAll(scope);
    }

    // ── Sync acknowledgement (M04/M06) ────────────────────────────────────────

    /**
     * Acknowledges synced operations: journal entries flip to synced, and the
     * facts/tombstones they produced clear `pendingSync`, making cap eviction
     * eligible again. Only after this may retention touch those records.
     */
    async markSynced(opIds: string[]): Promise<void> {
        if (opIds.length === 0) return;
        const acked = new Set(opIds);
        await this._storage.transact(['journal', 'facts', 'tombstones'], 'readwrite', async (tx) => {
            const journal = await tx.getAll<JournalEntry>('journal');
            for (const entry of journal) {
                if (acked.has(entry.opId) && !entry.synced) {
                    await tx.put('journal', { ...entry, synced: true });
                }
            }
            const facts = await tx.getAll<StoredFactRecord>('facts');
            for (const record of facts) {
                if (acked.has(record.opId) && record.pendingSync) {
                    await tx.put('facts', { ...record, pendingSync: false });
                }
            }
            const tombstones = await tx.getAll<TombstoneRecord>('tombstones');
            for (const tomb of tombstones) {
                if (acked.has(tomb.opId) && tomb.pendingSync) {
                    await tx.put('tombstones', { ...tomb, pendingSync: false });
                }
            }
        });
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
        const partition = scopePartitionKey(scope);
        return this._storage.transact(['facts'], 'readwrite', async (tx) => {
            const all = await tx.getAll<StoredFactRecord>('facts');
            const mine = all.filter((r) => scopePartitionKey(r.scope) === partition);
            if (mine.length <= maxEntries) return [];

            const rank = (e: StoredFactRecord): [number, number, number, number] =>
                [isExpired(e) ? 0 : 1, e.importance ?? 0.5, e.timestamp, e.seq];
            const evictable = mine
                .filter((r) => !r.pendingSync)
                .sort((a, b) => {
                    const ra = rank(a), rb = rank(b);
                    for (let i = 0; i < ra.length; i++) {
                        if (ra[i] !== rb[i]) return ra[i]! - rb[i]!;
                    }
                    return 0;
                });
            const victims = evictable.slice(0, Math.max(0, mine.length - maxEntries));
            for (const v of victims) await tx.delete('facts', v.id);
            return victims.map((v) => v.key);
        });
    }

    /**
     * Intentional retention: removes expired records. With sync enabled each
     * expiry is a scoped deletion — tombstone + journal in one transaction — so
     * the expiry propagates instead of resurrecting from a replica. Returns the
     * number of records purged.
     */
    async purgeExpired(scope: MemoryScope | undefined, isExpired: (record: StoredFactRecord) => boolean): Promise<number> {
        const partition = scopePartitionKey(scope);
        let revision = 0;
        const purged = await this._storage.transact(
            ['facts', 'tombstones', 'journal', 'meta'],
            'readwrite',
            async (tx): Promise<number> => {
                const all = await tx.getAll<StoredFactRecord>('facts');
                const expired = all.filter((r) => scopePartitionKey(r.scope) === partition && isExpired(r));
                if (expired.length === 0) return 0;
                let counters = await this._counters(tx);
                for (const record of expired) {
                    counters = { ...counters, revision: counters.revision + 1, opSeq: counters.opSeq + 1 };
                    const opId = `${this._contextId}:${counters.opSeq}`;
                    await tx.delete('facts', record.id);
                    if (this._syncEnabled) {
                        await tx.put('tombstones', {
                            id         : record.id,
                            key        : record.key,
                            scope      : record.scope,
                            deletedAt  : this._now(),
                            revision   : counters.revision,
                            opId,
                            pendingSync: true,
                        } satisfies TombstoneRecord);
                    }
                    await tx.put('journal', {
                        opId,
                        kind     : 'forget',
                        id       : record.id,
                        key      : record.key,
                        scope    : record.scope,
                        revision : counters.revision,
                        timestamp: this._now(),
                        synced   : !this._syncEnabled,
                    } satisfies JournalEntry);
                }
                revision = counters.revision;
                await tx.put('meta', { key: COUNTERS_ID, ...counters } satisfies MetaCounters);
                return expired.length;
            },
        );
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
            if (note.origin === this._contextId) return;
            listener(note);
        });
    }

    private _publish(scope: MemoryScope | undefined, revision: number): void {
        this._storage.publish?.({
            kind    : 'facts-changed',
            scope   : scope ? { ...scope } : undefined,
            revision,
            origin  : this._contextId,
        });
    }

    private async _counters(tx: { get<T>(store: 'meta', id: string): Promise<T | undefined> }): Promise<MetaCounters> {
        const existing = await tx.get<MetaCounters>('meta', COUNTERS_ID);
        return existing ?? { key: COUNTERS_ID, seq: 0, revision: 0, opSeq: 0 };
    }
}

function dedupeById(records: StoredFactRecord[]): StoredFactRecord[] {
    const seen = new Set<string>();
    const out: StoredFactRecord[] = [];
    for (const record of records) {
        if (seen.has(record.id)) continue;
        seen.add(record.id);
        out.push(record);
    }
    return out;
}

/**
 * Index lookup. `undefined` means the named index is absent and the caller
 * must fall back. An empty array is a real miss — do not scan after it.
 * Multi-entry tag indexes can return the same record once per matching tag
 * occurrence, so hits are de-duplicated by id.
 */
async function indexedFacts(
    tx: StorageTransaction,
    indexName: string,
    query: string,
): Promise<StoredFactRecord[] | undefined> {
    if (!tx.getByIndex) return undefined;
    const hits = await tx.getByIndex<StoredFactRecord>('facts', indexName, query);
    return hits === undefined ? undefined : dedupeById(hits);
}
