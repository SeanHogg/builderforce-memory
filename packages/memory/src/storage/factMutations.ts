/**
 * storage/factMutations.ts – The write primitives every repository mutation
 * shares, so the counter bump / tombstone / journal rules live in ONE place:
 *
 *   - {@link bumpOp}: next durable revision + operation id.
 *   - {@link journal}: append a sync-journal entry — ONLY when sync is enabled.
 *     Without a sync target there is nothing to acknowledge, so writing rows
 *     already marked synced just grew the journal forever.
 *   - {@link deleteWithTombstone}: delete a fact, tombstone it when sync is on,
 *     journal the delete — one call for forget, clear and TTL purge.
 *   - {@link clearTombstone}: a write resurrects a key (remember, import).
 */

import type {
    JournalEntry,
    MemoryScope,
    MetaCounters,
    StorageStoreName,
    StorageTransaction,
    TombstoneRecord,
} from './types.js';

export const COUNTERS_ID = 'counters';

/** The stores a mutating unit of work spans. */
export const WRITE_STORES: StorageStoreName[] = ['facts', 'tombstones', 'journal', 'meta'];

/** Who is writing, and under which retention rules. */
export interface MutationContext {
    readonly syncEnabled: boolean;
    readonly contextId: string;
    readonly now: () => number;
}

type CounterReader = Pick<StorageTransaction, 'get'>;

export async function readCounters(tx: CounterReader): Promise<MetaCounters> {
    const existing = await tx.get<MetaCounters>('meta', COUNTERS_ID);
    return existing ?? { key: COUNTERS_ID, seq: 0, revision: 0, opSeq: 0 };
}

export async function writeCounters(tx: StorageTransaction, counters: MetaCounters): Promise<void> {
    await tx.put('meta', { ...counters, key: COUNTERS_ID } satisfies MetaCounters);
}

/** The next revision + operation identity after `counters`. */
export function bumpOp(counters: MetaCounters, ctx: MutationContext): { counters: MetaCounters; opId: string } {
    const next = { ...counters, revision: counters.revision + 1, opSeq: counters.opSeq + 1 };
    return { counters: next, opId: `${ctx.contextId}:${next.opSeq}` };
}

/** Appends an unsynced journal entry when sync is enabled; a no-op otherwise. */
export async function journal(
    tx: StorageTransaction,
    ctx: MutationContext,
    entry: Omit<JournalEntry, 'synced'>,
): Promise<void> {
    if (!ctx.syncEnabled) return;
    await tx.put('journal', { ...entry, synced: false } satisfies JournalEntry);
}

/** Clears any tombstone for `id`; true when one existed (the write resurrected the key). */
export async function clearTombstone(tx: StorageTransaction, id: string): Promise<boolean> {
    const tombstone = await tx.get<TombstoneRecord>('tombstones', id);
    if (tombstone) await tx.delete('tombstones', id);
    return !!tombstone;
}

/**
 * Deletes one fact as a durable, propagating deletion: bump the counters,
 * delete, tombstone (sync on), journal (sync on). Returns the bumped counters;
 * the caller persists them once per transaction with {@link writeCounters}.
 */
export async function deleteWithTombstone(
    tx: StorageTransaction,
    ctx: MutationContext,
    record: { id: string; key: string; scope: MemoryScope },
    counters: MetaCounters,
    baseRevision?: number,
): Promise<MetaCounters> {
    const { counters: next, opId } = bumpOp(counters, ctx);
    const at = ctx.now();
    await tx.delete('facts', record.id);
    if (ctx.syncEnabled) {
        await tx.put('tombstones', {
            id         : record.id,
            key        : record.key,
            scope      : record.scope,
            deletedAt  : at,
            revision   : next.revision,
            opId,
            pendingSync: true,
        } satisfies TombstoneRecord);
    }
    await journal(tx, ctx, {
        opId,
        kind     : 'forget',
        id       : record.id,
        key      : record.key,
        scope    : record.scope,
        revision : next.revision,
        baseRevision,
        timestamp: at,
    });
    return next;
}
