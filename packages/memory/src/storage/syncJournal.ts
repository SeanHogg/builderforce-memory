/**
 * storage/syncJournal.ts – The sync-acknowledgement half of the journal (M04/M06).
 *
 * The journal holds only operations still owed to a sync target: entries are
 * written unsynced (and only when sync is enabled — see `factMutations.journal`)
 * and are DELETED once acknowledged, so the journal stays bounded by the
 * outstanding work instead of growing with every write ever made.
 */

import type { FactStoragePort, JournalEntry, StoredFactRecord, TombstoneRecord } from './types.js';

/** Journal entries not yet acknowledged by sync — the pending operation set. */
export async function pendingOps(storage: FactStoragePort): Promise<JournalEntry[]> {
    return storage.transact(['journal'], 'readonly', async (tx) => {
        const all = await tx.getAll<JournalEntry>('journal');
        return all.filter((j) => !j.synced);
    });
}

/**
 * Acknowledges synced operations: their journal entries are pruned, and the
 * facts/tombstones they produced clear `pendingSync`, making cap eviction
 * eligible again. Rows already marked synced (written by older versions,
 * which journaled every write even without sync) are pruned too.
 */
export async function markSynced(storage: FactStoragePort, opIds: string[]): Promise<void> {
    if (opIds.length === 0) return;
    const acked = new Set(opIds);
    await storage.transact(['journal', 'facts', 'tombstones'], 'readwrite', async (tx) => {
        const journal = await tx.getAll<JournalEntry>('journal');
        for (const entry of journal) {
            if (entry.synced || acked.has(entry.opId)) await tx.delete('journal', entry.opId);
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
