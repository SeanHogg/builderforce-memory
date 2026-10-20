/**
 * storage/factRetention.ts – Retention policy over the fact store (M06):
 * local cap eviction and intentional TTL purge.
 *
 *   - Cap eviction is cache management: synced records only, no tombstone, no
 *     journal. It counts in a READONLY transaction first and opens a read-write
 *     one only when the scope is actually over the cap — on the file adapter
 *     every read-write transaction rewrites and fsyncs the whole document, so
 *     an under-cap `remember` must not pay for a second one.
 *   - TTL purge is a deletion: each expiry goes through
 *     {@link deleteWithTombstone} so it propagates when sync is enabled.
 */

import { factsInScope } from './factQueries.js';
import {
    WRITE_STORES,
    deleteWithTombstone,
    readCounters,
    writeCounters,
    type MutationContext,
} from './factMutations.js';
import type { FactStoragePort, MemoryScope, StoredFactRecord } from './types.js';

type IsExpired = (record: StoredFactRecord) => boolean;

/** Lowest value first: expired, then least important, then oldest. */
function evictionOrder(isExpired: IsExpired): (a: StoredFactRecord, b: StoredFactRecord) => number {
    const rank = (e: StoredFactRecord): [number, number, number, number] =>
        [isExpired(e) ? 0 : 1, e.importance ?? 0.5, e.timestamp, e.seq];
    return (a, b) => {
        const ra = rank(a), rb = rank(b);
        for (let i = 0; i < ra.length; i++) {
            if (ra[i] !== rb[i]) return ra[i]! - rb[i]!;
        }
        return 0;
    };
}

/**
 * Removes the lowest-value SYNCED records in `scope` past `maxEntries`.
 * Returns the evicted keys (empty when under the cap — no write happens).
 */
export async function evictForCap(
    storage: FactStoragePort,
    scope: MemoryScope | undefined,
    maxEntries: number,
    isExpired: IsExpired,
): Promise<string[]> {
    const count = await storage.transact(['facts'], 'readonly', async (tx) => (await factsInScope(tx, scope)).length);
    if (count <= maxEntries) return [];
    return storage.transact(['facts'], 'readwrite', async (tx) => {
        // Re-read inside the write: another writer may have changed the set.
        const mine = await factsInScope(tx, scope);
        if (mine.length <= maxEntries) return [];
        const victims = mine
            .filter((r) => !r.pendingSync)
            .sort(evictionOrder(isExpired))
            .slice(0, Math.max(0, mine.length - maxEntries));
        for (const v of victims) await tx.delete('facts', v.id);
        return victims.map((v) => v.key);
    });
}

/**
 * Deletes every expired record in `scope` as a propagating deletion.
 * Returns how many were purged and the store revision after the purge.
 */
export async function purgeExpired(
    storage: FactStoragePort,
    ctx: MutationContext,
    scope: MemoryScope | undefined,
    isExpired: IsExpired,
): Promise<{ purged: number; revision: number }> {
    return storage.transact(WRITE_STORES, 'readwrite', async (tx) => {
        const expired = (await factsInScope(tx, scope)).filter(isExpired);
        if (expired.length === 0) return { purged: 0, revision: 0 };
        let counters = await readCounters(tx);
        for (const record of expired) counters = await deleteWithTombstone(tx, ctx, record, counters);
        await writeCounters(tx, counters);
        return { purged: expired.length, revision: counters.revision };
    });
}
