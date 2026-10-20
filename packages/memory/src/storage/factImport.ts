/**
 * storage/factImport.ts – Import INTO a scope (M03/M06): source timestamp,
 * sequence, tags, importance, TTL and provenance survive; `importedFrom` is
 * stamped onto the provenance; imported facts are `pendingSync` when sync is
 * enabled (new to THIS replica even if another store already had them).
 */

import { WRITE_STORES, bumpOp, clearTombstone, journal, readCounters, writeCounters, type MutationContext } from './factMutations.js';
import { scopedRecordId, type FactStoragePort, type MemoryScope, type StoredFactRecord } from './types.js';

export interface ImportResult {
    written: number;
    skipped: number;
}

/**
 * Writes `records` into `scope` in one transaction. 'merge' keeps the record
 * with the newer source timestamp; 'overwrite' always writes. Returns the
 * counts plus the store revision after the import (0 when nothing was written).
 */
export async function importInto(
    storage: FactStoragePort,
    ctx: MutationContext,
    scope: MemoryScope | undefined,
    records: StoredFactRecord[],
    strategy: 'merge' | 'overwrite',
    importedFrom?: string,
): Promise<ImportResult & { revision: number }> {
    return storage.transact(WRITE_STORES, 'readwrite', async (tx) => {
        let counters = await readCounters(tx);
        let written = 0;
        let skipped = 0;
        for (const incoming of records) {
            const id = scopedRecordId(scope, incoming.key);
            const existing = await tx.get<StoredFactRecord>('facts', id);
            if (strategy === 'merge' && existing && existing.timestamp >= incoming.timestamp) {
                skipped++;
                continue;
            }
            const bumped = bumpOp(counters, ctx);
            counters = { ...bumped.counters, seq: Math.max(counters.seq + 1, incoming.seq ?? 0) };
            const record: StoredFactRecord = {
                ...incoming,
                id,
                key         : incoming.key,
                scope       : { ...(scope ?? {}) },
                seq         : incoming.seq ?? counters.seq,
                revision    : counters.revision,
                opId        : bumped.opId,
                provenance  : {
                    ...incoming.provenance,
                    ...(importedFrom ? { importedFrom } : {}),
                    originalTimestamp: incoming.provenance?.originalTimestamp ?? incoming.timestamp,
                },
                pendingSync : ctx.syncEnabled,
            };
            await tx.put('facts', record);
            await clearTombstone(tx, id);
            await journal(tx, ctx, {
                opId     : bumped.opId,
                kind     : 'remember',
                id,
                key      : incoming.key,
                scope    : record.scope,
                revision : counters.revision,
                timestamp: record.timestamp,
            });
            written++;
        }
        if (written > 0) await writeCounters(tx, counters);
        return { written, skipped, revision: written > 0 ? counters.revision : 0 };
    });
}
