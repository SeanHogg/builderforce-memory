/**
 * storage/factQueries.ts – Read helpers shared by the repository and its
 * retention policy: partition filtering and index lookups with the
 * "missing index ⇒ fall back, empty hit ⇒ real miss" contract.
 */

import { scopePartitionKey, type MemoryScope, type StorageTransaction, type StoredFactRecord } from './types.js';

/** Every fact in exactly `scope`'s partition (never wider, never narrower). */
export async function factsInScope(tx: StorageTransaction, scope: MemoryScope | undefined): Promise<StoredFactRecord[]> {
    const partition = scopePartitionKey(scope);
    const all = await tx.getAll<StoredFactRecord>('facts');
    return all.filter((r) => scopePartitionKey(r.scope) === partition);
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
export async function indexedFacts(
    tx: StorageTransaction,
    indexName: string,
    query: string,
): Promise<StoredFactRecord[] | undefined> {
    if (!tx.getByIndex) return undefined;
    const hits = await tx.getByIndex<StoredFactRecord>('facts', indexName, query);
    return hits === undefined ? undefined : dedupeById(hits);
}
