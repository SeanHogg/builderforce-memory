/**
 * memory/records.ts – Mapping between the public {@link MemoryEntry} shape and
 * the durable {@link StoredFactRecord}, plus the TTL and ordering rules every
 * MemoryStore read shares.
 */

import type { RememberRecordInput } from '../storage/ScopedFactRepository.js';
import type { MemoryScope, StoredFactRecord } from '../storage/types.js';
import type { MemoryEntry, RememberOptions } from './types.js';

/** True once `ttlMs` has elapsed since the entry's (source) timestamp. */
export function isExpired(entry: { timestamp: number; ttlMs?: number }): boolean {
    if (entry.ttlMs == null) return false;
    return Date.now() > entry.timestamp + entry.ttlMs;
}

/**
 * Newest first; same-millisecond ties break by the durable write sequence so
 * ordering is deterministic (and survives restarts).
 */
export function newestFirst(a: StoredFactRecord, b: StoredFactRecord): number {
    return b.timestamp - a.timestamp || b.seq - a.seq;
}

/** Non-expired records, newest first, as public entries. */
export function liveEntries(records: StoredFactRecord[]): MemoryEntry[] {
    return records.filter((r) => !isExpired(r)).sort(newestFirst).map(toEntry);
}

export function toEntry(record: StoredFactRecord): MemoryEntry {
    return {
        key        : record.key,
        content    : record.content,
        timestamp  : record.timestamp,
        seq        : record.seq,
        ttlMs      : record.ttlMs,
        type       : record.type,
        tags       : record.tags,
        importance : record.importance,
        revision   : record.revision,
        opId       : record.opId,
        scope      : { ...record.scope },
        provenance : record.provenance,
        pendingSync: record.pendingSync,
    };
}

/** A `remember` call's options as a repository write (store default TTL applied). */
export function rememberInput(content: string, opts: RememberOptions | undefined, defaultTtlMs: number | undefined): RememberRecordInput {
    return {
        content,
        timestamp : opts?.timestamp,
        ttlMs     : opts?.ttlMs ?? defaultTtlMs,
        type      : opts?.type,
        tags      : opts?.tags,
        importance: opts?.importance,
        provenance: opts?.provenance,
    };
}

/** An exported entry as an import record for `scope` (id/revision reassigned by the repository). */
export function toImportRecord(e: MemoryEntry, scope: MemoryScope): StoredFactRecord {
    return {
        id         : '',
        key        : e.key,
        scope,
        content    : e.content,
        timestamp  : e.timestamp,
        seq        : e.seq ?? 0,
        revision   : 0,
        opId       : e.opId ?? '',
        ttlMs      : e.ttlMs,
        type       : e.type,
        tags       : e.tags,
        importance : e.importance,
        provenance : e.provenance,
        pendingSync: false,
    };
}
