/**
 * knowledge/stores/snapshotFormat.ts — the on-disk shapes of {@link LocalSnapshotStore}: the snapshot JSON and the
 * extract log's JSON Lines, and the pure functions that read them. No state, no persistence: the store owns both.
 */

import type { Extracted, ScopeRecord, StoredFact, StoredRelation } from '../model.js';

export interface KnowledgeSnapshot {
    version: 1;
    scopes: ScopeRecord[];
    facts: Record<string, StoredFact[]>;
    relations: Record<string, StoredRelation[]>;
    /** Present only when the store has no `extractedLog` (or in a snapshot written before the log existed). */
    extracted?: Record<string, Extracted[]>;
    /** Content hashes a replica asked the next review to re-check (see `KnowledgeStore.markRecheck`); absent when none. */
    recheck?: string[];
}

function asRecord<T>(v: unknown): Record<string, T[]> {
    return v && typeof v === 'object' && !Array.isArray(v) ? v as Record<string, T[]> : {};
}

export const emptySnapshot = (): KnowledgeSnapshot => ({ version: 1, scopes: [], facts: {}, relations: {} });

/** Reads a snapshot's text. THROWS when it does not parse, so the caller can keep the bytes aside before starting empty. */
export function parse(text: string | undefined): KnowledgeSnapshot {
    if (!text) return emptySnapshot();
    const s = JSON.parse(text) as Partial<KnowledgeSnapshot>;
    return {
        version: 1,
        scopes: Array.isArray(s.scopes) ? s.scopes : [],
        facts: asRecord(s.facts),
        relations: asRecord(s.relations),
        ...(s.extracted && typeof s.extracted === 'object' ? { extracted: s.extracted } : {}),
        ...(Array.isArray(s.recheck) ? { recheck: s.recheck.filter((h): h is string => typeof h === 'string') } : {}),
    };
}

export interface ParsedLog {
    entries: Map<string, Extracted[]>;
    /** The live (last) line of each hash, newline-terminated, in log order. */
    liveLines: Map<string, string>;
    liveBytes: number;
    /** Superseded lines, torn lines and blank lines. */
    deadBytes: number;
}

/**
 * The log's entries, last line per hash winning. A line that does not parse (torn by a crash or a failed
 * append) is skipped on its own: it never costs the other lines.
 */
export function parseLog(text: string | undefined): ParsedLog {
    const out: ParsedLog = { entries: new Map(), liveLines: new Map(), liveBytes: 0, deadBytes: 0 };
    if (!text) return out;
    for (const line of text.split('\n')) {
        if (!line.trim()) continue;
        try {
            const row = JSON.parse(line) as { h?: unknown; e?: unknown };
            if (typeof row.h === 'string' && Array.isArray(row.e)) {
                out.entries.set(row.h, row.e as Extracted[]);
                out.liveLines.delete(row.h); // re-set below so the map stays in log order of last occurrence
                out.liveLines.set(row.h, `${line}\n`);
            }
        } catch { /* torn line */ }
    }
    for (const l of out.liveLines.values()) out.liveBytes += l.length;
    out.deadBytes = Math.max(0, text.length - out.liveBytes);
    return out;
}
