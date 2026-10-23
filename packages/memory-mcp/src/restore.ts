/**
 * Restore — bringing the full text of a compaction stub back from an export.
 *
 * A stub written before compaction archived its originals is the ONLY copy left:
 * the text is gone (`contentLost`). The text often still exists elsewhere — a
 * memory export (`[{ key, content }]`), or the Evermind coordinator's `mem:` record
 * dump (`[{ id, text }]`) — and this module decides which record restores which
 * stub. Pure and dependency-free; the `memory_restore` tool does the I/O.
 *
 * Matching, in order:
 *   1. by key — a record's `key`, or its `id` with any `mem:` prefix dropped;
 *   2. by text — the record's first line starts with the stub's pointer line (the
 *      original first line the stub kept). Only a UNIQUE match restores; two stubs
 *      sharing a pointer are reported ambiguous rather than guessed.
 */

import { STUB_PREFIX, firstLine, isStub, stubPointer, type ArchivedFact } from "./compaction.js";

/** One record from a restore source, normalised. */
export interface RestoreRecord {
    /** The memory key the record names, when it names one. */
    key?: string;
    /** The full text. */
    content: string;
}

/** A stub that may be restored. */
export interface RestoreTarget {
    key: string;
    /** The stored stub body. */
    stub: string;
}

export interface RestoreMatch {
    key: string;
    content: string;
    via: "key" | "text";
}

export interface RestorePlan {
    matches: RestoreMatch[];
    /** Records that matched no target. */
    unmatched: number;
    /** Targets more than one stub could claim by text, left alone. */
    ambiguous: string[];
}

/** Array-valued properties a wrapped export may keep its rows under. */
const WRAPPER_KEYS = ["entries", "memories", "records", "items", "data", "facts"];

function rowsOf(parsed: unknown): unknown[] {
    if (Array.isArray(parsed)) return parsed;
    if (parsed && typeof parsed === "object") {
        const obj = parsed as Record<string, unknown>;
        for (const k of WRAPPER_KEYS) {
            const v = obj[k];
            if (Array.isArray(v)) return v;
        }
        // A lone record (a one-line JSON Lines file parses as plain JSON).
        if (typeof obj["content"] === "string" || typeof obj["text"] === "string") return [obj];
    }
    return [];
}

function toRecord(raw: unknown): RestoreRecord | null {
    if (!raw || typeof raw !== "object") return null;
    const r = raw as Record<string, unknown>;
    const content = typeof r["content"] === "string" ? r["content"] : typeof r["text"] === "string" ? r["text"] : null;
    if (content === null || !content.trim() || isStub(content)) return null;
    const named = typeof r["key"] === "string" ? r["key"] : typeof r["id"] === "string" ? r["id"] : undefined;
    const key = named?.replace(/^mem:/, "").trim();
    return key ? { key, content } : { content };
}

/**
 * Parse a restore source: a JSON array, an object wrapping one (`{ entries: [...] }`
 * and similar), or JSON Lines. Rows without text, or whose text is itself a stub,
 * are dropped — restoring a stub with a stub restores nothing.
 */
export function parseRestoreSource(text: string): RestoreRecord[] {
    let rows: unknown[];
    try {
        rows = rowsOf(JSON.parse(text));
    } catch {
        rows = [];
        for (const line of text.split(/\r?\n/)) {
            if (!line.trim()) continue;
            try {
                rows.push(JSON.parse(line));
            } catch {
                /* not a record */
            }
        }
    }
    return rows.map(toRecord).filter((r): r is RestoreRecord => r !== null);
}

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();

/** Shortest pointer line a text match may rest on — below it, a prefix proves nothing. */
export const MIN_POINTER_CHARS = 12;

/** Decide which record restores which stub. */
export function planRestore(records: RestoreRecord[], targets: RestoreTarget[]): RestorePlan {
    const byKey = new Map(targets.map((t) => [t.key, t]));
    const pointers = targets.map((t) => ({ key: t.key, pointer: norm(stubPointer(t.stub)) }));
    const matched = new Map<string, RestoreMatch>();
    const ambiguous = new Set<string>();
    let unmatched = 0;

    for (const r of records) {
        if (r.key && byKey.has(r.key)) {
            if (!matched.has(r.key)) matched.set(r.key, { key: r.key, content: r.content, via: "key" });
            continue;
        }
        const head = norm(firstLine(r.content, 100_000));
        const hits = pointers.filter((p) => p.pointer.length >= MIN_POINTER_CHARS && head.startsWith(p.pointer));
        if (hits.length === 1) {
            const key = hits[0]!.key;
            if (!matched.has(key)) matched.set(key, { key, content: r.content, via: "text" });
            continue;
        }
        if (hits.length > 1) for (const h of hits) ambiguous.add(h.key);
        else unmatched++;
    }
    // A key claimed unambiguously by key-match stays restored even if a text match was ambiguous.
    const matches = [...matched.values()];
    for (const m of matches) ambiguous.delete(m.key);
    return { matches, unmatched, ambiguous: [...ambiguous] };
}

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const STUB_VERSION = new RegExp(`^\\s*${escapeRegExp(STUB_PREFIX)} v(\\d+)\\]`);

/** The Evermind version a stub names (`[absorbed→Evermind v7] …` → 7), 0 when unreadable. */
export function stubVersion(stub: string): number {
    const m = STUB_VERSION.exec(stub);
    return m ? Number(m[1]) : 0;
}

/** The archive row that makes a restored stub lossless again. */
export function restoredArchive(match: RestoreMatch, stub: string, now = Date.now()): ArchivedFact {
    return { key: match.key, content: match.content, stub, version: stubVersion(stub), archivedAt: now };
}
