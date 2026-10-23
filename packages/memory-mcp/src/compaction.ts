/**
 * Compaction — turning an ABSORBED memory into a terse pointer stub, LOSSLESSLY.
 *
 * Once a fact has been folded into an Evermind (or any downstream learner) the
 * full body has done its job: keeping it verbatim in the store means every
 * subsequent recall pays for text the model already carries in its weights.
 * Compaction replaces the body with a one-line `[absorbed→Evermind vN] <first
 * line>` pointer, so the fact stops filling context while it stays findable.
 *
 * Two rules make that safe:
 *   1. The full original is ARCHIVED before the body is stubbed — in a sidecar
 *      `<memory>.archive.json` beside the snapshot, never inside it, so it survives
 *      the snapshot round-trip, an external rewrite of the snapshot, and the
 *      store's entry cap / eviction. Recall resolves a stub back to its archive.
 *   2. A fact is stubbed only after a RECALL PROBE passes: a question generated
 *      from the fact is put to the local Evermind model, and an answer that does
 *      not carry the fact (or no model at all) leaves the fact uncompacted —
 *      "absorbed" is proven, not asserted.
 *
 * A stub with no archive is a fact whose text is gone (compacted before rule 1
 * existed): {@link isContentLost} names it, and every reader labels it with
 * {@link CONTENT_LOST_LABEL} until `memory_restore` brings the text back.
 *
 * The rules live here — not in the tool handler and not in a transport — because
 * BOTH the in-process `memory_compact` tool and any external compactor (the
 * BuilderForce VS Code extension) must produce byte-identical stubs and the same
 * archive; otherwise re-running either one re-stubs the other's work.
 *
 * Dependency-free and published as its own subpath
 * (`@seanhogg/builderforce-memory-mcp/compaction`), so an external compactor imports
 * these rules instead of re-implementing them — the marker below is the shared
 * idempotency contract between the two processes.
 */

/**
 * Marker that opens every compacted stub — also the idempotency guard. An entry
 * whose content already starts with this is skipped, so compacting twice (or
 * from two processes) never double-stubs.
 */
export const STUB_PREFIX = "[absorbed→Evermind";

/** Default cap on the pointer line carried inside a stub. */
export const DEFAULT_STUB_CHARS = 140;

/** True when a body is already a compaction stub. */
export function isStub(content: string): boolean {
    return content.trimStart().startsWith(STUB_PREFIX);
}

/** The first non-empty line of a body, trimmed to `max` chars — the stub's pointer text. */
export function firstLine(content: string, max = DEFAULT_STUB_CHARS): string {
    const line =
        content
            .split(/\r?\n/)
            .map((l) => l.trim())
            .find((l) => l.length > 0) ?? "";
    return line.length > max ? `${line.slice(0, max - 1).trimEnd()}…` : line;
}

/** Build the terse stub that replaces an absorbed entry's body. */
export function memoryStub(content: string, version: number, maxChars = DEFAULT_STUB_CHARS): string {
    return `${STUB_PREFIX} v${version}] ${firstLine(content, maxChars)}`;
}

/** The pointer text inside a stub (after the `] `), without a trailing ellipsis. */
export function stubPointer(stub: string): string {
    const s = stub.trim();
    const close = s.indexOf("] ");
    const pointer = isStub(s) && close >= 0 ? s.slice(close + 2) : s;
    return pointer.replace(/…$/, "").trim();
}

/** One entry considered for compaction. */
export interface CompactionCandidate {
    key: string;
    content: string;
}

/**
 * The full original of a compacted fact — what the archive keeps so a stub is
 * never the only copy. Written BEFORE the stub replaces the body.
 */
export interface ArchivedFact {
    key: string;
    /** The original, full body. */
    content: string;
    /** The stub that replaced it. */
    stub: string;
    /** Evermind version that absorbed it. */
    version: number;
    /** Unix ms the archive row was written. */
    archivedAt: number;
}

/** A body rewrite the caller should apply — archive `archive` FIRST, then write `content`. */
export interface CompactionWrite {
    key: string;
    /** The stub that replaces the entry's body. */
    content: string;
    /** Characters recovered by this rewrite. */
    bytesSaved: number;
    /** The original body, to be archived before the stub is written. */
    archive: ArchivedFact;
}

/** Why a candidate was left alone. */
export interface CompactionSkip {
    key: string;
    reason:
        | "not_found"
        | "already_compacted"
        | "not_smaller"
        /** No local model to probe with — the fact cannot be proven absorbed. */
        | "no_probe"
        /** The model could not answer the probe question generated from the fact. */
        | "probe_failed"
        /** The caller cannot archive the original, so stubbing would lose it. */
        | "no_archive";
}

/** The decided plan: what to write, what to leave, and the total recovered. */
export interface CompactionPlan {
    writes: CompactionWrite[];
    skipped: CompactionSkip[];
    bytesSaved: number;
}

/** Options controlling how a stub body is produced. */
export interface CompactionOptions {
    /** Explicit replacement body. Omit to generate a pointer from each entry's own first line. */
    stub?: string;
    /** Evermind version that absorbed the facts; appears in a generated stub. Default 0. */
    version?: number;
    /** Max characters of a generated pointer line. Default {@link DEFAULT_STUB_CHARS}. */
    maxChars?: number;
    /** Clock for {@link ArchivedFact.archivedAt}. Default `Date.now()`. */
    now?: number;
}

/**
 * Decide the STRUCTURAL compaction for a set of candidates: what is already a
 * stub, what would not shrink. Pure — no I/O — and NOT the whole contract: a
 * caller must still archive each write's `archive` before applying it, and prove
 * absorption — {@link planLosslessCompaction} runs this and then the probe.
 * `undefined` content means the key was not found.
 */
export function planCompaction(
    candidates: Array<CompactionCandidate | { key: string; content: undefined }>,
    opts: CompactionOptions = {},
): CompactionPlan {
    const version = opts.version ?? 0;
    const now = opts.now ?? Date.now();
    const maxChars = Math.max(20, opts.maxChars ?? DEFAULT_STUB_CHARS);
    const writes: CompactionWrite[] = [];
    const skipped: CompactionSkip[] = [];
    let bytesSaved = 0;

    for (const c of candidates) {
        const content = c.content;
        if (typeof content !== "string" || !content.trim()) {
            skipped.push({ key: c.key, reason: "not_found" });
            continue;
        }
        if (isStub(content)) {
            skipped.push({ key: c.key, reason: "already_compacted" });
            continue;
        }
        const stub = opts.stub ?? memoryStub(content, version, maxChars);
        // Never grow an entry — a "compaction" that costs tokens is a regression.
        if (stub.length >= content.length) {
            skipped.push({ key: c.key, reason: "not_smaller" });
            continue;
        }
        const saved = content.length - stub.length;
        writes.push({
            key: c.key,
            content: stub,
            bytesSaved: saved,
            archive: { key: c.key, content, stub, version, archivedAt: now },
        });
        bytesSaved += saved;
    }

    return { writes, skipped, bytesSaved };
}

// ── Recall probe ─────────────────────────────────────────────────────────────

/**
 * The local model compaction asks before it stubs a fact. `answer` returns the
 * model's text for a question, or null when it cannot answer at all.
 */
export interface RecallProbe {
    answer(question: string): Promise<string | null>;
}

/** A question generated from one fact, and the token a correct answer must carry. */
export interface ProbeQuestion {
    question: string;
    expected: string;
}

const PROBE_BLANK = "____";

/** Tokens of a line that a cloze question can blank: alphanumeric, 4+ chars. */
function probeTokens(line: string): string[] {
    return (line.match(/[A-Za-z0-9][A-Za-z0-9_.-]*[A-Za-z0-9]/g) ?? []).filter((t) => t.length >= 4);
}

/** How identifying a token is: digits and capitals beat length (a name/version IS the fact). */
function salience(token: string): number {
    return (/[0-9]/.test(token) ? 100 : 0) + (/[A-Z]/.test(token) ? 50 : 0) + token.length;
}

/**
 * A cloze question over the fact's pointer line: its most identifying token is
 * blanked and the model must supply it. Deterministic, so the same fact always
 * gets the same question. Null when the line has nothing worth blanking.
 */
export function probeQuestionFor(key: string, content: string): ProbeQuestion | null {
    const line = firstLine(content, 400);
    const tokens = probeTokens(line);
    if (tokens.length === 0) return null;
    let expected = tokens[0]!;
    for (const t of tokens) if (salience(t) > salience(expected)) expected = t;
    const at = line.indexOf(expected);
    const cloze = `${line.slice(0, at)}${PROBE_BLANK}${line.slice(at + expected.length)}`;
    return {
        question: `Stored memory "${key}": ${cloze}\nThe missing word (${PROBE_BLANK}) is:`,
        expected,
    };
}

/** True when the model's answer carries the blanked token (case-insensitive). */
export function probePasses(answer: string | null | undefined, expected: string): boolean {
    if (typeof answer !== "string" || !expected) return false;
    return answer.toLowerCase().includes(expected.toLowerCase());
}

/** Options for {@link planLosslessCompaction}. */
export interface LosslessCompactionOptions extends CompactionOptions {
    /** The local model to prove absorption with. Null/absent → nothing is stubbed. */
    probe?: RecallProbe | null;
    /** False when the caller cannot archive originals → nothing is stubbed. Default true. */
    canArchive?: boolean;
}

/**
 * The full compaction contract: {@link planCompaction}'s structural rules, then a
 * recall probe per surviving write. A write is kept only when the model answers
 * the question generated from its fact; otherwise the fact stays uncompacted and
 * the reason is recorded. The caller then archives every write's `archive` BEFORE
 * applying its stub.
 */
export async function planLosslessCompaction(
    candidates: Array<CompactionCandidate | { key: string; content: undefined }>,
    opts: LosslessCompactionOptions = {},
): Promise<CompactionPlan> {
    const structural = planCompaction(candidates, opts);
    const writes: CompactionWrite[] = [];
    const skipped: CompactionSkip[] = [...structural.skipped];
    for (const w of structural.writes) {
        if (opts.canArchive === false) {
            skipped.push({ key: w.key, reason: "no_archive" });
            continue;
        }
        if (!opts.probe) {
            skipped.push({ key: w.key, reason: "no_probe" });
            continue;
        }
        const q = probeQuestionFor(w.key, w.archive.content);
        let answer: string | null = null;
        if (q) {
            try {
                answer = await opts.probe.answer(q.question);
            } catch {
                answer = null;
            }
        }
        if (!q || !probePasses(answer, q.expected)) {
            skipped.push({ key: w.key, reason: "probe_failed" });
            continue;
        }
        writes.push(w);
    }
    return { writes, skipped, bytesSaved: writes.reduce((n, w) => n + w.bytesSaved, 0) };
}

// ── Archive + lost content ───────────────────────────────────────────────────

/**
 * Label every reader attaches to a stub whose original text no longer exists
 * anywhere (compacted before archiving existed).
 */
export const CONTENT_LOST_LABEL = "(content lost — re-verify before relying on it)";

/**
 * Where the archive lives: beside the snapshot, under the same name — the same
 * convention as the vector cache (`memory.json` → `memory.archive.json`).
 */
export function defaultArchiveFile(memoryFile: string): string {
    return memoryFile.replace(/\.json$/i, "") + ".archive.json";
}

/** True when `content` is a stub AND no archived original exists for it. */
export function isContentLost(content: string, archived: ArchivedFact | undefined): boolean {
    return isStub(content) && !archived;
}

/** On-disk archive: key → the latest archived original for that key. Never pruned. */
export interface ArchiveSnapshot {
    version: 1;
    facts: Record<string, ArchivedFact>;
}

/** Parse archive text; an absent/corrupt file is an empty archive. */
export function parseArchive(text: string | null | undefined): Record<string, ArchivedFact> {
    if (!text) return {};
    let parsed: Partial<ArchiveSnapshot> | null;
    try {
        parsed = JSON.parse(text) as Partial<ArchiveSnapshot> | null;
    } catch {
        return {};
    }
    const facts = parsed && typeof parsed === "object" ? parsed.facts : undefined;
    if (!facts || typeof facts !== "object") return {};
    // Null prototype: a memory key is arbitrary text, "__proto__" included.
    const out: Record<string, ArchivedFact> = Object.create(null) as Record<string, ArchivedFact>;
    for (const [key, f] of Object.entries(facts)) {
        if (f && typeof f.content === "string") out[key] = { ...f, key };
    }
    return out;
}

/**
 * Merge `facts` into archive text, returning the next text. Rows are only ever
 * added or replaced by a newer archive of the same key — never removed — so a
 * forgotten or evicted stub keeps its original.
 */
export function mergeArchive(text: string | null | undefined, facts: ArchivedFact[]): string {
    const all = parseArchive(text);
    for (const f of facts) all[f.key] = f;
    const snap: ArchiveSnapshot = { version: 1, facts: all };
    return `${JSON.stringify(snap, null, 2)}\n`;
}
