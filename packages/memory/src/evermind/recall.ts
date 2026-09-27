/**
 * evermind/recall.ts — the lexical recall rules for a project Evermind's taught memories.
 *
 * The gateway ranks a project's retained contributions against a query ("which learned
 * memory would answer this?"), and the chat client decides which recalled memories an
 * answer RECONCILES (restates enough of to supersede). Both must key on the SAME
 * meaningful terms, so the tokenizer, stopword set and both rules live here, once.
 *
 * Pure and dependency-free: safe in a Worker, in Node, and in a browser bundle.
 */

/** Tiny code+English stopword set — dropped so recall keys on meaningful terms. */
const RECALL_STOPWORDS: ReadonlySet<string> = new Set([
    'the', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'on', 'for', 'with', 'is', 'are',
    'be', 'as', 'at', 'by', 'it', 'this', 'that', 'from', 'you', 'your', 'i', 'we',
    'they', 'he', 'she', 'can', 'will', 'how', 'do', 'does', 'what', 'why', 'when',
    'which', 'use', 'using', 'used', 'please', 'need', 'want', 'me', 'my', 'so', 'if',
]);

/** Lowercase, split on non-word runs, drop stopwords and 1-char tokens. */
export function recallTokens(text: string): string[] {
    return (text.toLowerCase().match(/[a-z0-9_]+/g) ?? []).filter((w) => w.length >= 2 && !RECALL_STOPWORDS.has(w));
}

/** One contribution the ranker scores (a subset of a retained-contribution entry). */
export interface RecallScorable {
    id: number;
    kind: 'text' | 'delta';
    version: number;
    at: number;
    weight: number;
    prompt?: string;
    text?: string;
    /** The chat that contributed this memory, when one did. Absent means project-wide. */
    chatId?: number;
}

/** A scored recall match — the entry plus its 0..1 lexical relevance to the query. */
export interface RankedEvermindRecall extends RecallScorable {
    /** Lexical relevance to the query, 0..1 (rounded to 3 dp). */
    score: number;
}

function termFreq(tokens: string[]): Map<string, number> {
    const m = new Map<string, number>();
    for (const tok of tokens) m.set(tok, (m.get(tok) ?? 0) + 1);
    return m;
}

/** Cosine similarity between two TF maps (0..1). Iterates the smaller map. */
function tfCosine(a: Map<string, number>, b: Map<string, number>): number {
    let na = 0;
    for (const v of a.values()) na += v * v;
    let nb = 0;
    for (const v of b.values()) nb += v * v;
    if (na === 0 || nb === 0) return 0;
    const [small, big] = a.size <= b.size ? [a, b] : [b, a];
    let dot = 0;
    for (const [k, v] of small) {
        const w = big.get(k);
        if (w) dot += v * w;
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** Below this cosine a match is noise, not recall. */
const MIN_RECALL_SCORE = 0.02;

/**
 * Rank contributions by lexical relevance to `query`, best first. Only entries carrying
 * text can match (a weight delta has none); scores at or below the noise floor are
 * dropped, and the result is capped at `limit` (default 8).
 */
export function rankEvermindRecall(
    query: string,
    recent: readonly RecallScorable[],
    opts?: { limit?: number },
): RankedEvermindRecall[] {
    const q = termFreq(recallTokens(query));
    if (q.size === 0) return [];
    const scored: RankedEvermindRecall[] = [];
    for (const e of recent) {
        const hay = `${e.prompt ?? ''} ${e.text ?? ''}`.trim();
        if (!hay) continue;
        const score = tfCosine(q, termFreq(recallTokens(hay)));
        if (score > MIN_RECALL_SCORE) scored.push({ ...e, score: Math.round(score * 1000) / 1000 });
    }
    scored.sort((a, b) => b.score - a.score || b.at - a.at);
    return scored.slice(0, Math.max(1, opts?.limit ?? 8));
}

/** Stable, bounded (djb2, base36) hash of a recall query — keeps a cache key finite. */
export function hashRecallQuery(query: string): string {
    let h = 5381;
    for (let i = 0; i < query.length; i++) h = ((h << 5) + h + query.charCodeAt(i)) | 0;
    return (h >>> 0).toString(36);
}

/**
 * Fraction of a recalled memory's meaningful tokens an answer must restate for the
 * turn to count as RECONCILING (superseding) that memory.
 */
const RECONCILE_OVERLAP = 0.6;

/**
 * How many recalled memories `answer` reconciles — restates enough of that the
 * contributed turn supersedes them. A heuristic over token overlap; it surfaces the
 * reconcile step and never gates learning.
 */
export function countReconciledMemories(items: ReadonlyArray<{ text: string }>, answer: string): number {
    const ans = new Set(recallTokens(answer));
    if (ans.size === 0) return 0;
    let n = 0;
    for (const it of items) {
        const mem = new Set(recallTokens(it.text));
        if (mem.size === 0) continue;
        let hit = 0;
        for (const tok of mem) if (ans.has(tok)) hit++;
        if (hit / mem.size >= RECONCILE_OVERLAP) n++;
    }
    return n;
}
