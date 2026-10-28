/**
 * order.ts — the ONE total order every ranker in this package returns.
 *
 * Score descending, then id ascending. Scores are quantized before they are
 * compared, so float noise in the last bits (a CPU vs a WebGPU embedding, a
 * different summation order) cannot flip a near-tie: two scores in the same
 * quantum ARE tied, and the id decides. Without this, ties fell back to input
 * order, and input order is not part of any index's state.
 *
 * Quanta: dense (cosine) scores to 1e-4, BM25 / RRF and other exact-arithmetic
 * scores to 1e-6. Ids compare by UTF-16 code unit (`<`), never `localeCompare`,
 * which depends on the host's ICU data.
 *
 * Pure and zero-dependency.
 */

/** Quantum for cosine / dot-product scores — absorbs cross-platform float noise. */
export const DENSE_SCORE_QUANTUM = 1e-4;
/** Quantum for BM25, RRF and other scores computed in exact-enough arithmetic. */
export const LEXICAL_SCORE_QUANTUM = 1e-6;

/** The quantized bucket of `score`. NaN sorts last; ±Infinity keep their place. */
export function quantizeScore(score: number, quantum: number = LEXICAL_SCORE_QUANTUM): number {
    if (Number.isNaN(score)) return -Infinity;
    if (!Number.isFinite(score)) return score;
    return Math.round(score / quantum);
}

/** Ascending id order by code unit. Numeric ids (a doc index) compare numerically. */
export function compareIds(a: string | number, b: string | number): number {
    return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * The primitive behind {@link compareHits}, for callers whose items keep score and
 * id apart (a node index → id lookup inside a graph walk) — no object per compare.
 */
export function compareScoreThenId(
    scoreA: number,
    idA: string | number,
    scoreB: number,
    idB: string | number,
    quantum: number = LEXICAL_SCORE_QUANTUM,
): number {
    const qa = quantizeScore(scoreA, quantum);
    const qb = quantizeScore(scoreB, quantum);
    if (qa !== qb) return qb > qa ? 1 : -1;
    return compareIds(idA, idB);
}

/** A ranked item: anything with an id and a score. */
export interface ScoredId {
    id: string | number;
    score: number;
}

/**
 * Score descending (quantized), then id ascending. Usable directly as an
 * `Array.prototype.sort` comparator (the quantum then defaults to 1e-6).
 */
export function compareHits(a: ScoredId, b: ScoredId, quantum: number = LEXICAL_SCORE_QUANTUM): number {
    return compareScoreThenId(a.score, a.id, b.score, b.id, quantum);
}

/** {@link compareHits} at the dense quantum (1e-4) — for cosine-scored lists. */
export function compareDenseHits(a: ScoredId, b: ScoredId): number {
    return compareScoreThenId(a.score, a.id, b.score, b.id, DENSE_SCORE_QUANTUM);
}
