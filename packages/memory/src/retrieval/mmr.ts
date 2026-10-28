/**
 * Maximal Marginal Relevance — ONE greedy selection loop, two similarity bases.
 *
 * MMR reranks a relevance-ordered list so the top of it is not five near-duplicates:
 * each step picks the candidate maximising `λ·relevance − (1−λ)·max sim(candidate,
 * already-selected)`. λ=1 is pure relevance; lower λ buys diversity.
 *
 * What differs between callers is only what "similar" means:
 *   • {@link maximalMarginalRelevance} — cosine over embedding vectors, relevance =
 *     cosine to the query vector. For retrievers that hold vectors;
 *   • {@link textMaximalMarginalRelevance} — Jaccard over word tokens, relevance =
 *     the candidate's own score min-max normalised onto [0,1]. For retrievers that
 *     only have text and a fused score (an FTS5 + sqlite-vec hybrid, say).
 *
 * Both run {@link mmrSelect}, so the selection rule itself exists once.
 *
 * Pure and zero-dependency.
 */

import { cosineSimilarity, jaccardSimilarity, tokenize } from '../similarity/index.js';
import { compareHits, compareIds, quantizeScore, DENSE_SCORE_QUANTUM } from './order.js';

export interface MmrSelectOptions<T> {
    /** Relevance of one candidate, on the same scale as {@link similarity}. */
    relevance: (item: T) => number;
    /** Similarity of two candidates. */
    similarity: (a: T, b: T) => number;
    /** 0 = maximum diversity, 1 = maximum relevance. Default 0.7. */
    lambda?: number;
    /** Stop after this many picks. Default: every candidate. */
    topK?: number;
    /**
     * Breaks an MMR tie in favour of the higher value. Ties are judged on the MMR
     * score quantized to {@link quantum}.
     */
    tieScore?: (item: T) => number;
    /**
     * The candidate's id — the last tie-break, ascending. Without it the explicit
     * id is the candidate's index in `items`, so the earlier candidate wins.
     */
    idOf?: (item: T) => string;
    /** Quantum the MMR score is compared at. Default 1e-4 (cosine noise). */
    quantum?: number;
}

/** Greedy MMR selection. Returns the picked candidates in pick order. */
export function mmrSelect<T>(items: readonly T[], opts: MmrSelectOptions<T>): T[] {
    const lambda = opts.lambda ?? 0.7;
    const quantum = opts.quantum ?? DENSE_SCORE_QUANTUM;
    const limit = Math.min(opts.topK ?? items.length, items.length);
    const remaining = [...items];
    const relevance = new Map<T, number>();
    const position = new Map<T, number>();
    items.forEach((item, i) => {
        relevance.set(item, opts.relevance(item));
        if (!position.has(item)) position.set(item, i);
    });
    const idOf = (item: T): string | number => (opts.idOf ? opts.idOf(item) : position.get(item)!);
    // Higher quantized MMR score, then higher tieScore, then lower id.
    const beats = (a: T, aScore: number, b: T, bScore: number): boolean => {
        const qa = quantizeScore(aScore, quantum);
        const qb = quantizeScore(bScore, quantum);
        if (qa !== qb) return qa > qb;
        if (opts.tieScore) {
            const ta = opts.tieScore(a);
            const tb = opts.tieScore(b);
            if (ta !== tb) return ta > tb;
        }
        return compareIds(idOf(a), idOf(b)) < 0;
    };

    const selected: T[] = [];
    while (selected.length < limit && remaining.length > 0) {
        let bestIdx = -1;
        let bestScore = -Infinity;
        for (let i = 0; i < remaining.length; i++) {
            const candidate = remaining[i]!;
            let maxSimToSelected = 0;
            for (const s of selected) {
                const sim = opts.similarity(candidate, s);
                if (sim > maxSimToSelected) maxSimToSelected = sim;
            }
            const score = lambda * relevance.get(candidate)! - (1 - lambda) * maxSimToSelected;
            if (bestIdx < 0 || beats(candidate, score, remaining[bestIdx]!, bestScore)) { bestScore = score; bestIdx = i; }
        }
        if (bestIdx < 0) break;
        selected.push(remaining.splice(bestIdx, 1)[0]!);
    }
    return selected;
}

export interface MmrCandidate {
    id: string;
    vector: Float32Array;
}

/**
 * Vector MMR. Selects up to `topK` candidates by cosine relevance to `queryVec`,
 * penalising cosine similarity to what is already selected. Candidates without
 * vectors should be filtered out by the caller (they cannot be MMR-scored).
 */
export function maximalMarginalRelevance(
    queryVec: Float32Array,
    candidates: MmrCandidate[],
    topK: number,
    lambda = 0.7,
): string[] {
    return mmrSelect(candidates, {
        relevance: (c) => cosineSimilarity(queryVec, c.vector),
        similarity: (a, b) => cosineSimilarity(a.vector, b.vector),
        lambda,
        topK,
        idOf: (c) => c.id,
    }).map((c) => c.id);
}

export interface TextMmrItem {
    id: string;
    /** The candidate's fused relevance score (any scale — it is normalised here). */
    score: number;
    /** The text diversity is judged on. */
    content: string;
}

/**
 * Text MMR. Reorders every item, judging diversity by Jaccard similarity of word
 * tokens, and relevance by the item's `score` min-max normalised onto [0,1] (all
 * equal → 1). A tie goes to the higher raw score, then the lower id. `lambda` is
 * clamped to [0,1]; at 1 this is a plain {@link compareHits} sort.
 */
export function textMaximalMarginalRelevance<T extends TextMmrItem>(items: readonly T[], lambda = 0.7): T[] {
    if (items.length <= 1) return [...items];
    const l = Math.max(0, Math.min(1, lambda));
    if (l === 1) return [...items].sort(compareHits);

    const tokens = new Map<string, Set<string>>();
    for (const item of items) tokens.set(item.id, new Set(tokenize(item.content)));
    const tokensOf = (item: T) => tokens.get(item.id) ?? new Set(tokenize(item.content));

    const scores = items.map((i) => i.score);
    const min = Math.min(...scores);
    const range = Math.max(...scores) - min;

    return mmrSelect(items, {
        relevance: (item) => (range === 0 ? 1 : (item.score - min) / range),
        similarity: (a, b) => jaccardSimilarity(tokensOf(a), tokensOf(b)),
        lambda: l,
        tieScore: (item) => item.score,
        idOf: (item) => item.id,
    });
}
