/**
 * Rank fusion + diversity reranking.
 *
 *   • Reciprocal Rank Fusion (RRF) merges the dense (vector) and sparse (BM25)
 *     rankings into one list without needing the two score scales to be
 *     commensurable — it fuses on RANK, not raw score. This is the standard,
 *     parameter-light way to combine hybrid retrieval signals.
 *   • Maximal Marginal Relevance (MMR) reranks the fused list to trade off
 *     relevance against novelty, so the top-k isn't five near-duplicate chunks.
 *
 * Pure and zero-dependency.
 */

import { cosineSimilarity } from '../similarity/index.js';

export interface RankedList {
    /** Ordered ids, most relevant first. */
    ids: string[];
    /** Optional weight for this list in the fusion (default 1). */
    weight?: number;
}

export interface FusedHit {
    id: string;
    score: number;
}

/**
 * Reciprocal Rank Fusion over any number of ranked lists.
 * score(d) = Σ_lists weight / (k + rank(d)).  `k` (default 60) damps the
 * contribution of low-ranked items; the canonical TREC value.
 */
export function reciprocalRankFusion(lists: RankedList[], k = 60): FusedHit[] {
    const acc = new Map<string, number>();
    for (const list of lists) {
        const weight = list.weight ?? 1;
        list.ids.forEach((id, rank) => {
            acc.set(id, (acc.get(id) ?? 0) + weight / (k + rank + 1));
        });
    }
    return [...acc.entries()]
        .map(([id, score]) => ({ id, score }))
        .sort((a, b) => b.score - a.score);
}

export interface MmrCandidate {
    id: string;
    vector: Float32Array;
}

/**
 * Maximal Marginal Relevance rerank. Greedily selects up to `topK` candidates,
 * each step maximising `λ·sim(query, d) − (1−λ)·max sim(d, already-selected)`.
 * λ=1 is pure relevance; lower λ injects diversity. Candidates without vectors
 * should be filtered out by the caller (they cannot be MMR-scored).
 */
export function maximalMarginalRelevance(
    queryVec: Float32Array,
    candidates: MmrCandidate[],
    topK: number,
    lambda = 0.7,
): string[] {
    const remaining = [...candidates];
    const selected: MmrCandidate[] = [];
    const relevance = new Map<string, number>();
    for (const c of remaining) relevance.set(c.id, cosineSimilarity(queryVec, c.vector));

    while (selected.length < topK && remaining.length > 0) {
        let bestIdx = 0;
        let bestScore = -Infinity;
        for (let i = 0; i < remaining.length; i++) {
            const c = remaining[i]!;
            let maxSimToSelected = 0;
            for (const s of selected) {
                const sim = cosineSimilarity(c.vector, s.vector);
                if (sim > maxSimToSelected) maxSimToSelected = sim;
            }
            const mmr = lambda * relevance.get(c.id)! - (1 - lambda) * maxSimToSelected;
            if (mmr > bestScore) { bestScore = mmr; bestIdx = i; }
        }
        selected.push(remaining.splice(bestIdx, 1)[0]!);
    }
    return selected.map(s => s.id);
}
