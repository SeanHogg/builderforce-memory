/**
 * Reciprocal Rank Fusion (RRF) — merges rankings (dense vector, sparse BM25, …) into
 * one list without needing their score scales to be commensurable: it fuses on
 * RANK, not raw score. The standard, parameter-light way to combine hybrid signals.
 * Score-based fusion lives in {@link ./weightedFusion}; diversity reranking in
 * {@link ./mmr}.
 *
 * Pure and zero-dependency.
 */

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

/** The canonical TREC damping constant for {@link rrfTerm}. */
export const RRF_DEFAULT_K = 60;

/**
 * One list's RRF contribution for the item at zero-based `rank`: `1 / (k + rank + 1)`.
 * Exported for a ranker that blends the rank signal into a score of its own rather
 * than fusing whole lists.
 */
export function rrfTerm(rank: number, k: number = RRF_DEFAULT_K): number {
    return 1 / (k + rank + 1);
}

/**
 * Reciprocal Rank Fusion over any number of ranked lists.
 * score(d) = Σ_lists weight / (k + rank(d)).  `k` (default 60) damps the
 * contribution of low-ranked items; the canonical TREC value.
 */
export function reciprocalRankFusion(lists: RankedList[], k = RRF_DEFAULT_K): FusedHit[] {
    const acc = new Map<string, number>();
    for (const list of lists) {
        const weight = list.weight ?? 1;
        list.ids.forEach((id, rank) => {
            acc.set(id, (acc.get(id) ?? 0) + weight * rrfTerm(rank, k));
        });
    }
    return [...acc.entries()]
        .map(([id, score]) => ({ id, score }))
        .sort((a, b) => b.score - a.score);
}
