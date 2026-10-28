/**
 * HybridRetriever — dense + sparse retrieval with rank fusion and diversity rerank.
 *
 * This is the piece that takes the memory layer from "cosine-only similarity" to a
 * full hybrid RAG retriever:
 *
 *   1. Dense:  cosine over embeddings (SSM hidden-state vectors, or any embedder).
 *   2. Sparse: BM25 lexical scoring (catches exact tokens dense search misses).
 *   3. Fuse:   Reciprocal Rank Fusion combines the two rankings.
 *   4. Signals: optional importance weighting and half-life recency decay over
 *      the fused scores (off by default; memory recall turns both on).
 *   5. Rerank: optional MMR pass for relevance/novelty trade-off (diversity).
 *
 * It is storage-agnostic — give it candidates (id + text + optional vector) and a
 * query (text + optional vector). It degrades gracefully: no query vector / no
 * candidate vectors → BM25-only; no overlap → dense-only.
 */

import { bm25Search, type Bm25Options } from './bm25.js';
import { reciprocalRankFusion } from './fusion.js';
import { mmrSelect } from './mmr.js';
import { denseSearch } from './hnsw.js';
import { calculateTemporalDecayMultiplier } from './decay.js';
import { cosineSimilarity } from '../similarity/index.js';
import { compareHits } from './order.js';

export interface RetrievalCandidate {
    id: string;
    text: string;
    /** Precomputed embedding. Omit to exclude this candidate from the dense pass. */
    vector?: Float32Array;
    /** Importance 0–1, read only when {@link HybridRetrieveOptions.importanceWeight} is on. Default 0.5. */
    importance?: number;
    /** Unix-ms write time, read only when {@link HybridRetrieveOptions.recencyHalfLifeDays} is set. */
    timestamp?: number;
}

export interface HybridQuery {
    text: string;
    /** Query embedding. Omit for BM25-only retrieval. */
    vector?: Float32Array;
}

export interface HybridRetrieveOptions {
    /** Number of results to return. Default 5. */
    topK?: number;
    /** RRF damping constant. Default 60. */
    rrfK?: number;
    /** Relative weight of the dense ranking in fusion. Default 1. */
    denseWeight?: number;
    /** Relative weight of the sparse (BM25) ranking in fusion. Default 1. */
    sparseWeight?: number;
    /** Apply MMR diversity rerank over the fused top results. Default true. */
    rerank?: boolean;
    /** MMR relevance/diversity trade-off (1 = pure relevance). Default 0.7. */
    mmrLambda?: number;
    /** BM25 tuning. */
    bm25?: Bm25Options;
    /**
     * Candidate count at/above which the dense pass uses an HNSW ANN index
     * instead of an exact cosine scan (O(log N) vs O(N)). Default 256.
     */
    annThreshold?: number;
    /**
     * Multiply each fused score by `0.5 + importance` (candidate importance,
     * default 0.5 ⇒ ×1) — the same factor the cloud recall applies. Default false.
     */
    importanceWeight?: boolean;
    /**
     * Half-life, in days, of the recency decay applied to each fused score from
     * the candidate's `timestamp` (see `decay.ts`). 0/unset = no decay. A
     * candidate with no timestamp is not decayed.
     */
    recencyHalfLifeDays?: number;
    /** Clock for the recency decay (Unix ms). Default `Date.now()`. */
    now?: number;
}

const DAY_MS = 86_400_000;

/**
 * The ranking-signal factor for one candidate: importance × recency decay.
 * Exactly 1 when neither signal is enabled, so the pipeline is unchanged.
 */
function signalFactor(c: RetrievalCandidate, opts: HybridRetrieveOptions, now: number): number {
    let f = 1;
    if (opts.importanceWeight) {
        const imp = Number.isFinite(c.importance) ? c.importance! : 0.5;
        f *= 0.5 + Math.max(0, Math.min(1, imp));
    }
    const halfLife = opts.recencyHalfLifeDays ?? 0;
    if (halfLife > 0 && Number.isFinite(c.timestamp)) {
        f *= calculateTemporalDecayMultiplier({ ageInDays: (now - c.timestamp!) / DAY_MS, halfLifeDays: halfLife });
    }
    return f;
}

export interface HybridHit {
    id: string;
    text: string;
    /** Fused RRF score (pre-rerank), times the ranking-signal factor when signals are on. */
    score: number;
}

/**
 * Runs the full hybrid pipeline over `candidates` and returns the top-K hits.
 * Pure given its inputs (embeddings are supplied by the caller) so it is directly
 * unit-testable without a model or vector DB.
 */
export function hybridRetrieve(
    query: HybridQuery,
    candidates: RetrievalCandidate[],
    opts: HybridRetrieveOptions = {},
): HybridHit[] {
    const topK = opts.topK ?? 5;
    if (candidates.length === 0) return [];

    const byId = new Map(candidates.map(c => [c.id, c]));

    // ── Dense ranking (cosine, ANN-accelerated above the threshold) ────────────
    let denseIds: string[] = [];
    if (query.vector) {
        const vectored = candidates
            .filter((c): c is RetrievalCandidate & { vector: Float32Array } => !!c.vector && c.vector.length > 0)
            .map(c => ({ id: c.id, vector: c.vector }));
        // Rank the full vectored set (k = all) so fusion sees the complete dense
        // ordering; denseSearch uses an exact scan under the threshold and HNSW
        // above it, so behaviour is identical for small stores.
        denseIds = denseSearch(query.vector, vectored, vectored.length, opts.annThreshold ?? 256).map(h => h.id);
    }

    // ── Sparse ranking (BM25) ─────────────────────────────────────────────────
    const sparseIds = bm25Search(query.text, candidates, opts.bm25).map(h => h.id);

    // ── Fuse ──────────────────────────────────────────────────────────────────
    const fused = reciprocalRankFusion(
        [
            { ids: denseIds, weight: opts.denseWeight ?? 1 },
            { ids: sparseIds, weight: opts.sparseWeight ?? 1 },
        ].filter(l => l.ids.length > 0),
        opts.rrfK ?? 60,
    );
    if (fused.length === 0) return [];

    // ── Signals (importance × recency decay over the fused scores) ────────────
    const now = opts.now ?? Date.now();
    const factor = new Map(fused.map(f => [f.id, signalFactor(byId.get(f.id)!, opts, now)]));
    const weighted = fused
        .map(f => ({ id: f.id, score: f.score * factor.get(f.id)! }))
        .sort(compareHits);

    // ── Rerank (MMR over the weighted top, using whatever vectors we have) ────
    const rerank = opts.rerank ?? true;
    let orderedIds: string[];
    if (rerank && query.vector) {
        const queryVec = query.vector;
        // Consider a generous window so MMR has room to diversify.
        const window = weighted.slice(0, Math.max(topK * 4, topK));
        const mmrCands = window
            .map(f => byId.get(f.id))
            .filter((c): c is RetrievalCandidate & { vector: Float32Array } => !!c && !!c.vector && c.vector.length > 0);
        if (mmrCands.length > 0) {
            // Relevance is cosine to the query scaled by the same signal factor, so
            // MMR cannot undo the importance/recency weighting (factor 1 when off).
            const reranked = mmrSelect(mmrCands, {
                relevance: (c) => cosineSimilarity(queryVec, c.vector) * factor.get(c.id)!,
                similarity: (a, b) => cosineSimilarity(a.vector, b.vector),
                lambda: opts.mmrLambda ?? 0.7,
                topK,
                idOf: (c) => c.id,
            }).map(c => c.id);
            // MMR only ranks the vectored subset; append any remaining ids after.
            const seen = new Set(reranked);
            orderedIds = [...reranked, ...weighted.map(f => f.id).filter(id => !seen.has(id))];
        } else {
            orderedIds = weighted.map(f => f.id);
        }
    } else {
        orderedIds = weighted.map(f => f.id);
    }

    const score = new Map(weighted.map(f => [f.id, f.score]));
    return orderedIds
        .slice(0, topK)
        .map(id => ({ id, text: byId.get(id)!.text, score: score.get(id)! }));
}
