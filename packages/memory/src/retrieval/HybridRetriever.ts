/**
 * HybridRetriever — dense + sparse retrieval with rank fusion and diversity rerank.
 *
 * This is the piece that takes the memory layer from "cosine-only similarity" to a
 * full hybrid RAG retriever:
 *
 *   1. Dense:  cosine over embeddings (SSM hidden-state vectors, or any embedder).
 *   2. Sparse: BM25 lexical scoring (catches exact tokens dense search misses).
 *   3. Fuse:   Reciprocal Rank Fusion combines the two rankings.
 *   4. Rerank: optional MMR pass for relevance/novelty trade-off (diversity).
 *
 * It is storage-agnostic — give it candidates (id + text + optional vector) and a
 * query (text + optional vector). It degrades gracefully: no query vector / no
 * candidate vectors → BM25-only; no overlap → dense-only.
 */

import { cosineSimilarity } from '../similarity/index.js';
import { bm25Search, type Bm25Options } from './bm25.js';
import { reciprocalRankFusion, maximalMarginalRelevance, type MmrCandidate } from './fusion.js';

export interface RetrievalCandidate {
    id: string;
    text: string;
    /** Precomputed embedding. Omit to exclude this candidate from the dense pass. */
    vector?: Float32Array;
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
}

export interface HybridHit {
    id: string;
    text: string;
    /** Fused RRF score (pre-rerank). */
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

    // ── Dense ranking (cosine) ────────────────────────────────────────────────
    let denseIds: string[] = [];
    if (query.vector) {
        denseIds = candidates
            .filter(c => c.vector && c.vector.length > 0)
            .map(c => ({ id: c.id, score: cosineSimilarity(query.vector!, c.vector!) }))
            .sort((a, b) => b.score - a.score)
            .map(h => h.id);
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

    // ── Rerank (MMR over fused top, using whatever vectors we have) ────────────
    const rerank = opts.rerank ?? true;
    let orderedIds: string[];
    if (rerank && query.vector) {
        // Consider a generous fused window so MMR has room to diversify.
        const window = fused.slice(0, Math.max(topK * 4, topK));
        const mmrCands: MmrCandidate[] = window
            .map(f => byId.get(f.id))
            .filter((c): c is RetrievalCandidate => !!c && !!c.vector && c.vector.length > 0)
            .map(c => ({ id: c.id, vector: c.vector! }));
        if (mmrCands.length > 0) {
            const reranked = maximalMarginalRelevance(query.vector, mmrCands, topK, opts.mmrLambda ?? 0.7);
            // MMR only ranks the vectored subset; append any remaining fused ids after.
            const seen = new Set(reranked);
            orderedIds = [...reranked, ...fused.map(f => f.id).filter(id => !seen.has(id))];
        } else {
            orderedIds = fused.map(f => f.id);
        }
    } else {
        orderedIds = fused.map(f => f.id);
    }

    const fusedScore = new Map(fused.map(f => [f.id, f.score]));
    return orderedIds
        .slice(0, topK)
        .map(id => ({ id, text: byId.get(id)!.text, score: fusedScore.get(id)! }));
}
