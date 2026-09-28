/**
 * Retrieval layer — chunking, BM25, rank and score fusion, MMR, recency decay, the
 * similarity primitives they rest on, and the HybridRetriever.
 *
 * Zero-dependency and engine-free: this is the `@seanhogg/builderforce-memory/retrieval`
 * subpath, which a Cloudflare Worker, Node and the browser can all import without
 * pulling the WebGPU engine. Every host ranks with THESE functions rather than a copy.
 */

export { cosineSimilarity, jaccardSimilarity, tokenize } from '../similarity/index.js';

export { chunkText } from './chunk.js';
export type { Chunk, ChunkOptions } from './chunk.js';

export { bm25Search, bm25Idf, bm25LengthNorm, bm25TermScore, BM25_DEFAULT_K1, BM25_DEFAULT_B } from './bm25.js';
export type { Bm25Doc, Bm25Hit, Bm25Options, Bm25Tokenizer } from './bm25.js';

export { reciprocalRankFusion, rrfTerm, RRF_DEFAULT_K } from './fusion.js';
export type { RankedList, FusedHit } from './fusion.js';

export { mmrSelect, maximalMarginalRelevance, textMaximalMarginalRelevance } from './mmr.js';
export type { MmrSelectOptions, MmrCandidate, TextMmrItem } from './mmr.js';

export {
    DEFAULT_HYBRID_VECTOR_WEIGHT,
    DEFAULT_HYBRID_TEXT_WEIGHT,
    normalizeHybridWeights,
    hybridScore,
    bm25RankToScore,
    fuseHybridArms,
} from './weightedFusion.js';
export type { HybridWeights, HybridArmScores } from './weightedFusion.js';

export { toDecayLambda, calculateTemporalDecayMultiplier, applyTemporalDecayToScore } from './decay.js';

export { hybridRetrieve } from './HybridRetriever.js';
export type {
    RetrievalCandidate,
    HybridQuery,
    HybridRetrieveOptions,
    HybridHit,
} from './HybridRetriever.js';

export { HnswIndex, denseSearch } from './hnsw.js';
export type { HnswOptions, SearchHit } from './hnsw.js';
