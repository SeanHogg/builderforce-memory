/**
 * The knowledge index contract — a deterministic, LEANN-style index over knowledge-graph
 * cards (design: Builderforce.ai/docs/design/deterministic-knowledge-index.md).
 *
 * Every result is a pure function of `(digest, query, opts)`: one total order
 * (`compareHits`), an insertion-order-free graph, integer (int8) traversal similarity.
 */

/** One card. `id` is stable (a knowledge-graph NodeRef); `hash` defaults to a hash of `text`. */
export interface IndexDoc {
    id: string;
    text: string;
    /** Exact-match keys (path, basename, symbol, heading slug, entity name) — case-folded on insert. */
    keys?: string[];
    hash?: string;
}

/** The embedder seam. Vectors from two fingerprints are never compared. */
export interface EmbedderPort {
    /** model id + version + dims. */
    fingerprint: string;
    dims: number;
    embed(texts: string[]): Promise<Float32Array[]>;
}

export type IndexTier = 'exact' | 'lexical' | 'dense';

export interface IndexHit {
    id: string;
    score: number;
    tier: IndexTier;
}

/** Neighbours of a node, supplied by the knowledge graph for recall expansion. */
export interface GraphNeighbours {
    neighbours(id: string, edgeKinds?: readonly string[]): ReadonlyArray<{ id: string; kind: string; weight: number }>;
}

export interface IndexParams {
    /** HNSW max neighbours for hubs (non-hubs keep m/2 after pruning). Default 16. */
    m: number;
    efConstruction: number;
    efSearch: number;
    /** Share of nodes (by in-degree) that keep full degree. Default 0.02. */
    hubShare: number;
    /** Dense candidates = k * rerankFactor, recomputed in float and reranked. Default 4. */
    rerankFactor: number;
    /** Drop int8 codes of non-hub nodes and recompute on the search path. Default false. */
    lean: boolean;
    /** RRF constant. Default 60. */
    rrfK: number;
}

export interface SimilarPair {
    a: string;
    b: string;
    similarity: number;
}

export interface KnowledgeIndex {
    /** Inserts or replaces; skips docs whose hash is unchanged. */
    upsert(docs: IndexDoc[]): Promise<void>;
    remove(ids: string[]): void;
    /** Cascade: exact → lexical → dense (when an embedder of the stored fingerprint is present) → RRF → compareHits. */
    search(q: string, opts?: { k?: number; tiers?: IndexTier[] }): Promise<IndexHit[]>;
    /** Adds graph neighbours (score × weight × 0.5 per hop), re-ordered by compareHits. */
    expand(hits: IndexHit[], graph: GraphNeighbours, opts?: { hops?: 1 | 2; edgeKinds?: string[] }): IndexHit[];
    /** Deterministic near-duplicate pairs over int8 codes; `idPrefix` filters (e.g. `src:`). */
    similarPairs(opts: { minSimilarity: number; maxPairs: number; idPrefix?: string }): SimilarPair[];
    /** Canonical rebuild (ids sorted) — the same doc set + fingerprint ⇒ the same bytes. */
    compact(): Promise<void>;
    /** sha-256 hex over the canonical state. */
    digest(): string;
    /** Tiers available right now (dense only with a matching embedder). */
    tiers(): IndexTier[];
    size(): number;
    serialize(): Uint8Array;
}
