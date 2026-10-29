/**
 * The knowledge index — a deterministic, LEANN-style index over knowledge-graph
 * cards (design: Builderforce.ai/docs/design/deterministic-knowledge-index.md).
 *
 * Exact → lexical → dense cascade; int8 codes with integer traversal; below
 * `exactBelow` cards an exact int8 scan, at or above it a high-degree-preserving
 * pruned graph stored as CSR; float vectors recomputed
 * only for the candidates being reranked. Every result is a pure function of
 * `(digest(), query, opts)`.
 *
 * Runs in a Worker, the browser and Node: no Node-only API anywhere.
 */

import type { EmbedderPort, GraphNeighbours, IndexDoc, IndexHit, IndexMode, IndexParams, IndexTier, KnowledgeIndex, SimilarPair } from './types.js';
import { availableTiers, canonicalKeys, createState, denseReady, graphNode, indexKeys, type DocEntry, type IndexState } from './state.js';
import { EmbeddingCache } from './embeddings.js';
import { cascadeSearch, denseCandidates, type SearchContext } from './cascade.js';
import { compactState } from './compact.js';
import { expandHits } from './expand.js';
import { similarPairs } from './pairs.js';
import { hashDocText, stateDigest } from './digest.js';
import { deserializeState, serializeState } from './serialize.js';

class DeterministicKnowledgeIndex implements KnowledgeIndex {
    private readonly _cache: EmbeddingCache | undefined;
    private _digest: string | null = null;

    constructor(
        private readonly _state: IndexState,
        private readonly _embedder: EmbedderPort | undefined,
    ) {
        this._cache = _embedder ? new EmbeddingCache(_embedder) : undefined;
    }

    private get _ctx(): SearchContext {
        return { state: this._state, embedder: this._embedder, cache: this._cache };
    }

    async upsert(docs: IndexDoc[]): Promise<void> {
        const s = this._state;
        const changed: DocEntry[] = [];
        for (const d of docs) {
            const entry: DocEntry = { id: d.id, text: d.text, keys: canonicalKeys(d.keys), hash: d.hash ?? hashDocText(d.text) };
            const prev = s.docs.get(d.id);
            if (prev) indexKeys(s, prev, false);
            if (prev && prev.hash === entry.hash) {
                prev.keys = entry.keys;
                indexKeys(s, prev, true);
                continue;
            }
            s.docs.set(entry.id, entry);
            indexKeys(s, entry, true);
            s.lexical.set(entry.id, entry.text);
            changed.push(entry);
        }
        this._digest = null;
        if (!denseReady(s, this._embedder) || !this._cache) return;
        // Only the cards that are live and not already coded in the graph need a vector.
        const toEmbed = changed.filter((e) => s.docs.get(e.id) === e && graphNode(s, e.id) < 0);
        const vecs = await this._cache.vectors(toEmbed);
        toEmbed.forEach((e, i) => { e.vec = vecs[i]; });
    }

    remove(ids: string[]): void {
        const s = this._state;
        for (const id of ids) {
            const doc = s.docs.get(id);
            if (!doc) continue;
            indexKeys(s, doc, false);
            s.lexical.delete(id);
            s.docs.delete(id);
        }
        this._digest = null;
    }

    search(q: string, opts?: { k?: number; tiers?: IndexTier[] }): Promise<IndexHit[]> {
        return cascadeSearch(this._ctx, q, opts);
    }

    expand(hits: IndexHit[], graph: GraphNeighbours, opts?: { hops?: 1 | 2; edgeKinds?: string[] }): IndexHit[] {
        return expandHits(hits, graph, opts);
    }

    similarPairs(opts: { minSimilarity: number; maxPairs: number; idPrefix?: string }): SimilarPair[] {
        return similarPairs(this._state, opts);
    }

    async compact(): Promise<void> {
        await compactState(this._state, this._embedder, this._cache);
        this._digest = null;
    }

    digest(): string {
        return (this._digest ??= stateDigest(this._state));
    }

    tiers(): IndexTier[] {
        return availableTiers(this._state, this._embedder);
    }

    has(id: string): boolean {
        return this._state.docs.has(id);
    }

    size(): number {
        return this._state.docs.size;
    }

    serialize(): Uint8Array {
        return serializeState(this._state);
    }

    /**
     * Diagnostics: the dense candidate ids for `q` before the float rerank (the
     * integer graph walk plus not-yet-compacted cards). Empty without the dense tier.
     */
    async denseCandidates(q: string, n: number): Promise<string[]> {
        if (!denseReady(this._state, this._embedder) || !this._cache) return [];
        return denseCandidates(this._ctx, await this._cache.query(q), n);
    }

    /** Diagnostics: stored bytes of the int8 codes and the CSR graph, against float32. */
    storage(): KnowledgeIndexStorage {
        const c = this._state.compacted;
        if (!c) return { mode: null, nodes: 0, codeBytes: 0, graphBytes: 0, float32Bytes: 0 };
        let codeBytes = 0;
        for (const code of c.codes) codeBytes += code?.byteLength ?? 0;
        let graphBytes = 0;
        for (const l of c.graph.layers) graphBytes += (l.nodes?.byteLength ?? 0) + l.offsets.byteLength + l.neighbours.byteLength;
        return { mode: c.mode, nodes: c.ids.length, codeBytes, graphBytes, float32Bytes: c.ids.length * this._state.dims * 4 };
    }
}

export interface KnowledgeIndexStorage {
    /** How the dense tier searches the codes; null before the first compaction with an embedder. */
    mode: IndexMode | null;
    nodes: number;
    codeBytes: number;
    graphBytes: number;
    /** What the same nodes would take as float32 vectors. */
    float32Bytes: number;
}

/** The concrete index type: the contract plus diagnostics. */
export type DeterministicKnowledgeIndexType = KnowledgeIndex & {
    denseCandidates(q: string, n: number): Promise<string[]>;
    storage(): KnowledgeIndexStorage;
};

export function createKnowledgeIndex(
    opts: { embedder?: EmbedderPort; params?: Partial<IndexParams> } = {},
): DeterministicKnowledgeIndexType {
    return new DeterministicKnowledgeIndex(createState(opts.params, opts.embedder), opts.embedder);
}

/**
 * Loads a serialized index. With no embedder, or one of another fingerprint, the
 * dense tier is absent until `compact()` re-embeds every card under the new one.
 */
export function loadKnowledgeIndex(bytes: Uint8Array, embedder?: EmbedderPort): DeterministicKnowledgeIndexType {
    return new DeterministicKnowledgeIndex(deserializeState(bytes), embedder);
}

export type { EmbedderPort, GraphNeighbours, IndexDoc, IndexHit, IndexMode, IndexParams, IndexTier, KnowledgeIndex, SimilarPair } from './types.js';
export { EXACT_PAIR_BUDGET } from './scan.js';
export { DEFAULT_INDEX_PARAMS } from './state.js';
export { codeTokenize } from './tokenize.js';
export { hashDocText } from './digest.js';
export { sha256, sha256Hex, Sha256 } from './sha256.js';
export { EXPAND_HOP_DECAY } from './expand.js';
