/**
 * hnsw.ts — a pure-TypeScript HNSW approximate-nearest-neighbour index.
 *
 * Recall over the fact store was an O(N) cosine scan per query, so the
 * "lifetime of always-current knowledge" thesis hit a wall around ~10⁴ facts.
 * HNSW (Hierarchical Navigable Small World) gives O(log N) expected query time
 * with high recall, and this implementation is dependency-free so it keeps the
 * engine/runtime zero-dep promise. For small sets an exact scan is both simpler
 * and faster, so callers use {@link denseSearch} which switches to HNSW only
 * above a threshold.
 *
 * Distance is cosine similarity (higher = closer); the graph stores `1 - cos`
 * as the metric so "nearer" is "smaller". Construction is deterministic given a
 * seed, so builds (and tests) reproduce exactly.
 */

import { cosineSimilarity } from '../similarity/index.js';

export interface HnswOptions {
    /** Max neighbours per node on layers > 0. Default 16. */
    M?: number;
    /** Candidate-list size during construction. Default 200. */
    efConstruction?: number;
    /** Default candidate-list size during search. Default 50. */
    efSearch?: number;
    /** Deterministic seed for layer assignment. Default 0x5eed. */
    seed?: number;
}

export interface SearchHit {
    id: string;
    /** Cosine similarity to the query (higher = closer). */
    score: number;
}

interface Node {
    id: string;
    vector: Float32Array;
    /** neighbours[layer] = list of node indices. */
    neighbours: number[][];
}

/** A small deterministic LCG → float in [0,1). Mirrors the engine's SeededRng. */
function makeRng(seed: number): () => number {
    let s = (seed >>> 0) || 1;
    return () => {
        s = (Math.imul(1664525, s) + 1013904223) >>> 0;
        return s / 0x1_0000_0000;
    };
}

export class HnswIndex {
    private readonly _M: number;
    private readonly _Mmax0: number;
    private readonly _efC: number;
    private readonly _efS: number;
    private readonly _mL: number;
    private readonly _rng: () => number;

    private readonly _nodes: Node[] = [];
    private readonly _idToIdx = new Map<string, number>();
    private _entry = -1; // index of the entry-point node
    private _maxLayer = -1;

    constructor(opts: HnswOptions = {}) {
        this._M = Math.max(2, opts.M ?? 16);
        this._Mmax0 = this._M * 2;
        this._efC = Math.max(this._M, opts.efConstruction ?? 200);
        this._efS = Math.max(1, opts.efSearch ?? 50);
        this._mL = 1 / Math.log(this._M);
        this._rng = makeRng(opts.seed ?? 0x5eed);
    }

    get size(): number {
        return this._nodes.length;
    }

    has(id: string): boolean {
        return this._idToIdx.has(id);
    }

    private _dist(a: Float32Array, b: Float32Array): number {
        return 1 - cosineSimilarity(a, b);
    }

    private _randomLayer(): number {
        return Math.floor(-Math.log(Math.max(this._rng(), 1e-12)) * this._mL);
    }

    /** Insert a vector. Ids are unique — re-adding a known id is ignored. */
    add(id: string, vector: Float32Array): void {
        if (this._idToIdx.has(id)) return;
        const layer = this._randomLayer();
        const node: Node = { id, vector, neighbours: Array.from({ length: layer + 1 }, () => []) };
        const idx = this._nodes.length;
        this._nodes.push(node);
        this._idToIdx.set(id, idx);

        if (this._entry === -1) {
            this._entry = idx;
            this._maxLayer = layer;
            return;
        }

        // Descend from the top to just above the new node's top layer.
        let ep = this._entry;
        for (let lc = this._maxLayer; lc > layer; lc--) {
            ep = this._greedyClosest(vector, ep, lc);
        }

        // From the node's top layer down to 0: find neighbours and link.
        for (let lc = Math.min(layer, this._maxLayer); lc >= 0; lc--) {
            const candidates = this._searchLayer(vector, ep, this._efC, lc);
            const Mmax = lc === 0 ? this._Mmax0 : this._M;
            const selected = candidates.slice(0, this._M).map((c) => c.idx);
            node.neighbours[lc] = selected.slice();
            // Add reverse links, pruning each neighbour back to Mmax.
            for (const nIdx of selected) {
                const nb = this._nodes[nIdx]!;
                const list = nb.neighbours[lc] ?? (nb.neighbours[lc] = []);
                list.push(idx);
                if (list.length > Mmax) this._prune(nb, lc, Mmax);
            }
            ep = candidates[0]?.idx ?? ep;
        }

        if (layer > this._maxLayer) {
            this._maxLayer = layer;
            this._entry = idx;
        }
    }

    /** Top-k nearest by cosine similarity (descending). */
    search(query: Float32Array, k: number, ef = this._efS): SearchHit[] {
        if (this._entry === -1 || k <= 0) return [];
        let ep = this._entry;
        for (let lc = this._maxLayer; lc > 0; lc--) {
            ep = this._greedyClosest(query, ep, lc);
        }
        const found = this._searchLayer(query, ep, Math.max(ef, k), 0);
        return found.slice(0, k).map((c) => ({ id: this._nodes[c.idx]!.id, score: 1 - c.dist }));
    }

    clear(): void {
        this._nodes.length = 0;
        this._idToIdx.clear();
        this._entry = -1;
        this._maxLayer = -1;
    }

    // ── internals ──────────────────────────────────────────────────────────────

    /** Greedy hill-climb to the single closest node to `q` on layer `lc`. */
    private _greedyClosest(q: Float32Array, start: number, lc: number): number {
        let best = start;
        let bestDist = this._dist(q, this._nodes[best]!.vector);
        let improved = true;
        while (improved) {
            improved = false;
            for (const nIdx of this._nodes[best]!.neighbours[lc] ?? []) {
                const d = this._dist(q, this._nodes[nIdx]!.vector);
                if (d < bestDist) {
                    bestDist = d;
                    best = nIdx;
                    improved = true;
                }
            }
        }
        return best;
    }

    /**
     * ef-search on a single layer: returns candidates sorted nearest-first.
     * A bounded best-first expansion (the standard HNSW layer search).
     */
    private _searchLayer(q: Float32Array, entry: number, ef: number, lc: number): Array<{ idx: number; dist: number }> {
        const visited = new Set<number>([entry]);
        const entryDist = this._dist(q, this._nodes[entry]!.vector);
        // `candidates` is a min-heap-ish frontier; `results` keeps the best ef.
        const candidates: Array<{ idx: number; dist: number }> = [{ idx: entry, dist: entryDist }];
        const results: Array<{ idx: number; dist: number }> = [{ idx: entry, dist: entryDist }];

        while (candidates.length > 0) {
            // Pop nearest candidate.
            let ci = 0;
            for (let i = 1; i < candidates.length; i++) if (candidates[i]!.dist < candidates[ci]!.dist) ci = i;
            const cur = candidates.splice(ci, 1)[0]!;
            // Farthest in results.
            const worst = results.reduce((m, r) => (r.dist > m.dist ? r : m), results[0]!);
            if (cur.dist > worst.dist && results.length >= ef) break;

            for (const nIdx of this._nodes[cur.idx]!.neighbours[lc] ?? []) {
                if (visited.has(nIdx)) continue;
                visited.add(nIdx);
                const d = this._dist(q, this._nodes[nIdx]!.vector);
                const farthest = results.reduce((m, r) => (r.dist > m.dist ? r : m), results[0]!);
                if (results.length < ef || d < farthest.dist) {
                    candidates.push({ idx: nIdx, dist: d });
                    results.push({ idx: nIdx, dist: d });
                    if (results.length > ef) {
                        // Drop the current farthest.
                        let wi = 0;
                        for (let i = 1; i < results.length; i++) if (results[i]!.dist > results[wi]!.dist) wi = i;
                        results.splice(wi, 1);
                    }
                }
            }
        }
        return results.sort((a, b) => a.dist - b.dist);
    }

    /** Keep only the `Mmax` closest neighbours of `node` on layer `lc`. */
    private _prune(node: Node, lc: number, Mmax: number): void {
        const list = node.neighbours[lc]!;
        list.sort((a, b) => this._dist(node.vector, this._nodes[a]!.vector) - this._dist(node.vector, this._nodes[b]!.vector));
        node.neighbours[lc] = list.slice(0, Mmax);
    }
}

/**
 * Dense KNN with an exact-scan fast path: below `threshold` candidates a linear
 * cosine scan is simpler and faster; at/above it, build an HNSW index and query
 * it. Returned hits are cosine-similarity descending. Pure given its inputs.
 */
export function denseSearch(
    query: Float32Array,
    items: Array<{ id: string; vector: Float32Array }>,
    k: number,
    threshold = 256,
    hnsw?: HnswOptions,
): SearchHit[] {
    if (items.length === 0 || k <= 0) return [];
    if (items.length < threshold) {
        return items
            .map((it) => ({ id: it.id, score: cosineSimilarity(query, it.vector) }))
            .sort((a, b) => b.score - a.score)
            .slice(0, k);
    }
    const index = new HnswIndex(hnsw);
    for (const it of items) index.add(it.id, it.vector);
    return index.search(query, k);
}
