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
 * as the metric so "nearer" is "smaller".
 *
 * Determinism: a node's level is {@link levelOf} its id — a geometric draw from
 * the id's sha-256, never a sequential RNG — and every ordering (candidates,
 * pruning, results) is {@link compareScoreThenId}: score, then id. So the graph
 * depends only on the vectors and the insertion order, and
 * {@link HnswIndex.canonicalBuild} (id-sorted inserts) removes the latter: the
 * same set of (id, vector) gives the same graph whatever order it arrived in.
 */

import { cosineSimilarity } from '../similarity/index.js';
import { compareScoreThenId, compareDenseHits, compareIds, DENSE_SCORE_QUANTUM } from './order.js';
import { sha256 } from './knowledge-index/sha256.js';

export interface HnswOptions {
    /** Max neighbours per node on layers > 0. Default 16. */
    M?: number;
    /** Candidate-list size during construction. Default 200. */
    efConstruction?: number;
    /** Default candidate-list size during search. Default 50. */
    efSearch?: number;
    /**
     * Retained for API compatibility. It no longer affects anything: levels come
     * from each id's hash ({@link levelOf}), so a build does not depend on a seed
     * or on how many inserts preceded a node.
     */
    seed?: number;
}

export interface SearchHit {
    id: string;
    /** Cosine similarity to the query (higher = closer). */
    score: number;
}

/** Any numeric vector: a float embedding, or int8 codes (cosine over codes is exact). */
export type HnswVector = ArrayLike<number>;

interface Node {
    id: string;
    vector: HnswVector;
    /** neighbours[layer] = list of node indices. */
    neighbours: number[][];
}

/** A read-only view of the built graph, for a consumer that stores it elsewhere (CSR). */
export interface HnswGraphView {
    /** Node ids, by node index (insertion order). */
    ids: readonly string[];
    /** neighbours[node][layer] = node indices. */
    neighbours: ReadonlyArray<ReadonlyArray<readonly number[]>>;
    entry: number;
    maxLayer: number;
}

/**
 * The HNSW level of `id`: a geometric draw with P(level ≥ L) = M^-L, taken from the
 * eight 32-bit words of sha-256(id) — each word below 2^32 / M is one more level.
 * Integer-only, so it is the same on every platform and independent of insertion.
 */
export function levelOf(id: string, M: number): number {
    const h = sha256(id);
    const view = new DataView(h.buffer, h.byteOffset, h.byteLength);
    const threshold = 0x1_0000_0000 / M;
    let level = 0;
    while (level < 8 && view.getUint32(level * 4) < threshold) level++;
    return level;
}

export class HnswIndex {
    private readonly _M: number;
    private readonly _Mmax0: number;
    private readonly _efC: number;
    private readonly _efS: number;

    private readonly _nodes: Node[] = [];
    private readonly _idToIdx = new Map<string, number>();
    private _entry = -1; // index of the entry-point node
    private _maxLayer = -1;

    constructor(opts: HnswOptions = {}) {
        this._M = Math.max(2, opts.M ?? 16);
        this._Mmax0 = this._M * 2;
        this._efC = Math.max(this._M, opts.efConstruction ?? 200);
        this._efS = Math.max(1, opts.efSearch ?? 50);
    }

    /**
     * Builds an index by inserting `items` in id order — the canonical build. The
     * same (id, vector) set yields the same graph regardless of the input order.
     * Ids are unique; for a duplicated id the first in input order wins.
     */
    static canonicalBuild(items: ReadonlyArray<{ id: string; vector: HnswVector }>, opts?: HnswOptions): HnswIndex {
        const index = new HnswIndex(opts);
        const sorted = items.map((it, i) => ({ it, i }))
            .sort((a, b) => compareIds(a.it.id, b.it.id) || a.i - b.i);
        for (const { it } of sorted) index.add(it.id, it.vector);
        return index;
    }

    get size(): number {
        return this._nodes.length;
    }

    has(id: string): boolean {
        return this._idToIdx.has(id);
    }

    /** The built graph, by node index — what a CSR store and pruning read. */
    graph(): HnswGraphView {
        return {
            ids: this._nodes.map((n) => n.id),
            neighbours: this._nodes.map((n) => n.neighbours),
            entry: this._entry,
            maxLayer: this._maxLayer,
        };
    }

    private _dist(a: HnswVector, b: HnswVector): number {
        return 1 - cosineSimilarity(a, b);
    }

    /** Nearest first; equal (quantized) distances by id ascending. */
    private readonly _byDist = (a: { idx: number; dist: number }, b: { idx: number; dist: number }): number =>
        compareScoreThenId(1 - a.dist, this._nodes[a.idx]!.id, 1 - b.dist, this._nodes[b.idx]!.id, DENSE_SCORE_QUANTUM);

    /** Insert a vector. Ids are unique — re-adding a known id is ignored. */
    add(id: string, vector: HnswVector): void {
        if (this._idToIdx.has(id)) return;
        const layer = levelOf(id, this._M);
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

    /** Top-k nearest by cosine similarity (descending; equal scores by id). */
    search(query: HnswVector, k: number, ef = this._efS): SearchHit[] {
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
    private _greedyClosest(q: HnswVector, start: number, lc: number): number {
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
    private _searchLayer(q: HnswVector, entry: number, ef: number, lc: number): Array<{ idx: number; dist: number }> {
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
        return results.sort(this._byDist);
    }

    /** Keep only the `Mmax` closest neighbours of `node` on layer `lc`. */
    private _prune(node: Node, lc: number, Mmax: number): void {
        const ranked = node.neighbours[lc]!.map((idx) => ({ idx, dist: this._dist(node.vector, this._nodes[idx]!.vector) }));
        node.neighbours[lc] = ranked.sort(this._byDist).slice(0, Mmax).map((r) => r.idx);
    }
}

/**
 * Dense KNN with an exact-scan fast path: below `threshold` candidates a linear
 * cosine scan is simpler and faster; at/above it, build an HNSW index (the
 * canonical, id-ordered build) and query it. Returned hits are in
 * {@link compareDenseHits} order. Pure given its inputs, and independent of their order.
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
            .sort(compareDenseHits)
            .slice(0, k);
    }
    return HnswIndex.canonicalBuild(items, hnsw).search(query, k);
}
