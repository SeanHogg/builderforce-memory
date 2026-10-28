/**
 * csr.ts — the knowledge index's graph, stored as CSR (compressed sparse rows).
 *
 * The graph is built by the package's HNSW over int8 codes (canonical, id-ordered
 * inserts, hash-derived levels), its base layer is pruned (`prune.ts`), and every
 * layer is flattened into a `Uint32Array` of offsets plus a `Uint32Array` of
 * neighbours: no per-node arrays, a fixed byte layout to serialize and hash.
 *
 * Layer 0 holds every node and is indexed by node directly. An upper layer holds
 * only the nodes that reach it, listed ascending in `nodes`.
 */

import { HnswIndex } from '../hnsw.js';
import type { IndexParams } from './types.js';
import { pruneBaseLayer } from './prune.js';
import { cosineInt8 } from './quantize.js';

export interface CsrLayer {
    /** Upper layers: the node indices present, ascending. Absent on layer 0 (every node). */
    nodes?: Uint32Array;
    /** Row `r`'s neighbours are `neighbours[offsets[r] .. offsets[r + 1])`. */
    offsets: Uint32Array;
    neighbours: Uint32Array;
}

export interface CsrGraph {
    /** Entry node for search; -1 for an empty graph. */
    entry: number;
    /** layers[0] is the base layer; layers.length - 1 is the top layer. */
    layers: CsrLayer[];
}

export const EMPTY_GRAPH: CsrGraph = Object.freeze({ entry: -1, layers: [] }) as CsrGraph;

/** Flattens per-row neighbour lists into one CSR layer. */
export function toCsrLayer(rows: ReadonlyArray<readonly number[]>, nodes?: Uint32Array): CsrLayer {
    const offsets = new Uint32Array(rows.length + 1);
    let total = 0;
    rows.forEach((row, r) => {
        total += row.length;
        offsets[r + 1] = total;
    });
    const neighbours = new Uint32Array(total);
    rows.forEach((row, r) => neighbours.set(row, offsets[r]!));
    return nodes ? { nodes, offsets, neighbours } : { offsets, neighbours };
}

/** The row of `node` in `layer`, or -1 when the node is not on it. */
function rowOf(layer: CsrLayer, node: number): number {
    if (!layer.nodes) return node < layer.offsets.length - 1 ? node : -1;
    let lo = 0;
    let hi = layer.nodes.length - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >>> 1;
        const v = layer.nodes[mid]!;
        if (v === node) return mid;
        if (v < node) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/** Neighbours of `node` on `layer` (empty when the node is not on it). */
export function neighboursOf(layer: CsrLayer, node: number): Uint32Array {
    const r = rowOf(layer, node);
    if (r < 0) return layer.neighbours.subarray(0, 0);
    return layer.neighbours.subarray(layer.offsets[r]!, layer.offsets[r + 1]!);
}

/**
 * Builds the pruned CSR graph over `codes`, whose order is the id order of `ids`
 * (already sorted and unique). Distances are cosine over the int8 codes, which is
 * exact integer arithmetic plus correctly-rounded sqrt/division — so the same codes
 * give the same graph on every platform.
 */
export function buildGraph(
    ids: readonly string[],
    codes: readonly Int8Array[],
    params: IndexParams,
): { graph: CsrGraph; hubs: Uint8Array } {
    if (ids.length === 0) return { graph: EMPTY_GRAPH, hubs: new Uint8Array(0) };
    const hnsw = HnswIndex.canonicalBuild(
        ids.map((id, i) => ({ id, vector: codes[i]! })),
        { M: params.m, efConstruction: params.efConstruction, efSearch: params.efSearch },
    );
    const view = hnsw.graph();
    const { rows, hubs } = pruneBaseLayer(
        view.neighbours.map((layers) => layers[0] ?? []),
        params.m,
        params.hubShare,
        (a, b) => cosineInt8(codes[a]!, codes[b]!),
    );
    const layers: CsrLayer[] = [toCsrLayer(rows)];
    for (let l = 1; l <= view.maxLayer; l++) {
        const present: number[] = [];
        view.neighbours.forEach((layersOfNode, node) => {
            if (layersOfNode.length > l) present.push(node);
        });
        layers.push(toCsrLayer(present.map((node) => view.neighbours[node]![l]!), Uint32Array.from(present)));
    }
    return { graph: { entry: view.entry, layers }, hubs };
}
