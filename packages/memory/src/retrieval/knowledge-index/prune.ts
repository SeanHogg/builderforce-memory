/**
 * prune.ts — high-degree-preserving pruning of the base layer (the LEANN storage
 * technique).
 *
 * Most of a graph index's bytes are base-layer edges. Searches route through the
 * few well-connected hubs, so those keep their full degree and every other node
 * keeps half: the top `hubShare` of nodes by in-degree keep `m` edges, the rest
 * `m / 2`. Upper layers are tiny and stay unpruned.
 *
 * Every choice is a total order (in-degree, then node index; similarity, then
 * node index — and node index is id order), so pruning is deterministic.
 */

import { compareScoreThenId, DENSE_SCORE_QUANTUM } from '../order.js';

/**
 * Prunes base-layer rows. `similarity(a, b)` scores two nodes (higher = nearer).
 * Each kept row is ordered nearest first.
 */
export function pruneBaseLayer(
    rows: ReadonlyArray<readonly number[]>,
    m: number,
    hubShare: number,
    similarity: (a: number, b: number) => number,
): { rows: number[][]; hubs: Uint8Array } {
    const n = rows.length;
    const inDegree = new Uint32Array(n);
    for (const row of rows) for (const to of row) inDegree[to] += 1;

    const hubs = new Uint8Array(n);
    const hubCount = n === 0 ? 0 : Math.min(n, Math.max(1, Math.ceil(n * Math.max(0, hubShare))));
    const byDegree = Array.from({ length: n }, (_, i) => i)
        .sort((a, b) => inDegree[b]! - inDegree[a]! || a - b);
    for (let i = 0; i < hubCount; i++) hubs[byDegree[i]!] = 1;

    const full = Math.max(1, m);
    const half = Math.max(1, Math.floor(m / 2));
    const pruned = rows.map((row, node) => {
        const keep = hubs[node] ? full : half;
        return [...new Set(row)]
            .filter((to) => to !== node)
            .map((to) => ({ to, score: similarity(node, to) }))
            .sort((a, b) => compareScoreThenId(a.score, a.to, b.score, b.to, DENSE_SCORE_QUANTUM))
            .slice(0, keep)
            .map((e) => e.to);
    });
    return { rows: pruned, hubs };
}
