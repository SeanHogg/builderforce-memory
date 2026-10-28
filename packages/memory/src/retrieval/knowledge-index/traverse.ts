/**
 * traverse.ts — HNSW search over the CSR graph with integer scores.
 *
 * Greedy descent through the upper layers, then a bounded best-first beam on the
 * base layer. Scores are int8 dot products (exact integers), and equal scores
 * break on node index (= id order), so the candidates are bit-identical on every
 * platform for the same codes and query code.
 *
 * Lean mode stores no code for most nodes; `ensure` is called with the nodes about
 * to be scored so the caller can recompute their codes first. It is awaited only
 * when it returns a promise, so the non-lean walk never yields.
 */

import { neighboursOf, type CsrGraph } from './csr.js';

export interface GraphScorer {
    /** The node's score against the query; higher is nearer. */
    score(node: number): number;
    /** Make codes available for `nodes` before they are scored (lean mode). */
    ensure?(nodes: number[]): Promise<void> | undefined;
}

export interface GraphCandidate {
    node: number;
    score: number;
}

/** Higher score first, then lower node index. */
function before(a: GraphCandidate, b: GraphCandidate): boolean {
    return a.score > b.score || (a.score === b.score && a.node < b.node);
}

/** Inserts `c` into `list`, kept in {@link before} order. */
function insertOrdered(list: GraphCandidate[], c: GraphCandidate): void {
    let lo = 0;
    let hi = list.length;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (before(list[mid]!, c)) lo = mid + 1;
        else hi = mid;
    }
    list.splice(lo, 0, c);
}

/** The scorer's `ensure`, or `undefined` when there is nothing to wait for. */
function ensure(scorer: GraphScorer, nodes: number[]): Promise<void> | undefined {
    if (!scorer.ensure || nodes.length === 0) return undefined;
    return scorer.ensure(nodes);
}

/** The best `ef` nodes for the query, in {@link before} order. */
export async function searchGraph(graph: CsrGraph, scorer: GraphScorer, ef: number): Promise<GraphCandidate[]> {
    if (graph.entry < 0 || graph.layers.length === 0 || ef <= 0) return [];
    { const p = ensure(scorer, [graph.entry]); if (p) await p; }
    let ep: GraphCandidate = { node: graph.entry, score: scorer.score(graph.entry) };

    for (let l = graph.layers.length - 1; l > 0; l--) {
        let improved = true;
        while (improved) {
            improved = false;
            const nbs = Array.from(neighboursOf(graph.layers[l]!, ep.node));
            const p = ensure(scorer, nbs);
            if (p) await p;
            for (const node of nbs) {
                const c = { node, score: scorer.score(node) };
                if (before(c, ep)) {
                    ep = c;
                    improved = true;
                }
            }
        }
    }

    const base = graph.layers[0]!;
    const visited = new Set<number>([ep.node]);
    const frontier: GraphCandidate[] = [ep];
    const results: GraphCandidate[] = [ep];
    while (frontier.length > 0) {
        const cur = frontier.shift()!;
        if (results.length >= ef && before(results[results.length - 1]!, cur)) break;
        const fresh = Array.from(neighboursOf(base, cur.node)).filter((node) => !visited.has(node));
        for (const node of fresh) visited.add(node);
        const p = ensure(scorer, fresh);
        if (p) await p;
        for (const node of fresh) {
            const c = { node, score: scorer.score(node) };
            if (results.length < ef || before(c, results[results.length - 1]!)) {
                insertOrdered(frontier, c);
                insertOrdered(results, c);
                if (results.length > ef) results.pop();
            }
        }
    }
    return results;
}
