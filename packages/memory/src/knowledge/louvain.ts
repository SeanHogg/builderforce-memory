/**
 * knowledge/louvain.ts — the community-detection algorithms behind communities.ts: Louvain modularity
 * maximisation and deterministic label propagation, over an undirected weighted adjacency. Pure; internal
 * to the knowledge module (the public surface is communities.ts).
 *
 * Determinism: nodes are visited in index order; at most COMMUNITY_MAX_PASSES aggregation passes of at most
 * COMMUNITY_MAX_ITERATIONS sweeps; a node stays put on a tie, otherwise the lowest community id wins;
 * no randomness.
 */

import type { NodeRef } from './model.js';
import { compareStr, round6 } from './refs.js';

export const COMMUNITY_MAX_PASSES = 10;
export const COMMUNITY_MAX_ITERATIONS = 20;
export const COMMUNITY_LP_ITERATIONS = 20;
const EPS = 1e-12;

export type Adj = Array<Map<number, number>>;

/** Undirected weighted graph over the sorted refs; self-references and non-positive weights are dropped. */
export function buildAdjacency(order: readonly NodeRef[], edges: ReadonlyArray<{ from: NodeRef; to: NodeRef; weight: number }>): Adj {
    const index = new Map(order.map((r, i) => [r, i] as const));
    const adj: Adj = order.map(() => new Map());
    // Sorted, so the floating-point sums are the same whatever order the edges arrive in.
    const sorted = [...edges].sort((x, y) => compareStr(x.from, y.from) || compareStr(x.to, y.to) || x.weight - y.weight);
    for (const e of sorted) {
        const a = index.get(e.from), b = index.get(e.to);
        if (a === undefined || b === undefined || a === b || !(e.weight > 0)) continue;
        adj[a]!.set(b, (adj[a]!.get(b) ?? 0) + e.weight);
        adj[b]!.set(a, (adj[b]!.get(a) ?? 0) + e.weight);
    }
    return adj;
}

export const degreeOf = (m: Map<number, number>): number => {
    let s = 0;
    for (const w of m.values()) s += w;
    return s;
};

/** Community ids renumbered 0.. in order of first appearance, so the next level is order-independent. */
function relabel(comm: readonly number[]): { labels: number[]; count: number } {
    const next = new Map<number, number>();
    const labels = comm.map((c) => {
        if (!next.has(c)) next.set(c, next.size);
        return next.get(c)!;
    });
    return { labels, count: next.size };
}

/**
 * One Louvain level: each node moves to the neighbouring community with the best modularity gain. It moves only
 * for a strictly better gain (a tie keeps it where it is; among tied better communities the lowest id wins).
 * `self` is each node's internal weight (aggregated levels), `twoM` the total degree.
 */
export function localMoving(adj: Adj, self: readonly number[], twoM: number): { comm: number[]; count: number; moved: boolean } {
    const n = adj.length;
    const comm = Array.from({ length: n }, (_, i) => i);
    const k = adj.map((m, i) => degreeOf(m) + 2 * self[i]!);
    const tot = [...k];
    let moved = false;
    for (let it = 0; it < COMMUNITY_MAX_ITERATIONS; it++) {
        let changed = false;
        for (let i = 0; i < n; i++) {
            const own = comm[i]!;
            const links = new Map<number, number>();
            for (const [j, w] of adj[i]!) links.set(comm[j]!, (links.get(comm[j]!) ?? 0) + w);
            tot[own]! -= k[i]!;
            const gain = (c: number) => (links.get(c) ?? 0) - (tot[c]! * k[i]!) / twoM;
            let best = own, bestGain = gain(own);
            for (const c of [...links.keys()].sort((a, b) => a - b)) {
                if (c === own) continue;
                const g = gain(c);
                if (g > bestGain + EPS) { best = c; bestGain = g; }
            }
            tot[best]! += k[i]!;
            if (best !== own) { comm[i] = best; changed = true; moved = true; }
        }
        if (!changed) break;
    }
    const { labels, count } = relabel(comm);
    return { comm: labels, count, moved };
}

/** Louvain: local moving, then aggregation, up to COMMUNITY_MAX_PASSES; returns a community label per node. */
export function louvain(adj0: Adj): number[] {
    let adj = adj0;
    let self: number[] = adj0.map(() => 0);
    const twoM = adj0.reduce((s, m) => s + degreeOf(m), 0);
    let membership = adj0.map((_, i) => i);
    for (let pass = 0; pass < COMMUNITY_MAX_PASSES; pass++) {
        const { comm, count, moved } = localMoving(adj, self, twoM);
        if (!moved) break;
        membership = membership.map((c) => comm[c]!);
        const nextAdj: Adj = Array.from({ length: count }, () => new Map());
        const nextSelf: number[] = new Array<number>(count).fill(0);
        for (let i = 0; i < adj.length; i++) {
            nextSelf[comm[i]!]! += self[i]!;
            for (const [j, w] of adj[i]!) {
                const a = comm[i]!, b = comm[j]!;
                if (a === b) nextSelf[a]! += w / 2; // each internal pair is seen from both ends
                else nextAdj[a]!.set(b, (nextAdj[a]!.get(b) ?? 0) + w);
            }
        }
        adj = nextAdj;
        self = nextSelf;
        if (count === 1) break;
    }
    return membership;
}

/** Deterministic label propagation: sorted order, each node takes the heaviest neighbour label; it keeps its own on a tie, else the smallest label wins. */
export function labelPropagation(adj: Adj): number[] {
    const label = adj.map((_, i) => i);
    for (let it = 0; it < COMMUNITY_LP_ITERATIONS; it++) {
        let changed = false;
        for (let i = 0; i < adj.length; i++) {
            const votes = new Map<number, number>();
            for (const [j, w] of adj[i]!) votes.set(label[j]!, (votes.get(label[j]!) ?? 0) + w);
            let best = label[i]!, bestW = votes.get(best) ?? 0;
            for (const l of [...votes.keys()].sort((a, b) => a - b)) {
                if (votes.get(l)! > bestW + EPS) { best = l; bestW = votes.get(l)!; }
            }
            if (best !== label[i]) { label[i] = best; changed = true; }
        }
        if (!changed) break;
    }
    return label;
}

/** Q = Σc [Lc/m − (Dc/2m)²] for a partition of the undirected graph, 6 decimals. */
export function modularityOf(adj: Adj, comm: readonly number[]): number {
    let twoM = 0;
    const inside = new Map<number, number>(), degree = new Map<number, number>();
    for (let i = 0; i < adj.length; i++) {
        for (const [j, w] of adj[i]!) {
            twoM += w;
            degree.set(comm[i]!, (degree.get(comm[i]!) ?? 0) + w);
            if (comm[i] === comm[j]) inside.set(comm[i]!, (inside.get(comm[i]!) ?? 0) + w);
        }
    }
    if (twoM === 0) return 0;
    let q = 0;
    for (const [c, d] of degree) q += (inside.get(c) ?? 0) / twoM - (d / twoM) ** 2;
    return round6(q);
}
