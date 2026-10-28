/**
 * knowledge/importance.ts — deterministic PageRank (node size in the view, the
 * truncation order, the cards' "top neighbours").
 *
 * Fixed iterations, damping 0.85, weighted out-edges, dangling mass spread uniformly,
 * nodes iterated in ref order so the floating-point sums are the same everywhere.
 * Scores are normalized so the top node is 1, and rounded to 6 decimals.
 */

import type { KNode, NodeRef } from './model.js';
import { compareStr, round6 } from './refs.js';

export const PAGERANK_DAMPING = 0.85;
export const PAGERANK_ITERATIONS = 40;

export function pageRank(
    refs: readonly NodeRef[],
    edges: ReadonlyArray<{ from: NodeRef; to: NodeRef; weight: number }>,
    iterations: number = PAGERANK_ITERATIONS,
    damping: number = PAGERANK_DAMPING,
): Map<NodeRef, number> {
    const order = [...new Set(refs)].sort(compareStr);
    const n = order.length;
    const out = new Map<NodeRef, number>();
    if (n === 0) return out;
    const index = new Map(order.map((r, i) => [r, i] as const));
    const outWeight = new Float64Array(n);
    const inbound: Array<Array<{ from: number; w: number }>> = order.map(() => []);
    const sorted = [...edges].sort((a, b) => compareStr(a.from, b.from) || compareStr(a.to, b.to));
    for (const e of sorted) {
        const f = index.get(e.from), t = index.get(e.to);
        const w = e.weight > 0 ? e.weight : 0;
        if (f === undefined || t === undefined || f === t || w === 0) continue;
        outWeight[f] += w;
        inbound[t]!.push({ from: f, w });
    }
    let rank = new Float64Array(n).fill(1 / n);
    for (let it = 0; it < iterations; it++) {
        let dangling = 0;
        for (let i = 0; i < n; i++) if (outWeight[i] === 0) dangling += rank[i]!;
        const next = new Float64Array(n);
        const base = (1 - damping) / n + (damping * dangling) / n;
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (const { from, w } of inbound[i]!) sum += (rank[from]! * w) / outWeight[from]!;
            next[i] = base + damping * sum;
        }
        rank = next;
    }
    let max = 0;
    for (let i = 0; i < n; i++) max = Math.max(max, rank[i]!);
    for (let i = 0; i < n; i++) out.set(order[i]!, max > 0 ? round6(rank[i]! / max) : 0);
    return out;
}

/** Importance descending, then ref — the one order "top N" means everywhere. */
export function byImportance(a: Pick<KNode, 'importance' | 'ref'>, b: Pick<KNode, 'importance' | 'ref'>): number {
    return b.importance - a.importance || compareStr(a.ref, b.ref);
}

/** Area importance: the sum of its members' importance, renormalized so the top area is 1. */
export function areaImportance(members: ReadonlyMap<NodeRef, readonly number[]>): Map<NodeRef, number> {
    const sums = new Map<NodeRef, number>();
    let max = 0;
    for (const area of [...members.keys()].sort(compareStr)) {
        const sum = members.get(area)!.reduce((s, v) => s + v, 0);
        sums.set(area, sum);
        max = Math.max(max, sum);
    }
    for (const [area, sum] of sums) sums.set(area, max > 0 ? round6(sum / max) : 0);
    return sums;
}
