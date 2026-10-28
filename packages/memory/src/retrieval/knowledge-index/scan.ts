/**
 * scan.ts — exact mode: the dense tier and `similarPairs` with no graph.
 *
 * Below `exactBelow` cards an HNSW graph costs more to build than it saves, so
 * `compact()` builds none and these replace the walk:
 *
 *   - {@link scanCodes}: an exact integer dot product against every live int8 code,
 *     the best `n` kept with ties on node index (= id order). Same order as the
 *     graph walk's candidates, so the float rerank after it is unchanged.
 *   - {@link exactPairCandidates}: the node pairs `similarPairs` scores. All pairs
 *     while there are at most {@link EXACT_PAIR_BUDGET}; above that, deterministic
 *     sign-bit blocking (see the function), capped at the same budget.
 */

import { graphNode, type Compacted, type IndexState } from './state.js';
import { dotInt8 } from './quantize.js';
import { before, insertOrdered, type GraphCandidate } from './traverse.js';

/** The most node pairs `similarPairs` scores in exact mode. */
export const EXACT_PAIR_BUDGET = 250_000;
/** Blocking tables: a pair is a candidate when it shares a bucket in any one of them. */
export const PAIR_BLOCK_TABLES = 8;
/** Sign bits per table (2^bits buckets each). */
export const PAIR_BLOCK_BITS = 10;
/** Within a bucket, each node is paired with at most this many following members (id order). */
export const PAIR_BLOCK_WINDOW = 32;

/** The best `n` live coded nodes by integer dot product with `qc`, in {@link before} order. */
export function scanCodes(state: IndexState, c: Compacted, qc: Int8Array, n: number): GraphCandidate[] {
    const top: GraphCandidate[] = [];
    if (n <= 0) return top;
    for (let node = 0; node < c.ids.length; node++) {
        const code = c.codes[node];
        if (!code || graphNode(state, c.ids[node]!) !== node) continue;
        const cand = { node, score: dotInt8(qc, code) };
        if (top.length < n || before(cand, top[top.length - 1]!)) {
            insertOrdered(top, cand);
            if (top.length > n) top.pop();
        }
    }
    return top;
}

/** The dimensions table `t` reads its sign bits from, spread evenly across `dims`. */
function blockDims(t: number, dims: number): number[] {
    const stride = Math.max(1, Math.floor(dims / (PAIR_BLOCK_TABLES * PAIR_BLOCK_BITS)));
    return Array.from({ length: PAIR_BLOCK_BITS }, (_, j) => ((t * PAIR_BLOCK_BITS + j) * stride) % dims);
}

/**
 * Candidate `[lo, hi]` node pairs (lo < hi) among `nodes` (ascending, all coded).
 *
 * When every pair fits in {@link EXACT_PAIR_BUDGET} (about 700 nodes), every pair is
 * a candidate. Above that, each of {@link PAIR_BLOCK_TABLES} tables buckets the nodes
 * by the signs of {@link PAIR_BLOCK_BITS} fixed code dimensions; a near-duplicate
 * agrees on almost every sign, so it lands with its twin in at least one table.
 * Inside a bucket a node is paired with its next {@link PAIR_BLOCK_WINDOW} members.
 * Tables run in order, buckets in key order, members in node order, and generation
 * stops at the budget — so the candidates are a pure function of the codes, and the
 * cost is at most `EXACT_PAIR_BUDGET` cosine evaluations whatever the size.
 */
export function exactPairCandidates(
    codes: ReadonlyArray<Int8Array | undefined>,
    nodes: readonly number[],
    dims: number,
): Array<[number, number]> {
    const out: Array<[number, number]> = [];
    const n = nodes.length;
    if ((n * (n - 1)) / 2 <= EXACT_PAIR_BUDGET) {
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) out.push([nodes[i]!, nodes[j]!]);
        return out;
    }
    if (dims <= 0) return out;
    const seen = new Set<number>();
    for (let t = 0; t < PAIR_BLOCK_TABLES; t++) {
        const bits = blockDims(t, dims);
        const buckets = new Map<number, number[]>();
        for (const node of nodes) {
            const code = codes[node]!;
            let key = 0;
            for (const d of bits) key = key * 2 + ((code[d] ?? 0) > 0 ? 1 : 0);
            const members = buckets.get(key);
            if (members) members.push(node);
            else buckets.set(key, [node]);
        }
        for (const key of [...buckets.keys()].sort((a, b) => a - b)) {
            const members = buckets.get(key)!;
            for (let i = 0; i < members.length; i++) {
                const end = Math.min(members.length, i + 1 + PAIR_BLOCK_WINDOW);
                for (let j = i + 1; j < end; j++) {
                    const pairKey = members[i]! * codes.length + members[j]!;
                    if (seen.has(pairKey)) continue;
                    seen.add(pairKey);
                    out.push([members[i]!, members[j]!]);
                    if (out.length >= EXACT_PAIR_BUDGET) return out;
                }
            }
        }
    }
    return out;
}
