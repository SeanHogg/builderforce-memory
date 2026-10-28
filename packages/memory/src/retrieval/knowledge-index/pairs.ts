/**
 * pairs.ts — near-duplicate pairs over the int8 codes (the Duplication lens,
 * review-time entity resolution, contradiction candidates).
 *
 * Graph mode: near-duplicates are each other's nearest neighbours, so the candidate
 * pairs are the compacted graph's edges on every layer rather than all n² pairs.
 * Exact mode (no graph): {@link exactPairCandidates} — all pairs while they fit
 * `EXACT_PAIR_BUDGET`, deterministic sign-bit blocking above it, never more
 * than the budget. Similarity is {@link cosineInt8} (exact integers,
 * correctly-rounded sqrt/division). The order is total: similarity (quantized
 * 1e-4) descending, then `a`, then `b`, and `a < b` by id in every pair.
 *
 * Cards not yet compacted, and lean-mode nodes whose codes were dropped, are not
 * paired until the next compaction (the call is synchronous; it never embeds).
 */

import { compareIds, quantizeScore, DENSE_SCORE_QUANTUM } from '../order.js';
import type { SimilarPair } from './types.js';
import { graphNode, type Compacted, type IndexState } from './state.js';
import { neighboursOf } from './csr.js';
import { cosineInt8 } from './quantize.js';
import { exactPairCandidates } from './scan.js';

export interface SimilarPairsOptions {
    minSimilarity: number;
    maxPairs: number;
    idPrefix?: string;
}

/** Graph mode: every edge on every layer between two usable nodes, once, as `[lo, hi]`. */
function graphPairCandidates(c: Compacted, usable: (node: number) => boolean): Array<[number, number]> {
    const seen = new Set<string>();
    const out: Array<[number, number]> = [];
    for (let a = 0; a < c.ids.length; a++) {
        if (!usable(a)) continue;
        for (const layer of c.graph.layers) {
            for (const b of neighboursOf(layer, a)) {
                if (b === a || !usable(b)) continue;
                const lo = Math.min(a, b);
                const hi = Math.max(a, b);
                const key = `${lo}:${hi}`;
                if (seen.has(key)) continue;
                seen.add(key);
                out.push([lo, hi]);
            }
        }
    }
    return out;
}

export function similarPairs(state: IndexState, opts: SimilarPairsOptions): SimilarPair[] {
    const c = state.compacted;
    if (!c || opts.maxPairs <= 0) return [];
    const prefix = opts.idPrefix ?? '';
    const usable = (node: number) =>
        !!c.codes[node] && c.ids[node]!.startsWith(prefix) && graphNode(state, c.ids[node]!) === node;

    const candidates = c.mode === 'exact'
        ? exactPairCandidates(c.codes, c.ids.map((_, node) => node).filter(usable), state.dims)
        : graphPairCandidates(c, usable);
    const pairs: Array<SimilarPair & { q: number }> = [];
    for (const [lo, hi] of candidates) {
        const similarity = cosineInt8(c.codes[lo]!, c.codes[hi]!);
        if (similarity >= opts.minSimilarity) {
            pairs.push({ a: c.ids[lo]!, b: c.ids[hi]!, similarity, q: quantizeScore(similarity, DENSE_SCORE_QUANTUM) });
        }
    }
    return pairs
        .sort((x, y) => (y.q - x.q) || compareIds(x.a, y.a) || compareIds(x.b, y.b))
        .slice(0, opts.maxPairs)
        .map(({ a, b, similarity }) => ({ a, b, similarity }));
}
