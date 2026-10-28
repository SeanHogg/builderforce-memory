/**
 * cascade.ts — the knowledge index's search: cheapest tier first.
 *
 *   1. exact    — the folded `keys` map; a hit scores 1.0, and when the exact hits
 *                 fill `k` nothing else runs;
 *   2. lexical  — BM25 over card text with the code-aware tokenizer;
 *   3. dense    — int8 graph search for `k × rerankFactor` candidates (plus the
 *                 cards not yet compacted), then a float rerank over vectors
 *                 recomputed through the hash-keyed LRU;
 *   4. fuse     — RRF over the lexical and dense lists; exact hits stay on top at
 *                 1.0; everything in {@link compareHits} order.
 *
 * Dense runs only with an embedder of the stored fingerprint — otherwise the tier
 * is absent, never silently replaced.
 */

import { reciprocalRankFusion } from '../fusion.js';
import { compareDenseHits, compareHits } from '../order.js';
import { cosineSimilarity } from '../../similarity/index.js';
import type { EmbedderPort, IndexHit, IndexTier } from './types.js';
import { availableTiers, graphNode, type DocEntry, type IndexState } from './state.js';
import type { EmbeddingCache } from './embeddings.js';
import { dotInt8, quantize } from './quantize.js';
import { searchGraph } from './traverse.js';
import { foldKey } from './tokenize.js';

export interface SearchContext {
    state: IndexState;
    embedder?: EmbedderPort;
    /** The embedder's recompute cache; present whenever `embedder` is. */
    cache?: EmbeddingCache;
}

export interface SearchOptions {
    k?: number;
    tiers?: IndexTier[];
}

/** Exact-tier ids for `q`, ascending. */
function exactIds(state: IndexState, q: string): string[] {
    const ids = state.keys.get(foldKey(q));
    return ids ? [...ids].sort() : [];
}

/**
 * Dense candidates before the float rerank: the best `n` live graph nodes by
 * integer dot product, plus the best `n` cards not yet in the graph. Ids, unordered
 * between the two sources (the rerank orders them).
 */
export async function denseCandidates(ctx: SearchContext, qv: Float32Array, n: number): Promise<string[]> {
    const { state, cache } = ctx;
    const out = new Set<string>();
    const c = state.compacted;
    if (c && state.scale > 0) {
        const qc = quantize(qv, state.scale);
        const recomputed = new Map<number, Int8Array>();
        const codeOf = (node: number) => c.codes[node] ?? recomputed.get(node);
        const found = await searchGraph(c.graph, {
            score: (node) => {
                const code = codeOf(node);
                return code ? dotInt8(qc, code) : -Infinity;
            },
            ensure: cache && state.params.lean ? (nodes) => {
                const live = nodes.filter((node) => !codeOf(node) && graphNode(state, c.ids[node]!) === node);
                if (live.length === 0) return undefined;
                return cache.vectors(live.map((node) => state.docs.get(c.ids[node]!)!)).then((vecs) => {
                    live.forEach((node, i) => recomputed.set(node, quantize(vecs[i]!, state.scale)));
                });
            } : undefined,
        }, Math.max(state.params.efSearch, n));
        for (const hit of found) {
            if (out.size >= n) break;
            if (hit.score !== -Infinity && graphNode(state, c.ids[hit.node]!) === hit.node) out.add(c.ids[hit.node]!);
        }
    }
    const pending: Array<{ id: string; score: number }> = [];
    for (const doc of state.docs.values()) {
        if (doc.vec && graphNode(state, doc.id) < 0) pending.push({ id: doc.id, score: cosineSimilarity(qv, doc.vec) });
    }
    for (const hit of pending.sort(compareDenseHits).slice(0, n)) out.add(hit.id);
    return [...out];
}

/** The dense tier: candidates, then float recompute + rerank, best first. */
async function denseRank(ctx: SearchContext, q: string, n: number): Promise<Array<{ id: string; score: number }>> {
    const cache = ctx.cache!;
    const qv = await cache.query(q);
    const ids = await denseCandidates(ctx, qv, n);
    const docs = ids.map((id) => ctx.state.docs.get(id)!);
    const needVec = docs.filter((d) => !d.vec);
    const fresh = await cache.vectors(needVec);
    const vecOf = new Map<DocEntry, Float32Array>(needVec.map((d, i) => [d, fresh[i]!]));
    return docs
        .map((d) => ({ id: d.id, score: cosineSimilarity(qv, d.vec ?? vecOf.get(d)!) }))
        .sort(compareDenseHits)
        .slice(0, n);
}

/** Runs the cascade. A pure function of the state (its digest), `q` and `opts`. */
export async function cascadeSearch(ctx: SearchContext, q: string, opts: SearchOptions = {}): Promise<IndexHit[]> {
    const { state } = ctx;
    const k = Math.max(0, opts.k ?? 10);
    if (k === 0 || state.docs.size === 0) return [];
    const available = availableTiers(state, ctx.embedder);
    const run = new Set((opts.tiers ?? available).filter((t) => available.includes(t)));

    const exact: IndexHit[] = run.has('exact')
        ? exactIds(state, q).map((id) => ({ id, score: 1, tier: 'exact' as const }))
        : [];
    if (exact.length >= k) return exact.slice(0, k);
    const exactSet = new Set(exact.map((h) => h.id));

    const n = Math.max(k, k * state.params.rerankFactor);
    const lexical = run.has('lexical')
        ? state.lexical.search(q, n + exact.length).filter((h) => !exactSet.has(h.id)).map((h) => h.id)
        : [];
    const dense = run.has('dense')
        ? (await denseRank(ctx, q, n + exact.length)).filter((h) => !exactSet.has(h.id)).map((h) => h.id)
        : [];

    const lexicalSet = new Set(lexical);
    const fused = reciprocalRankFusion([{ ids: lexical }, { ids: dense }].filter((l) => l.ids.length > 0), state.params.rrfK)
        .map((h): IndexHit => ({ id: h.id, score: h.score, tier: lexicalSet.has(h.id) ? 'lexical' : 'dense' }));
    return [...exact, ...fused].sort(compareHits).slice(0, k);
}
