/**
 * compact.ts — the canonical rebuild.
 *
 * Every live card, in id order, is coded (an unchanged node keeps its code; a new
 * or changed card is quantized from its float vector) and the graph is rebuilt
 * from those codes. Since levels come from id hashes, inserts are id-ordered and
 * every tie breaks on id, the same card set and fingerprint give the same bytes
 * whatever order the upserts arrived in.
 *
 * The int8 scale is fitted at the first compaction and fixed after it. An embedder
 * whose fingerprint differs from the stored one is never mixed in: compaction
 * drops every stored vector and code and re-embeds all cards under it.
 */

import { compareIds } from '../order.js';
import type { EmbedderPort } from './types.js';
import { graphNode, type DocEntry, type IndexState } from './state.js';
import type { EmbeddingCache } from './embeddings.js';
import { fitScale, quantize } from './quantize.js';
import { buildGraph } from './csr.js';

/** Float vectors for the docs that have neither a reusable code nor a pending vector. */
async function vectorsFor(cache: EmbeddingCache, docs: readonly DocEntry[]): Promise<Map<string, Float32Array>> {
    const vecs = await cache.vectors(docs);
    return new Map(docs.map((d, i) => [d.id, vecs[i]!]));
}

export async function compactState(state: IndexState, embedder: EmbedderPort | undefined, cache: EmbeddingCache | undefined): Promise<void> {
    // Nothing to code with: exact and lexical need no compaction, and a stored
    // graph of another fingerprint is kept for an embedder that matches it.
    if (!embedder || !cache) return;
    if (embedder.fingerprint !== state.fingerprint || embedder.dims !== state.dims) {
        state.fingerprint = embedder.fingerprint;
        state.dims = embedder.dims;
        state.scale = 0;
        state.compacted = null;
        for (const doc of state.docs.values()) doc.vec = undefined;
    }

    const ids = [...state.docs.keys()].sort(compareIds);
    const docs = ids.map((id) => state.docs.get(id)!);
    if (docs.length === 0) {
        state.compacted = null;
        return;
    }

    const old = state.compacted;
    const reused = docs.map((d) => {
        const node = graphNode(state, d.id);
        return node >= 0 && state.scale > 0 ? old!.codes[node] : undefined;
    });
    const missing = await vectorsFor(cache, docs.filter((d, i) => !reused[i] && !d.vec));
    const floatOf = (d: DocEntry) => d.vec ?? missing.get(d.id)!;

    if (state.scale === 0) state.scale = fitScale(docs.map(floatOf));
    const codes = docs.map((d, i) => reused[i] ?? quantize(floatOf(d), state.scale));

    const { graph, hubs } = buildGraph(ids, codes, state.params);
    state.compacted = {
        ids,
        hashes: docs.map((d) => d.hash),
        codes: state.params.lean ? codes.map((code, i) => (hubs[i] ? code : undefined)) : codes,
        hubs,
        graph,
        nodeOf: new Map(ids.map((id, i) => [id, i])),
    };
    for (const doc of docs) doc.vec = undefined;
}
