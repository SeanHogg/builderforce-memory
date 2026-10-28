/**
 * digest.ts — the name of an index state.
 *
 * sha-256 over the canonical encoding of everything a result depends on: the
 * fingerprint, dims, params and int8 scale; every live card's (id, hash, keys) in
 * id order; and the compacted snapshot (its mode — exact scan or graph — node ids
 * and hashes, hubs, the CSR graph, empty in exact mode).
 * Every search result is a pure function of `(digest, query, opts)`, which is what
 * lets a logged digest replay a disputed answer.
 */

import { compareIds } from '../order.js';
import type { IndexParams } from './types.js';
import type { IndexState } from './state.js';
import type { CsrGraph } from './csr.js';
import { ByteWriter } from './bytes.js';
import { sha256Hex } from './sha256.js';

/** The default content hash of a card's text. */
export function hashDocText(text: string): string {
    return sha256Hex(text);
}

/** Params in a fixed field order, for the digest and the serialized header. */
export function writeParams(w: ByteWriter, p: IndexParams): void {
    w.u32(p.m).u32(p.efConstruction).u32(p.efSearch).f64(p.hubShare).u32(p.rerankFactor).u8(p.lean ? 1 : 0).u32(p.rrfK).u32(p.exactBelow);
}

/** The graph's layers: entry, then per layer its node list, offsets and neighbours. */
export function writeGraph(w: ByteWriter, g: CsrGraph): void {
    w.i32(g.entry).u32(g.layers.length);
    for (const layer of g.layers) w.u32s(layer.nodes ?? []).u32s(layer.offsets).u32s(layer.neighbours);
}

export function stateDigest(state: IndexState): string {
    const w = new ByteWriter();
    w.str('bf-knowledge-index/digest/2').str(state.fingerprint).u32(state.dims).f64(state.scale);
    writeParams(w, state.params);
    const ids = [...state.docs.keys()].sort(compareIds);
    w.u32(ids.length);
    for (const id of ids) {
        const doc = state.docs.get(id)!;
        w.str(id).str(doc.hash).u32(doc.keys.length);
        for (const key of doc.keys) w.str(key);
    }
    const c = state.compacted;
    w.u8(c ? 1 : 0);
    if (c) {
        w.u8(c.mode === 'exact' ? 1 : 0);
        w.u32(c.ids.length);
        for (let i = 0; i < c.ids.length; i++) w.str(c.ids[i]!).str(c.hashes[i]!);
        w.bytes(c.hubs);
        writeGraph(w, c.graph);
    }
    return sha256Hex(w.finish());
}
