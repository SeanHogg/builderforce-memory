/**
 * serialize.ts — the versioned binary form of an index state.
 *
 * Layout (little-endian, via `bytes.ts`), version 2:
 *   magic "BFKI" · u32 version
 *   fingerprint · dims · scale · params (`writeParams`, ending in `exactBelow`)
 *   docs (id order): id · text · hash · keys · optional pending float vector
 *   u8 compacted? → u8 mode (1 exact, 0 graph) · node ids · node hashes · hubs ·
 *                   code-present flags · codes · graph (empty in exact mode)
 *
 * Version 1 (2026.10.29) had no `exactBelow` and no mode byte; every v1 index was a
 * graph. It still loads: as graph mode, with `exactBelow` 0 so a later `compact()`
 * keeps building the graph it was made with. Writing is always the current version.
 *
 * Texts travel with the index because the lexical tier, the recompute path and a
 * re-embed under a new fingerprint all need them. Loading rebuilds the derived
 * structures (key map, BM25 postings) and gives back the same digest.
 */

import { compareIds } from '../order.js';
import type { IndexParams } from './types.js';
import { createState, indexKeys, type Compacted, type DocEntry, type IndexState } from './state.js';
import type { CsrLayer } from './csr.js';
import { ByteReader, ByteWriter } from './bytes.js';
import { writeGraph, writeParams } from './digest.js';

const MAGIC = [0x42, 0x46, 0x4b, 0x49]; // "BFKI"
export const SERIALIZE_VERSION = 2;
/** The oldest version `deserializeState` still reads. */
const MIN_READ_VERSION = 1;

export function serializeState(state: IndexState): Uint8Array {
    const w = new ByteWriter();
    w.raw(Uint8Array.from(MAGIC)).u32(SERIALIZE_VERSION);
    w.str(state.fingerprint).u32(state.dims).f64(state.scale);
    writeParams(w, state.params);

    const ids = [...state.docs.keys()].sort(compareIds);
    w.u32(ids.length);
    for (const id of ids) {
        const doc = state.docs.get(id)!;
        w.str(doc.id).str(doc.text).str(doc.hash).u32(doc.keys.length);
        for (const key of doc.keys) w.str(key);
        w.u8(doc.vec ? 1 : 0);
        if (doc.vec) w.f32s(doc.vec);
    }

    const c = state.compacted;
    w.u8(c ? 1 : 0);
    if (c) {
        w.u8(c.mode === 'exact' ? 1 : 0);
        w.u32(c.ids.length);
        for (let i = 0; i < c.ids.length; i++) w.str(c.ids[i]!).str(c.hashes[i]!);
        w.bytes(c.hubs);
        w.bytes(Uint8Array.from(c.codes, (code) => (code ? 1 : 0)));
        for (const code of c.codes) if (code) w.bytes(code);
        writeGraph(w, c.graph);
    }
    return w.finish();
}

function readParams(r: ByteReader, version: number): IndexParams {
    return {
        m: r.u32(),
        efConstruction: r.u32(),
        efSearch: r.u32(),
        hubShare: r.f64(),
        rerankFactor: r.u32(),
        lean: r.u8() === 1,
        rrfK: r.u32(),
        // v1 indexes were always graphs: keep building one.
        exactBelow: version >= 2 ? r.u32() : 0,
    };
}

export function deserializeState(bytes: Uint8Array): IndexState {
    const r = new ByteReader(bytes);
    const magic = r.raw(4);
    if (MAGIC.some((b, i) => magic[i] !== b)) throw new Error('knowledge-index: not a serialized index');
    const version = r.u32();
    if (version < MIN_READ_VERSION || version > SERIALIZE_VERSION) throw new Error(`knowledge-index: unsupported version ${version}`);

    const fingerprint = r.str();
    const dims = r.u32();
    const scale = r.f64();
    const state = createState(readParams(r, version));
    Object.assign(state, { fingerprint, dims, scale });

    const docCount = r.u32();
    for (let i = 0; i < docCount; i++) {
        const doc: DocEntry = { id: r.str(), text: r.str(), hash: r.str(), keys: [] };
        const keyCount = r.u32();
        for (let k = 0; k < keyCount; k++) doc.keys.push(r.str());
        if (r.u8() === 1) doc.vec = r.f32s();
        state.docs.set(doc.id, doc);
        indexKeys(state, doc, true);
        state.lexical.set(doc.id, doc.text);
    }

    if (r.u8() === 1) {
        const mode: Compacted['mode'] = version >= 2 && r.u8() === 1 ? 'exact' : 'graph';
        const n = r.u32();
        const ids: string[] = [];
        const hashes: string[] = [];
        for (let i = 0; i < n; i++) {
            ids.push(r.str());
            hashes.push(r.str());
        }
        const hubs = r.bytes();
        const present = r.bytes();
        const codes: Compacted['codes'] = Array.from(present, (p) => {
            if (p !== 1) return undefined;
            const raw = r.bytes();
            return new Int8Array(raw.buffer, raw.byteOffset, raw.byteLength);
        });
        const entry = r.i32();
        const layerCount = r.u32();
        const layers: CsrLayer[] = [];
        for (let l = 0; l < layerCount; l++) {
            const nodes = r.u32s();
            const offsets = r.u32s();
            const neighbours = r.u32s();
            layers.push(l === 0 ? { offsets, neighbours } : { nodes, offsets, neighbours });
        }
        state.compacted = {
            mode,
            ids,
            hashes,
            codes,
            hubs,
            graph: { entry, layers },
            nodeOf: new Map(ids.map((id, i) => [id, i])),
        };
    }
    return state;
}
