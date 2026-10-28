/**
 * state.ts — the knowledge index's in-memory state, shared by the cascade,
 * compaction, digest and serializer. Everything a result depends on lives here,
 * so `digest()` can name it and `serialize()` can carry it.
 */

import type { EmbedderPort, IndexMode, IndexParams, IndexTier } from './types.js';
import type { CsrGraph } from './csr.js';
import { LexicalIndex } from './lexical.js';
import { foldKey } from './tokenize.js';

export const DEFAULT_INDEX_PARAMS: Readonly<IndexParams> = Object.freeze({
    m: 16,
    efConstruction: 200,
    efSearch: 64,
    hubShare: 0.02,
    rerankFactor: 4,
    lean: false,
    rrfK: 60,
    exactBelow: 20000,
});

/** One live card. `vec` is its unit float embedding while it is not yet in the graph. */
export interface DocEntry {
    id: string;
    text: string;
    /** Case-folded, de-duplicated, sorted. */
    keys: string[];
    hash: string;
    vec?: Float32Array;
}

/** The canonical, id-ordered snapshot written by `compact()`. Node index = rank of its id. */
export interface Compacted {
    /** `exact`: no graph (empty `graph`, zero `hubs`, every code kept); `graph`: HNSW over the codes. */
    mode: IndexMode;
    ids: string[];
    /** The doc hash each node was coded from; a node whose doc changed is stale. */
    hashes: string[];
    /** int8 codes by node; `undefined` where lean mode dropped a non-hub's code. */
    codes: Array<Int8Array | undefined>;
    /** 1 for a hub (top `hubShare` by in-degree). */
    hubs: Uint8Array;
    graph: CsrGraph;
    nodeOf: Map<string, number>;
}

export interface IndexState {
    params: IndexParams;
    /** Fingerprint of every stored vector and code; '' when none was ever embedded. */
    fingerprint: string;
    dims: number;
    /** The int8 scale, fixed at the first compaction; 0 until then. */
    scale: number;
    docs: Map<string, DocEntry>;
    /** Folded key → ids. */
    keys: Map<string, Set<string>>;
    lexical: LexicalIndex;
    compacted: Compacted | null;
}

export function createState(params: Partial<IndexParams> | undefined, embedder?: EmbedderPort): IndexState {
    return {
        params: { ...DEFAULT_INDEX_PARAMS, ...params },
        fingerprint: embedder?.fingerprint ?? '',
        dims: embedder?.dims ?? 0,
        scale: 0,
        docs: new Map(),
        keys: new Map(),
        lexical: new LexicalIndex(),
        compacted: null,
    };
}

/** Folded, unique, sorted keys. */
export function canonicalKeys(keys: readonly string[] | undefined): string[] {
    return [...new Set((keys ?? []).map(foldKey).filter(Boolean))].sort();
}

/** Adds (`on`) or removes a doc's keys from the exact-match map. */
export function indexKeys(state: IndexState, doc: DocEntry, on: boolean): void {
    for (const key of doc.keys) {
        let ids = state.keys.get(key);
        if (on) {
            if (!ids) state.keys.set(key, (ids = new Set()));
            ids.add(doc.id);
        } else if (ids) {
            ids.delete(doc.id);
            if (ids.size === 0) state.keys.delete(key);
        }
    }
}

/** The node index of `id` when its doc is live in the graph at the coded hash, else -1. */
export function graphNode(state: IndexState, id: string): number {
    const c = state.compacted;
    if (!c) return -1;
    const node = c.nodeOf.get(id);
    if (node === undefined) return -1;
    return state.docs.get(id)?.hash === c.hashes[node] ? node : -1;
}

/** The search mode a compaction of `size` cards uses under `params`. */
export function modeFor(params: IndexParams, size: number): IndexMode {
    return size < params.exactBelow ? 'exact' : 'graph';
}

/** Whether `embedder` may produce vectors comparable with the stored ones. */
export function denseReady(state: IndexState, embedder: EmbedderPort | undefined): embedder is EmbedderPort {
    return !!embedder && embedder.fingerprint === state.fingerprint && embedder.dims === state.dims;
}

/** The tiers a search can run right now. */
export function availableTiers(state: IndexState, embedder: EmbedderPort | undefined): IndexTier[] {
    return denseReady(state, embedder) ? ['exact', 'lexical', 'dense'] : ['exact', 'lexical'];
}
