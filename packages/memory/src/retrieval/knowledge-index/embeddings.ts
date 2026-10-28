/**
 * embeddings.ts — the index's recompute path: card text → unit float vector,
 * through a bounded LRU keyed by the card's content hash.
 *
 * LEANN's point is that most vectors need not be stored: the dense tier
 * recomputes float vectors only for the handful of candidates it reranks, and
 * lean mode recomputes the codes it dropped. Keying by hash means an unchanged
 * card is never re-embedded while it stays warm, and a changed one never serves
 * a stale vector.
 */

import type { EmbedderPort } from './types.js';
import { toUnit } from './quantize.js';

/** Default number of vectors held. */
export const EMBEDDING_CACHE_MAX = 8192;
/** Texts per `embed` call. */
export const EMBED_BATCH = 256;

export interface Embeddable {
    hash: string;
    text: string;
}

export class EmbeddingCache {
    private readonly _lru = new Map<string, Float32Array>();

    constructor(
        private readonly _embedder: EmbedderPort,
        private readonly _max: number = EMBEDDING_CACHE_MAX,
    ) {}

    /** Unit vectors for `items`, in order. Misses are embedded in batches of {@link EMBED_BATCH}. */
    async vectors(items: readonly Embeddable[]): Promise<Float32Array[]> {
        const out: Array<Float32Array | undefined> = items.map((it) => this._touch(it.hash));
        const missing: number[] = [];
        const seen = new Map<string, number>();
        items.forEach((it, i) => {
            if (out[i] || seen.has(it.hash)) return;
            seen.set(it.hash, missing.length);
            missing.push(i);
        });
        if (missing.length > 0) {
            const raw: Float32Array[] = [];
            for (let start = 0; start < missing.length; start += EMBED_BATCH) {
                const batch = missing.slice(start, start + EMBED_BATCH).map((i) => items[i]!.text);
                raw.push(...(await this._embedder.embed(batch)));
            }
            missing.forEach((i, j) => {
                const vec = raw[j];
                if (!vec) throw new Error('knowledge-index: embedder returned fewer vectors than texts');
                this._put(items[i]!.hash, toUnit(vec));
            });
            items.forEach((it, i) => {
                if (!out[i]) out[i] = this._touch(it.hash) ?? toUnit(raw[seen.get(it.hash)!]!);
            });
        }
        return out as Float32Array[];
    }

    /** A query (not a card) vector: not cached, the query space is unbounded. */
    async query(text: string): Promise<Float32Array> {
        const [vec] = await this._embedder.embed([text]);
        if (!vec) throw new Error('knowledge-index: embedder returned no vector for the query');
        return toUnit(vec);
    }

    private _touch(hash: string): Float32Array | undefined {
        const hit = this._lru.get(hash);
        if (hit) {
            this._lru.delete(hash);
            this._lru.set(hash, hit);
        }
        return hit;
    }

    private _put(hash: string, vec: Float32Array): void {
        this._lru.delete(hash);
        this._lru.set(hash, vec);
        while (this._lru.size > this._max) {
            const oldest = this._lru.keys().next().value as string;
            this._lru.delete(oldest);
        }
    }
}
