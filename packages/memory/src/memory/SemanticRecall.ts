/**
 * memory/SemanticRecall.ts – The query side of {@link MemoryStore}: semantic
 * (SSM-embedding cosine, Jaccard fallback) and hybrid (RRF + MMR) recall over
 * an already-loaded fact list, plus the derived caches that make it cheap:
 *
 *   - an LRU content→embedding cache (EVM-8);
 *   - a persistent dense vector store keyed by `key@seq` and the HNSW index
 *     built from it, reused until the fact set changes (EVM-1 / EVM-1b).
 *
 * Every cache is model-specific and is dropped when the runtime identity
 * changes, or when the owning store calls {@link SemanticRecall.invalidate}
 * (clear, cross-context write).
 */

import { tokenize, jaccardSimilarity } from '../similarity/index.js';
import { hybridRetrieve, denseSearch, HnswIndex, type RetrievalCandidate, type HybridRetrieveOptions } from '../retrieval/index.js';
import type { MemoryEntry, RankedRecall, ScoredMemory, SSMRuntimeRef } from './types.js';

export interface SemanticRecallOptions {
    /** Fact count at/above which dense search uses the HNSW index. */
    annThreshold: number;
    /** Max content→embedding vectors held in the LRU cache. */
    embedCacheMax: number;
    /** Hits scoring below this are dropped (0 = keep everything). */
    minRelevance: number;
}

interface DenseView {
    byKey: Map<string, MemoryEntry>;
    items: Array<{ id: string; vector: Float32Array }>;
    index: HnswIndex | null;
}

export class SemanticRecall {
    private readonly _embedCache = new Map<string, Float32Array>();
    private _embedRuntime: unknown = null;
    private readonly _denseVectors = new Map<string, Float32Array>();
    private _denseIndex: HnswIndex | null = null;
    /** Signature of the fact set the current index was built from. */
    private _denseSig = '';
    private _denseRuntime: unknown = null;

    constructor(private readonly _opts: SemanticRecallOptions) {}

    /** Drops every derived cache (the fact set changed underneath us). */
    invalidate(): void {
        this._embedCache.clear();
        this.resetDense();
    }

    /** Drops the dense vectors + index but keeps content embeddings. */
    resetDense(): void {
        this._denseVectors.clear();
        this._denseIndex = null;
        this._denseSig = '';
    }

    /**
     * Top-K entries of `all` by real 0..1 relevance: SSM-embedding cosine when
     * `runtime.embed` works, Jaccard word overlap otherwise.
     */
    async similar(all: MemoryEntry[], query: string, topK: number, runtime?: SSMRuntimeRef): Promise<ScoredMemory[]> {
        if (all.length === 0) return [];
        if (runtime != null && typeof runtime.embed === 'function') {
            const queryVec = await this._embed(runtime, query);
            if (queryVec) {
                // EVM-1b: sync the persistent vector store + index (embeds only the
                // delta), then query it — no per-call re-embed of every fact.
                const dense = await this._syncDense(runtime, all);
                if (dense) {
                    const hits = dense.index
                        ? dense.index.search(queryVec, topK)
                        : denseSearch(queryVec, dense.items, topK, this._opts.annThreshold);
                    return this._floor(hits
                        .map((h) => { const entry = dense.byKey.get(h.id); return entry ? { entry, score: h.score } : null; })
                        .filter((e): e is ScoredMemory => !!e));
                }
            }
            // Any embedding failure falls through to the Jaccard path below.
        }
        const queryTokens = new Set(tokenize(query));
        const scored = all.map((entry) => ({ entry, score: jaccardSimilarity(queryTokens, new Set(tokenize(entry.content))) }));
        scored.sort((a, b) => b.score - a.score);
        return this._floor(scored).slice(0, topK);
    }

    /**
     * Hybrid recall: dense + BM25 fused by the shared `hybridRetrieve` primitive,
     * with an honest `method` — `'embedding'` only when the query AND at least
     * one candidate really embedded.
     */
    async ranked(
        all: MemoryEntry[],
        query: string,
        topK: number,
        runtime?: SSMRuntimeRef,
        opts?: HybridRetrieveOptions,
    ): Promise<RankedRecall> {
        if (all.length === 0) return { hits: [], method: 'lexical' };
        const canEmbed = runtime != null && typeof runtime.embed === 'function';
        const queryVec = canEmbed ? (await this._embed(runtime, query)) ?? undefined : undefined;
        const candidates: RetrievalCandidate[] = [];
        for (const entry of all) {
            const vector = queryVec ? (await this._embed(runtime, entry.content)) ?? undefined : undefined;
            candidates.push({ id: entry.key, text: entry.content, vector });
        }
        const dense = !!queryVec && candidates.some((c) => !!c.vector);
        const hits = hybridRetrieve({ text: query, vector: queryVec }, candidates, { topK, annThreshold: this._opts.annThreshold, ...opts });
        const byKey = new Map(all.map((e) => [e.key, e]));
        return {
            hits: this._floor(hits
                .map((h) => { const entry = byKey.get(h.id); return entry ? { entry, score: h.score } : null; })
                .filter((h): h is ScoredMemory => !!h)),
            method: dense ? 'embedding' : 'lexical',
        };
    }

    private _floor(hits: ScoredMemory[]): ScoredMemory[] {
        const min = this._opts.minRelevance;
        return min > 0 ? hits.filter((h) => h.score >= min) : hits;
    }

    /**
     * Cached embedding for `text`, computed via `runtime.embed()` on a miss.
     * `null` (never throws) when embedding is unavailable, so callers fall back
     * to lexical similarity.
     */
    private async _embed(runtime: SSMRuntimeRef, text: string): Promise<Float32Array | null> {
        if (runtime !== this._embedRuntime) {
            this._embedCache.clear();
            this._embedRuntime = runtime;
        }
        const cached = this._embedCache.get(text);
        if (cached) {
            // LRU touch: re-insert so this entry becomes most-recently-used. (EVM-8)
            this._embedCache.delete(text);
            this._embedCache.set(text, cached);
            return cached;
        }
        let vec: unknown;
        try {
            vec = await runtime.embed(text);
        } catch {
            // Embedding unavailable (no GPU, destroyed runtime, etc.) — the
            // documented lexical fallback, reported to callers as `method`.
            return null;
        }
        if (!(vec instanceof Float32Array) || vec.length === 0) return null;
        if (this._embedCache.size >= this._opts.embedCacheMax) {
            const lru = this._embedCache.keys().next().value;
            if (lru !== undefined) this._embedCache.delete(lru);
        }
        this._embedCache.set(text, vec);
        return vec;
    }

    /**
     * Synchronise the dense vector store + HNSW index to `all`: embeds ONLY
     * new/changed facts (keyed by `key@seq`), prunes removed ones, and rebuilds
     * the index only when the fact set changed. `null` when any embedding is
     * unavailable (caller falls back to lexical).
     */
    private async _syncDense(runtime: SSMRuntimeRef, all: MemoryEntry[]): Promise<DenseView | null> {
        if (runtime !== this._denseRuntime) {
            this.resetDense();
            this._denseRuntime = runtime;
        }
        const vkeyOf = (e: MemoryEntry): string => `${e.key}@${e.seq ?? 0}`;
        const currentVkeys = new Set<string>();
        let sig = `${all.length}`;
        for (const e of all) { const k = vkeyOf(e); currentVkeys.add(k); sig += '|' + k; }

        const byKey = new Map<string, MemoryEntry>();
        const items: Array<{ id: string; vector: Float32Array }> = [];
        for (const entry of all) {
            const vk = vkeyOf(entry);
            let vec = this._denseVectors.get(vk);
            if (!vec) {
                const v = await this._embed(runtime, entry.content);
                if (!v) return null;
                vec = v;
                this._denseVectors.set(vk, vec);
            }
            items.push({ id: entry.key, vector: vec });
            byKey.set(entry.key, entry);
        }
        if (this._denseVectors.size > currentVkeys.size) {
            for (const k of this._denseVectors.keys()) {
                if (!currentVkeys.has(k)) this._denseVectors.delete(k);
            }
        }
        if (all.length >= this._opts.annThreshold) {
            if (sig !== this._denseSig || this._denseIndex === null) {
                const index = new HnswIndex();
                for (const it of items) index.add(it.id, it.vector);
                this._denseIndex = index;
            }
        } else {
            this._denseIndex = null;
        }
        this._denseSig = sig;
        return { byKey, items, index: this._denseIndex };
    }
}
