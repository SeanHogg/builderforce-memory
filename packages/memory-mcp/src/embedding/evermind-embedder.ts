/**
 * evermind-embedder — SSM-embedding recall for a HEADLESS host, with no GPU.
 *
 * The stdio/HTTP servers used to rank recall by word overlap because standing up
 * the embedding model was assumed to mean standing up WebGPU. It does not:
 * `EvermindLM` is a pure-CPU model and `EvermindLM.embed()` mean-pools + L2-
 * normalises the same hidden state the GPU `HybridMambaModel.embed()` does, to the
 * same contract. So a headless Node process can embed — it just needs the
 * checkpoint and the tokenizer that produced its token ids.
 *
 * This module is the loader for that pair, plus the ONE thing a per-session
 * subprocess needs to make embedding affordable: a vector cache that OUTLIVES the
 * process. Recall embeds the query plus every stored memory; recomputing 200
 * memory embeddings on the first recall of every session would cost more than the
 * lexical ranking it replaces. The cache is stamped with the model's fingerprint,
 * so an adapted checkpoint invalidates it wholesale rather than silently mixing
 * vectors from two different models.
 *
 * It mirrors how this package already persists memories: a JSON file alongside the
 * memory snapshot, written by the process that owns it, tolerant of a corrupt or
 * absent file (drop it and recompute — a cache is never the source of truth).
 */

import { dynamicImport } from "../dynamic-import.js";
import { loadEvermindPackage } from "../model/evermind-package.js";

/** The seam `MemoryStore.recallRanked(query, k, runtime)` consumes. */
export interface TextEmbedderLike {
    embed(text: string): Promise<Float32Array>;
    /** Vector width. */
    readonly dimensions: number;
    /** Model identity — vectors are only comparable within one fingerprint. */
    readonly fingerprint: string;
}

/** A text embedder that also owns a persistent cache. */
export interface PersistentTextEmbedder extends TextEmbedderLike {
    /** Write any pending vectors to disk now. Safe to call repeatedly. */
    flush(): void;
}

export interface EvermindEmbedderOptions {
    /**
     * Absolute path to a `.evermind` package holding an `evermind-lm` checkpoint.
     * This is the model whose hidden state becomes the embedding.
     */
    modelFile: string;
    /**
     * Absolute path to the tokenizer JSON that produced the checkpoint's token ids
     * (`BPETokenizer.toObject()`, or a Hugging Face `tokenizer.json`).
     *
     * Only needed for a package that does NOT embed its own tokenizer. Token ids are
     * meaningless without the exact vocabulary behind them, so an `.evermind` built
     * with `PackageMeta.tokenizer` carries one and is self-contained — that is
     * preferred over any file, because a file can be the WRONG vocabulary while the
     * embedded one is checksummed and vocab-size-checked against the checkpoint.
     * Resolution order: the package's own tokenizer, then this path, then a sibling
     * `<modelFile>.tokenizer.json`.
     */
    tokenizerFile?: string;
    /**
     * Absolute path for the persistent vector cache. Omit to disable persistence
     * (vectors are still cached in memory for the life of the process).
     */
    cacheFile?: string;
    /** Max vectors retained in the cache (LRU past this). Default 4000. */
    maxEntries?: number;
    /** Debounce before a dirty cache is written back, ms. Default 500. */
    flushDelayMs?: number;
}

type FsLike = {
    readFileSync(path: string, enc: "utf8"): string;
    writeFileSync(path: string, data: string): void;
    existsSync(path: string): boolean;
};

const DEFAULT_MAX_ENTRIES = 4000;
const DEFAULT_FLUSH_DELAY_MS = 500;

/** The on-disk cache shape: a fingerprint stamp plus text→vector rows. */
interface VectorCacheFile {
    fingerprint: string;
    dimensions: number;
    /** [text, vector] pairs, least-recently-used first (insertion order is the LRU). */
    vectors: Array<[string, number[]]>;
}

/**
 * Build a CPU text embedder from a `.evermind` package + its tokenizer, wrapped in
 * a persistent vector cache.
 *
 * Returns `null` — never throws — when the engine peer is absent, a file is
 * missing, or the package is not an `evermind-lm`. Recall then stays on the
 * lexical ranking and says so, which is the correct degrade: a memory server that
 * refuses to start because an OPTIONAL model is unavailable is worse than one that
 * ranks by word overlap.
 */
export async function createEvermindEmbedder(
    opts: EvermindEmbedderOptions,
): Promise<PersistentTextEmbedder | null> {
    try {
        const loaded = await loadEvermindPackage(opts.modelFile, opts.tokenizerFile);
        if (!loaded) return null;
        const engine = (await dynamicImport("@seanhogg/builderforce-memory-engine")) as {
            EvermindTextEmbedder?: new (model: unknown, codec: unknown) => TextEmbedderLike;
        };
        if (!engine?.EvermindTextEmbedder) return null;
        const inner = new engine.EvermindTextEmbedder(loaded.pkg.loadLM(), loaded.codec);
        return new CachedTextEmbedder(inner, loaded.fs, opts);
    } catch {
        // Any failure to stand the model up is a degrade, not a fault.
        return null;
    }
}

/**
 * Wraps a text embedder with an LRU vector cache mirrored to a JSON file.
 *
 * Exported for hosts that already hold an embedder (a GPU runtime, say) and want
 * the same cross-process cache — there is one implementation of "cache these
 * vectors on disk", not one per embedder.
 */
export class CachedTextEmbedder implements PersistentTextEmbedder {
    readonly dimensions: number;
    readonly fingerprint: string;

    /** text → vector. Map insertion order is the LRU order. */
    private readonly cache = new Map<string, Float32Array>();
    private readonly file: string | undefined;
    private readonly maxEntries: number;
    private readonly flushDelayMs: number;
    private dirty = false;
    private timer: ReturnType<typeof setTimeout> | null = null;

    constructor(
        private readonly inner: TextEmbedderLike,
        private readonly fs: FsLike,
        opts: Pick<EvermindEmbedderOptions, "cacheFile" | "maxEntries" | "flushDelayMs">,
    ) {
        this.dimensions = inner.dimensions;
        this.fingerprint = inner.fingerprint;
        this.file = opts.cacheFile;
        this.maxEntries = Math.max(1, opts.maxEntries ?? DEFAULT_MAX_ENTRIES);
        this.flushDelayMs = Math.max(0, opts.flushDelayMs ?? DEFAULT_FLUSH_DELAY_MS);
        this.hydrate();
    }

    async embed(text: string): Promise<Float32Array> {
        const hit = this.cache.get(text);
        if (hit) {
            // LRU touch — re-insert so this entry becomes most-recently-used.
            this.cache.delete(text);
            this.cache.set(text, hit);
            return hit;
        }
        const vec = await this.inner.embed(text);
        this.put(text, vec);
        return vec;
    }

    /** Write pending vectors now, cancelling any scheduled write. */
    flush(): void {
        if (this.timer) {
            clearTimeout(this.timer);
            this.timer = null;
        }
        if (!this.dirty || !this.file) return;
        const payload: VectorCacheFile = {
            fingerprint: this.fingerprint,
            dimensions: this.dimensions,
            vectors: [...this.cache.entries()].map(([text, v]) => [text, Array.from(v)]),
        };
        try {
            this.fs.writeFileSync(this.file, JSON.stringify(payload));
            this.dirty = false;
        } catch {
            // A cache that cannot be written is still a working in-memory cache.
        }
    }

    private put(text: string, vec: Float32Array): void {
        if (this.cache.size >= this.maxEntries) {
            const lru = this.cache.keys().next().value;
            if (lru !== undefined) this.cache.delete(lru);
        }
        this.cache.set(text, vec);
        this.dirty = true;
        this.schedule();
    }

    /**
     * Debounce the write-back: one recall embeds the query plus every new memory,
     * and that burst should cost ONE file write, not one per vector.
     */
    private schedule(): void {
        if (!this.file || this.timer) return;
        this.timer = setTimeout(() => {
            this.timer = null;
            this.flush();
        }, this.flushDelayMs);
        // Never hold the process open for a cache write.
        (this.timer as unknown as { unref?: () => void }).unref?.();
    }

    /**
     * Read the cache back. A cache stamped with a DIFFERENT fingerprint is from a
     * different (or adapted) model, so its vectors are not comparable with this
     * model's — it is dropped whole rather than mixed in.
     */
    private hydrate(): void {
        if (!this.file || !this.fs.existsSync(this.file)) return;
        let parsed: VectorCacheFile;
        try {
            parsed = JSON.parse(this.fs.readFileSync(this.file, "utf8")) as VectorCacheFile;
        } catch {
            return;
        }
        if (!parsed || parsed.fingerprint !== this.fingerprint || !Array.isArray(parsed.vectors)) return;
        for (const row of parsed.vectors) {
            if (!Array.isArray(row) || typeof row[0] !== "string" || !Array.isArray(row[1])) continue;
            if (row[1].length !== this.dimensions) continue;
            this.cache.set(row[0], Float32Array.from(row[1]));
        }
    }
}
