/**
 * SemanticCache – an embedding-keyed read-through cache for LLM completions.
 *
 * Unlike the exact-match ResponseCache (which keys on the byte-identical prompt),
 * this keys on the *meaning* of the query: it embeds the query and serves a
 * cached answer when a stored entry is within `threshold` cosine similarity.
 * That catches paraphrases — "fix the auth bug" ≈ "login is broken" — which is
 * where real frontier-call avoidance (and token savings) comes from.
 *
 * Two tiers, mirroring the project's L1-in-process / L2-shared read-through
 * pattern:
 *   - L1: an in-process vector list, scanned locally (fast, offline-capable).
 *   - L2: an optional shared backend (e.g. the BuilderForce.ai gateway vector
 *         store) so a hit on one surface — web or agent — benefits the other.
 *
 * Fully portable: the embedder and the L2 backend are injected, so the same
 * class runs in the browser (WebGPU SSM + native fetch) and in Node (the agent's
 * `@webgpu/node` SSM + fetch) with no environment-specific forks.
 */

import { cosineSimilarity } from '../similarity/index.js';

/** Produces an embedding vector for a piece of text (the on-device SSM, typically). */
export type Embedder = (text: string) => Promise<Float32Array>;

/**
 * The shared (L2) cache tier. Implemented by `FetchSemanticCacheBackend` against
 * the gateway, but any store satisfying this shape can be injected.
 */
export interface SemanticCacheBackend {
    /** Returns the best stored entry at/above `threshold` cosine similarity, or undefined. */
    lookup(embedding: Float32Array, threshold: number): Promise<{ response: string; score: number } | undefined>;
    /** Persists an embedding → response association. */
    store(embedding: Float32Array, response: string, meta?: Record<string, unknown>): Promise<void>;
}

export interface SemanticCacheHit {
    response: string;
    /** Cosine similarity of the matched entry to the query. */
    score: number;
    /** Which tier served the hit. */
    tier: 'l1' | 'l2';
}

export interface SemanticCacheOptions {
    /** Embeds queries. Required — this is what makes the cache semantic. */
    embed: Embedder;
    /**
     * Cosine similarity at/above which a stored entry counts as a hit.
     * Higher = stricter (fewer false hits, lower hit rate). Default: 0.92.
     */
    threshold?: number;
    /** Max L1 entries retained (oldest evicted first). Default: 500. */
    maxEntries?: number;
    /** Optional TTL (ms) for L1 entries. Omit for no expiry. */
    ttlMs?: number;
    /** Optional shared L2 backend (e.g. the gateway). */
    l2?: SemanticCacheBackend;
    /**
     * When true (default), an answer served by L2 is also written into L1 so the
     * next local lookup is a fast hit — read-through cache warming.
     */
    warmL1FromL2?: boolean;
}

interface L1Entry { embedding: Float32Array; response: string; timestamp: number; }

const DEFAULT_THRESHOLD   = 0.92;
const DEFAULT_MAX_ENTRIES = 500;

export class SemanticCache {
    private readonly _embed      : Embedder;
    private readonly _threshold  : number;
    private readonly _maxEntries : number;
    private readonly _ttlMs      : number | undefined;
    private readonly _l2         : SemanticCacheBackend | undefined;
    private readonly _warmL1     : boolean;
    private readonly _l1         : L1Entry[] = [];

    private _l1Hits = 0;
    private _l2Hits = 0;
    private _misses = 0;

    constructor(opts: SemanticCacheOptions) {
        this._embed      = opts.embed;
        this._threshold  = opts.threshold  ?? DEFAULT_THRESHOLD;
        this._maxEntries = opts.maxEntries ?? DEFAULT_MAX_ENTRIES;
        this._ttlMs      = opts.ttlMs;
        this._l2         = opts.l2;
        this._warmL1     = opts.warmL1FromL2 ?? true;
    }

    /**
     * Read-through entry point: returns a cached answer for a semantically-similar
     * prior query, otherwise runs `generate()`, stores the result in both tiers,
     * and returns it. Embeds the query exactly once (lookup + store share it).
     */
    async getOrGenerate(
        query: string,
        generate: () => Promise<string>,
        meta?: Record<string, unknown>,
    ): Promise<{ response: string; cached: boolean; tier?: 'l1' | 'l2'; score?: number }> {
        const qv  = await this._embed(query);
        const hit = await this._lookupVec(qv);
        if (hit) return { response: hit.response, cached: true, tier: hit.tier, score: hit.score };

        const response = await generate();
        await this._storeVec(qv, response, meta);
        return { response, cached: false };
    }

    /** Looks up a semantically-similar cached answer without generating on a miss. */
    async lookup(query: string): Promise<SemanticCacheHit | undefined> {
        return this._lookupVec(await this._embed(query));
    }

    /** Stores a query → response association in both tiers. */
    async store(query: string, response: string, meta?: Record<string, unknown>): Promise<void> {
        await this._storeVec(await this._embed(query), response, meta);
    }

    /** Drops all L1 entries. Does not touch the shared L2 backend. */
    clear(): void {
        this._l1.length = 0;
    }

    /** Current L1 entry count. */
    get size(): number {
        return this._l1.length;
    }

    /** Cumulative hit/miss counters across both tiers — for measuring savings. */
    get stats(): { l1Hits: number; l2Hits: number; misses: number } {
        return { l1Hits: this._l1Hits, l2Hits: this._l2Hits, misses: this._misses };
    }

    // ── Internals (operate on a precomputed embedding) ────────────────────────

    private async _lookupVec(qv: Float32Array): Promise<SemanticCacheHit | undefined> {
        const local = this._searchL1(qv);
        if (local) {
            this._l1Hits++;
            return { response: local.response, score: local.score, tier: 'l1' };
        }

        if (this._l2) {
            // L2 is best-effort: a gateway error degrades to local-only, never throws.
            const remote = await this._l2.lookup(qv, this._threshold).catch(() => undefined);
            if (remote && remote.score >= this._threshold) {
                if (this._warmL1) this._addL1(qv, remote.response);
                this._l2Hits++;
                return { response: remote.response, score: remote.score, tier: 'l2' };
            }
        }

        this._misses++;
        return undefined;
    }

    private async _storeVec(qv: Float32Array, response: string, meta?: Record<string, unknown>): Promise<void> {
        this._addL1(qv, response);
        if (this._l2) {
            // Best-effort: failing to share to L2 must not fail the caller's request.
            await this._l2.store(qv, response, meta).catch(() => { /* swallow — local copy still cached */ });
        }
    }

    /** Linear cosine scan over L1, dropping expired entries en route. */
    private _searchL1(qv: Float32Array): { response: string; score: number } | undefined {
        const now = Date.now();
        let best: L1Entry | undefined;
        let bestScore = -Infinity;

        for (let i = this._l1.length - 1; i >= 0; i--) {
            const entry = this._l1[i]!;
            if (this._ttlMs != null && now > entry.timestamp + this._ttlMs) {
                this._l1.splice(i, 1);
                continue;
            }
            const score = cosineSimilarity(qv, entry.embedding);
            if (score > bestScore) {
                bestScore = score;
                best = entry;
            }
        }

        return best && bestScore >= this._threshold
            ? { response: best.response, score: bestScore }
            : undefined;
    }

    private _addL1(qv: Float32Array, response: string): void {
        this._l1.push({ embedding: qv, response, timestamp: Date.now() });
        while (this._l1.length > this._maxEntries) this._l1.shift();
    }
}
