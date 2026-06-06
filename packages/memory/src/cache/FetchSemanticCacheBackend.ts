/**
 * FetchSemanticCacheBackend – the shared (L2) tier of the SemanticCache, backed
 * by the BuilderForce.ai gateway's vector store over HTTP.
 *
 * One client used by both consumers (browser + agent) so a semantic hit on one
 * surface is reusable by the other. Pure `fetch` — no environment-specific deps;
 * inject `fetchImpl` in tests.
 *
 * Wire protocol (gateway `/v1/semantic-cache`):
 *   POST /lookup  { embedding: number[], threshold, namespace? } → { hit?: { response, score } }
 *   POST /store   { embedding: number[], response, namespace?, meta? } → 2xx
 */

import type { SemanticCacheBackend } from './SemanticCache.js';

export interface FetchSemanticCacheBackendOptions {
    /** Gateway base URL, e.g. 'https://api.builderforce.ai'. Trailing slash trimmed. */
    baseUrl  : string;
    /** Tenant API key (sent as a bearer token). */
    apiKey   : string;
    /**
     * Optional cache partition. Scope hits to a tenant/model/agent so unrelated
     * traffic can't cross-hit. Defaults to the gateway's per-tenant default.
     */
    namespace? : string;
    /** Injectable fetch (defaults to global fetch). */
    fetchImpl? : typeof fetch;
}

export class FetchSemanticCacheBackend implements SemanticCacheBackend {
    private readonly _base      : string;
    private readonly _apiKey    : string;
    private readonly _namespace : string | undefined;
    private readonly _fetch     : typeof fetch;

    constructor(opts: FetchSemanticCacheBackendOptions) {
        this._base      = opts.baseUrl.replace(/\/$/, '');
        this._apiKey    = opts.apiKey;
        this._namespace = opts.namespace;
        this._fetch     = opts.fetchImpl ?? fetch;
    }

    async lookup(embedding: Float32Array, threshold: number): Promise<{ response: string; score: number } | undefined> {
        const res = await this._fetch(`${this._base}/v1/semantic-cache/lookup`, {
            method : 'POST',
            headers: this._headers(),
            body   : JSON.stringify({
                embedding: Array.from(embedding),
                threshold,
                ...(this._namespace ? { namespace: this._namespace } : {}),
            }),
        });
        if (!res.ok) return undefined;
        const json = await res.json().catch(() => null) as { hit?: { response?: unknown; score?: unknown } } | null;
        const hit = json?.hit;
        if (!hit || typeof hit.response !== 'string' || typeof hit.score !== 'number') return undefined;
        return { response: hit.response, score: hit.score };
    }

    async store(embedding: Float32Array, response: string, meta?: Record<string, unknown>): Promise<void> {
        await this._fetch(`${this._base}/v1/semantic-cache/store`, {
            method : 'POST',
            headers: this._headers(),
            body   : JSON.stringify({
                embedding: Array.from(embedding),
                response,
                ...(this._namespace ? { namespace: this._namespace } : {}),
                ...(meta ? { meta } : {}),
            }),
        });
    }

    private _headers(): Record<string, string> {
        return {
            'Content-Type' : 'application/json',
            Authorization  : `Bearer ${this._apiKey}`,
        };
    }
}
