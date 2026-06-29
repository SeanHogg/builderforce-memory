/**
 * transports/auth.ts — auth + rate-limiting primitives for the HTTP transport.
 *
 * The networked MCP surface is multi-tenant: one bearer token must map to ONE
 * tenant's memory namespace, the comparison must not leak the secret through a
 * timing side channel, and a noisy or hostile caller must not be able to hammer
 * the server. These helpers provide exactly those three pieces, dependency-free
 * (node:crypto only).
 */

import { createHash, timingSafeEqual } from "node:crypto";

/** SHA-256 hex digest of a token — the stable, non-reversible lookup key. */
export function hashToken(token: string): string {
    return createHash("sha256").update(token, "utf8").digest("hex");
}

/**
 * Constant-time string equality. Hashes both sides to fixed-length digests
 * first, so neither the comparison nor the input length leaks the secret.
 */
export function timingSafeEqualStr(a: string, b: string): boolean {
    const ha = createHash("sha256").update(a, "utf8").digest();
    const hb = createHash("sha256").update(b, "utf8").digest();
    return timingSafeEqual(ha, hb);
}

/**
 * Build a tenant lookup keyed by token HASH (never the raw token). Resolving a
 * presented token is a hash + map lookup — no per-character secret comparison,
 * so there is no early-return timing channel on the token value.
 */
export function buildTenantIndex<B>(tenants: Record<string, B>): Map<string, B> {
    const index = new Map<string, B>();
    for (const [token, backend] of Object.entries(tenants)) {
        if (token) index.set(hashToken(token), backend);
    }
    return index;
}

/** Fixed-window rate limiter. Memory-bounded by pruning expired windows. */
export class RateLimiter {
    private readonly _hits = new Map<string, { count: number; resetAt: number }>();

    constructor(
        private readonly windowMs: number,
        private readonly max: number,
        private readonly now: () => number = () => Date.now(),
    ) {}

    /** Returns true when the call is allowed; false when the window is exhausted. */
    check(key: string): boolean {
        const t = this.now();
        const entry = this._hits.get(key);
        if (!entry || t >= entry.resetAt) {
            if (this._hits.size > 10_000) this._prune(t);
            this._hits.set(key, { count: 1, resetAt: t + this.windowMs });
            return true;
        }
        if (entry.count >= this.max) return false;
        entry.count++;
        return true;
    }

    private _prune(t: number): void {
        for (const [k, v] of this._hits) {
            if (t >= v.resetAt) this._hits.delete(k);
        }
    }
}

/** Extract the bearer token from an Authorization header value, or undefined. */
export function bearerToken(header: string | string[] | undefined): string | undefined {
    const value = Array.isArray(header) ? header[0] : header;
    if (!value) return undefined;
    const m = /^Bearer\s+(.+)$/i.exec(value.trim());
    return m ? m[1]!.trim() : undefined;
}
