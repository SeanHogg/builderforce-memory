/**
 * ResponseCache – a small, dependency-free read-through cache for transformer
 * bridge completions.
 *
 * This is the canonical cache for this library: a single bounded LRU with an
 * optional TTL, not an ad-hoc Map inlined at a call site. It exists because an
 * external LLM call is the most expensive thing the runtime does — identical
 * (model, system, prompt, sampling) requests should never be billed twice.
 *
 * Scope is in-process by design: this package targets the browser and Node, so
 * there is no shared KV / cross-isolate tier to propagate to (unlike the
 * BuilderForce.ai gateway, whose read-through cache is L1 Map + L2 KV). A
 * consumer that needs cross-process sharing can wrap a bridge with its own
 * distributed cache using the same `CachingBridge` shape.
 */

export interface ResponseCacheOptions {
    /**
     * Maximum number of entries retained. Oldest-accessed entries are evicted
     * first once the bound is reached. Default: 500.
     */
    maxEntries? : number;
    /**
     * Optional time-to-live in milliseconds. Entries older than this are treated
     * as misses and dropped on access. Omit for no expiry (cache until evicted).
     */
    ttlMs?      : number;
}

interface CacheRecord {
    value     : string;
    timestamp : number;
}

const DEFAULT_MAX_ENTRIES = 500;

export class ResponseCache {
    private readonly _maxEntries : number;
    private readonly _ttlMs      : number | undefined;
    // Map preserves insertion order; re-insertion on hit gives us LRU ordering.
    private readonly _store = new Map<string, CacheRecord>();

    private _hits   = 0;
    private _misses = 0;

    constructor(opts: ResponseCacheOptions = {}) {
        this._maxEntries = opts.maxEntries ?? DEFAULT_MAX_ENTRIES;
        this._ttlMs      = opts.ttlMs;
    }

    /**
     * Returns the cached value for `key`, or `undefined` on a miss (including an
     * expired entry, which is also evicted). A hit refreshes recency.
     */
    get(key: string): string | undefined {
        const record = this._store.get(key);
        if (!record) {
            this._misses++;
            return undefined;
        }
        if (this._isExpired(record)) {
            this._store.delete(key);
            this._misses++;
            return undefined;
        }
        // Refresh recency: delete + re-insert moves the key to the newest slot.
        this._store.delete(key);
        this._store.set(key, record);
        this._hits++;
        return record.value;
    }

    /** Stores `value` under `key`, evicting the least-recently-used entry if full. */
    set(key: string, value: string, now: number): void {
        if (this._store.has(key)) this._store.delete(key);
        this._store.set(key, { value, timestamp: now });

        while (this._store.size > this._maxEntries) {
            const oldest = this._store.keys().next().value;
            if (oldest === undefined) break;
            this._store.delete(oldest);
        }
    }

    /** Drops all cached entries. */
    clear(): void {
        this._store.clear();
    }

    /** Current entry count (including not-yet-evicted expired entries). */
    get size(): number {
        return this._store.size;
    }

    /** Cumulative hit / miss counters, for observability and cache-tuning. */
    get stats(): { hits: number; misses: number } {
        return { hits: this._hits, misses: this._misses };
    }

    private _isExpired(record: CacheRecord): boolean {
        if (this._ttlMs == null) return false;
        // `now` is read at access time so a single import of Date is enough; the
        // caller-supplied `now` on set() keeps insertion timestamps consistent.
        return Date.now() > record.timestamp + this._ttlMs;
    }
}

/**
 * Builds a stable, collision-resistant cache key from the request shape. Any
 * field that changes the model's output must be part of the key.
 */
export function buildCacheKey(parts: {
    prompt       : string;
    model?       : string;
    systemPrompt? : string;
    maxTokens?   : number;
    temperature? : number;
    topP?        : number;
}): string {
    // JSON of a fixed-order tuple — deterministic and unambiguous (a delimiter
    // string could collide across fields; positional JSON cannot).
    return JSON.stringify([
        parts.model        ?? '',
        parts.systemPrompt ?? '',
        parts.maxTokens    ?? '',
        parts.temperature  ?? '',
        parts.topP         ?? '',
        parts.prompt,
    ]);
}
