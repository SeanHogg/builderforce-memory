/**
 * MemoryBackend — the storage seam every transport is written against.
 *
 * The MCP tools (src/tools.ts) and all three transports (SDK / stdio / HTTP)
 * depend ONLY on this interface, never on a concrete store. Ship the local
 * `MemoryStoreBackend` (IndexedDB via @builderforce/memory) today; drop in a
 * networked builderforce.ai adapter later with zero changes to the tools or
 * transports.
 */

/** A single recalled memory, normalised across backends. */
export interface RecallHit {
    /** Stable identifier for the memory. */
    key: string;
    /** The stored value. */
    content: string;
    /**
     * Optional relevance score (higher = closer). Semantic backends that expose
     * ranking can populate this; the local MemoryStore ranks but does not surface
     * a score, so it is left undefined and recall order carries the signal.
     */
    score?: number;
    /** Tags for grouping/filtering. */
    tags?: string[];
    /** Importance weight 0–1. */
    importance?: number;
    /** Unix-ms write time. */
    timestamp?: number;
}

/** Arguments for writing a memory. */
export interface RememberInput {
    key: string;
    content: string;
    tags?: string[];
    /** Importance weight 0–1. */
    importance?: number;
    /** Time-to-live in milliseconds. */
    ttlMs?: number;
}

/**
 * The minimal capability surface the MCP layer needs. Deliberately small —
 * the token-saving design exposes recall-on-demand, not a "dump everything"
 * call, so this interface has no `recallAll`.
 */
export interface MemoryBackend {
    /**
     * Semantic top-K recall. Backends with an embedding model (the SSM
     * runtime) should use it; lexical fallback is acceptable. `topK` is already
     * clamped by the caller — the backend may return fewer, never more.
     */
    recall(query: string, topK: number): Promise<RecallHit[]>;

    /** Exact lookup by key. Returns undefined when absent or expired. */
    get(key: string): Promise<RecallHit | undefined>;

    /** All non-expired entries carrying `tag`, capped to `limit`. */
    recallByTag(tag: string, limit: number): Promise<RecallHit[]>;

    /** Store or overwrite a memory. Optional — read-only backends omit it. */
    remember?(input: RememberInput): Promise<void>;

    /** Delete a memory by key. Optional — read-only backends omit it. */
    forget?(key: string): Promise<void>;
}
