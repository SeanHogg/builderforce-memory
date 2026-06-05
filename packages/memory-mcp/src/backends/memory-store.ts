/**
 * MemoryStoreBackend — local adapter mapping @builderforce/memory's MemoryStore
 * onto the MemoryBackend seam.
 *
 * Recall quality: when an SSM runtime is supplied it is forwarded to
 * `recallSimilar`, so recall uses SSM-embedding cosine similarity and improves
 * as the model is adapted/distilled. With no runtime it transparently falls
 * back to Jaccard word-overlap (still useful, just lexical).
 */

import type { MemoryBackend, RecallHit, RememberInput } from "../backend.js";

// Structural views of the @builderforce/memory surface we use, so this package
// type-checks without a hard dependency on the runtime package.
interface MemoryEntryLike {
    key: string;
    content: string;
    timestamp?: number;
    tags?: string[];
    importance?: number;
}

interface MemoryStoreLike {
    remember(key: string, content: string, opts?: { ttlMs?: number; tags?: string[]; importance?: number }): Promise<void>;
    recall(key: string): Promise<MemoryEntryLike | undefined>;
    recallByTag(tag: string): Promise<MemoryEntryLike[]>;
    recallSimilar(query: string, topK: number, runtime?: unknown): Promise<MemoryEntryLike[]>;
    forget(key: string): Promise<void>;
}

function toHit(e: MemoryEntryLike): RecallHit {
    return {
        key: e.key,
        content: e.content,
        tags: e.tags,
        importance: e.importance,
        timestamp: e.timestamp,
    };
}

export class MemoryStoreBackend implements MemoryBackend {
    constructor(
        private readonly store: MemoryStoreLike,
        /** Optional SSMRuntime; enables embedding-based recall when present. */
        private readonly runtime?: unknown,
    ) {}

    async recall(query: string, topK: number): Promise<RecallHit[]> {
        const entries = await this.store.recallSimilar(query, topK, this.runtime);
        return entries.map(toHit);
    }

    async get(key: string): Promise<RecallHit | undefined> {
        const e = await this.store.recall(key);
        return e ? toHit(e) : undefined;
    }

    async recallByTag(tag: string, limit: number): Promise<RecallHit[]> {
        const entries = await this.store.recallByTag(tag);
        return entries.slice(0, limit).map(toHit);
    }

    async remember(input: RememberInput): Promise<void> {
        await this.store.remember(input.key, input.content, {
            ttlMs: input.ttlMs,
            tags: input.tags,
            importance: input.importance,
        });
    }

    async forget(key: string): Promise<void> {
        await this.store.forget(key);
    }
}

/** Options for {@link createLocalMemoryStoreBackend}. */
export interface LocalBackendOptions {
    /** IndexedDB database name. Defaults to MemoryStore's own default ('ssmjs'). */
    dbName?: string;
    /**
     * Optional SSMRuntime for embedding-based recall. Omit for lexical (Jaccard)
     * recall. In the agent-runtime, pass `ssmMemoryService.runtime` here to reuse
     * the already-loaded hippocampus instead of standing up a second model.
     */
    runtime?: unknown;
}

/**
 * Builds a MemoryStoreBackend over a fresh MemoryStore, wiring fake-indexeddb in
 * Node exactly as SsmMemoryService does. @builderforce/memory and fake-indexeddb
 * are imported indirectly so they remain optional peers — a consumer that only
 * wants a custom backend (or the HTTP thin-client) never has to install them.
 */
export async function createLocalMemoryStoreBackend(opts: LocalBackendOptions = {}): Promise<MemoryBackend> {
    // Indirect import prevents the bundler/tsc from resolving optional peers.
    const _import = (m: string): Promise<unknown> =>
        // eslint-disable-next-line @typescript-eslint/no-implied-eval, no-new-func
        new Function("m", "return import(m)")(m) as Promise<unknown>;

    const memoryMod = (await _import("@builderforce/memory")) as { MemoryStore: new (o: unknown) => MemoryStoreLike };
    const { MemoryStore } = memoryMod;

    // IndexedDB shim for Node. In the browser the global is used automatically.
    let idbFactory: unknown;
    try {
        const fake = (await _import("fake-indexeddb")) as { IDBFactory: new () => unknown };
        idbFactory = new fake.IDBFactory();
    } catch {
        // Browser or a host that provides global indexedDB — MemoryStore handles it.
    }

    const store = new MemoryStore({ idbFactory, dbName: opts.dbName });
    return new MemoryStoreBackend(store, opts.runtime);
}
