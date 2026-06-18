/**
 * MemoryStoreBackend — local adapter mapping @seanhogg/builderforce-memory's MemoryStore
 * onto the MemoryBackend seam.
 *
 * Recall quality: when an SSM runtime is supplied it is forwarded to
 * `recallSimilar`, so recall uses SSM-embedding cosine similarity and improves
 * as the model is adapted/distilled. With no runtime it transparently falls
 * back to Jaccard word-overlap (still useful, just lexical).
 */

import type { MemoryBackend, RecallHit, RememberInput } from "../backend.js";

// Structural views of the @seanhogg/builderforce-memory surface we use, so this package
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
    recallAll(): Promise<MemoryEntryLike[]>;
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
    /**
     * Absolute path to a JSON file that mirrors the store to disk. Without it the
     * store is purely in-memory (fake-indexeddb) and evaporates when the process
     * exits — fine for a long-lived server, fatal for a per-session subprocess
     * (e.g. an MCP stdio client that respawns the server each launch).
     *
     * When set, the store is hydrated from the file on creation and re-snapshotted
     * after every remember/forget, giving durable cross-process memory. TTLs are
     * dropped on persist: the snapshot is the durable long-term tier.
     */
    persistFile?: string;
}

/** The on-disk snapshot shape — a flat array of durable entries. */
interface SnapshotEntry {
    key: string;
    content: string;
    tags?: string[];
    importance?: number;
}

type FsLike = {
    readFileSync(path: string, enc: "utf8"): string;
    writeFileSync(path: string, data: string): void;
    mkdirSync(path: string, opts: { recursive: boolean }): void;
    existsSync(path: string): boolean;
};
type PathLike = { dirname(p: string): string };

/**
 * Wraps a MemoryStoreBackend so every write is mirrored to a JSON file, and
 * hydrates that file back into the store on boot. This is what turns a respawned
 * stdio subprocess into a persistent memory: the store itself is in-memory, the
 * file is the source of truth across process lifetimes.
 */
class DiskPersistedBackend implements MemoryBackend {
    constructor(
        private readonly inner: MemoryStoreBackend,
        private readonly store: MemoryStoreLike,
        private readonly file: string,
        private readonly fs: FsLike,
    ) {}

    recall(query: string, topK: number): Promise<RecallHit[]> {
        return this.inner.recall(query, topK);
    }
    get(key: string): Promise<RecallHit | undefined> {
        return this.inner.get(key);
    }
    recallByTag(tag: string, limit: number): Promise<RecallHit[]> {
        return this.inner.recallByTag(tag, limit);
    }

    async remember(input: RememberInput): Promise<void> {
        await this.inner.remember(input);
        await this.snapshot();
    }

    async forget(key: string): Promise<void> {
        await this.inner.forget(key);
        await this.snapshot();
    }

    private async snapshot(): Promise<void> {
        const entries = await this.store.recallAll();
        const out: SnapshotEntry[] = entries.map((e) => ({
            key: e.key,
            content: e.content,
            tags: e.tags,
            importance: e.importance,
        }));
        this.fs.writeFileSync(this.file, JSON.stringify(out, null, 2));
    }
}

/** Reads a snapshot file and replays it into the store as durable (no-TTL) entries. */
async function hydrateFromDisk(store: MemoryStoreLike, file: string, fs: FsLike): Promise<void> {
    if (!fs.existsSync(file)) return;
    let parsed: unknown;
    try {
        parsed = JSON.parse(fs.readFileSync(file, "utf8"));
    } catch {
        // Corrupt/partial snapshot — start clean rather than crash the server.
        return;
    }
    if (!Array.isArray(parsed)) return;
    for (const raw of parsed as SnapshotEntry[]) {
        if (!raw || typeof raw.key !== "string" || typeof raw.content !== "string") continue;
        await store.remember(raw.key, raw.content, { tags: raw.tags, importance: raw.importance });
    }
}

/**
 * Builds a MemoryStoreBackend over a fresh MemoryStore, wiring fake-indexeddb in
 * Node exactly as SsmMemoryService does. @seanhogg/builderforce-memory and fake-indexeddb
 * are imported indirectly so they remain optional peers — a consumer that only
 * wants a custom backend (or the HTTP thin-client) never has to install them.
 */
export async function createLocalMemoryStoreBackend(opts: LocalBackendOptions = {}): Promise<MemoryBackend> {
    // Indirect import prevents the bundler/tsc from resolving optional peers.
    const _import = (m: string): Promise<unknown> =>
        // eslint-disable-next-line @typescript-eslint/no-implied-eval, no-new-func
        new Function("m", "return import(m)")(m) as Promise<unknown>;

    const memoryMod = (await _import("@seanhogg/builderforce-memory")) as { MemoryStore: new (o: unknown) => MemoryStoreLike };
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
    const backend = new MemoryStoreBackend(store, opts.runtime);

    if (!opts.persistFile) return backend;

    // Disk-mirror requested. node:fs/path are loaded indirectly so a browser
    // bundle of this module never statically pulls in Node builtins.
    const fs = (await _import("node:fs")) as FsLike;
    const path = (await _import("node:path")) as PathLike;
    const dir = path.dirname(opts.persistFile);
    if (dir && !fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    await hydrateFromDisk(store, opts.persistFile, fs);
    return new DiskPersistedBackend(backend, store, opts.persistFile, fs);
}
