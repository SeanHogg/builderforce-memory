/**
 * MemoryStoreBackend — local adapter mapping @seanhogg/builderforce-memory's MemoryStore
 * onto the MemoryBackend seam.
 *
 * Recall quality: when a text embedder is supplied it is forwarded to the store's
 * `recallRanked`, so the ordering is the shared hybrid retrieval — the SSM
 * embedding's cosine ranking FUSED with BM25 by Reciprocal Rank Fusion — and it
 * sharpens as the model is adapted/distilled. With no embedder the same call
 * degrades to the lexical ranking alone. Either way the backend reports which of
 * the two produced the ordering (`method`), so a silent degrade is visible.
 *
 * The embedder does NOT have to be a GPU runtime: `EvermindTextEmbedder` from the
 * engine is a pure-CPU `EvermindLM`, which is what lets the headless stdio server
 * rank recall semantically (see ./evermind-embedder.ts).
 *
 * Ranking signals: recall also weights each fused score by the fact's importance
 * (x(0.5+importance), the cloud recall's factor) and a half-life recency decay —
 * both ON by default here ({@link DEFAULT_RECALL_RANKING}).
 *
 * Compaction stubs: a stored body that is an `[absorbed→Evermind …]` stub is
 * resolved to its ARCHIVED original (../persistence/fact-archive.ts) on every read,
 * so recall returns the full text; a stub with no archive is flagged `contentLost`.
 */

import type { MemoryBackend, RankedRecall, RecallHit, RememberInput } from "../backend.js";
import { defaultArchiveFile, isContentLost, isStub, type ArchivedFact } from "../compaction.js";
import { dynamicImport } from "../dynamic-import.js";
import { FileFactArchive, MemoryFactArchive, type FactArchive } from "../persistence/fact-archive.js";
import { SharedJsonFile, type SharedFileFs } from "../persistence/shared-json-file.js";

// Structural views of the @seanhogg/builderforce-memory surface we use, so this package
// type-checks without a hard dependency on the runtime package.
interface MemoryEntryLike {
    key: string;
    content: string;
    timestamp?: number;
    tags?: string[];
    importance?: number;
}

/**
 * Recall ranking signals — the subset of the store's `HybridRetrieveOptions` this
 * backend sets. Structural, so this package needs no type import from the runtime.
 */
export interface RecallRanking {
    /** Weight each fused score by `0.5 + importance`. */
    importanceWeight?: boolean;
    /** Half-life (days) of the recency decay over each fact's write time; 0 = off. */
    recencyHalfLifeDays?: number;
}

/** Recall ranking defaults: importance ON, 90-day half-life recency decay ON. */
export const DEFAULT_RECALL_RANKING: Readonly<Required<RecallRanking>> = Object.freeze({
    importanceWeight: true,
    recencyHalfLifeDays: 90,
});

interface MemoryStoreLike {
    remember(
        key: string,
        content: string,
        opts?: { ttlMs?: number; tags?: string[]; importance?: number; timestamp?: number },
    ): Promise<void>;
    recall(key: string): Promise<MemoryEntryLike | undefined>;
    recallAll(): Promise<MemoryEntryLike[]>;
    recallByTag(tag: string): Promise<MemoryEntryLike[]>;
    /** Hybrid (dense⊕BM25) recall with each hit's fused score and the ranker used. */
    recallRanked(
        query: string,
        topK: number,
        runtime?: unknown,
        opts?: RecallRanking,
    ): Promise<{ hits: Array<{ entry: MemoryEntryLike; score: number }>; method: "embedding" | "lexical" }>;
    forget(key: string): Promise<void>;
}

/**
 * Normalise a stored entry into a hit, resolving a compaction stub: with an archived
 * original the hit carries the FULL text (and the stored stub in `stub`); without one
 * it keeps the stub and is flagged `contentLost`.
 */
function toHit(e: MemoryEntryLike, archived: ArchivedFact | undefined, score?: number): RecallHit {
    const base: RecallHit = {
        key: e.key,
        content: e.content,
        ...(score === undefined ? {} : { score }),
        tags: e.tags,
        importance: e.importance,
        timestamp: e.timestamp,
    };
    if (!isStub(e.content)) return base;
    if (archived) return { ...base, content: archived.content, stub: e.content };
    return { ...base, contentLost: true };
}

export interface MemoryStoreBackendOptions {
    /** Where compacted originals are kept. Default: an in-process archive. */
    archive?: FactArchive;
    /** Recall ranking signals. Default {@link DEFAULT_RECALL_RANKING}; `false` = fusion alone. */
    ranking?: RecallRanking | false;
}

export class MemoryStoreBackend implements MemoryBackend {
    private readonly facts: FactArchive;
    private readonly ranking: RecallRanking | undefined;

    constructor(
        private readonly store: MemoryStoreLike,
        /** Optional SSMRuntime; enables embedding-based recall when present. */
        private readonly runtime?: unknown,
        opts: MemoryStoreBackendOptions = {},
    ) {
        this.facts = opts.archive ?? new MemoryFactArchive();
        this.ranking = opts.ranking === false ? undefined : { ...DEFAULT_RECALL_RANKING, ...opts.ranking };
    }

    /** Plain recall — {@link recallRanked} with the ranker's name dropped. ONE path. */
    async recall(query: string, topK: number): Promise<RecallHit[]> {
        return (await this.recallRanked(query, topK)).hits;
    }

    async recallRanked(query: string, topK: number): Promise<RankedRecall> {
        const ranked = await this.store.recallRanked(query, topK, this.runtime, this.ranking);
        const rows = this.facts.all();
        return { hits: ranked.hits.map((h) => toHit(h.entry, rows[h.entry.key], h.score)), method: ranked.method };
    }

    async get(key: string): Promise<RecallHit | undefined> {
        const e = await this.store.recall(key);
        return e ? toHit(e, this.facts.get(key)) : undefined;
    }

    async recallByTag(tag: string, limit: number): Promise<RecallHit[]> {
        const entries = await this.store.recallByTag(tag);
        const rows = this.facts.all();
        return entries.slice(0, limit).map((e) => toHit(e, rows[e.key]));
    }

    async archive(facts: ArchivedFact[]): Promise<void> {
        this.facts.putMany(facts);
    }

    /** Keys whose stored body is a stub with no archived original. */
    async lostKeys(): Promise<string[]> {
        const rows = this.facts.all();
        return (await this.store.recallAll()).filter((e) => isContentLost(e.content, rows[e.key])).map((e) => e.key);
    }

    /** Every archived row, read once — the snapshot writer's `contentLost` source. */
    archivedRows(): Readonly<Record<string, ArchivedFact>> {
        return this.facts.all();
    }

    async remember(input: RememberInput): Promise<void> {
        await this.store.remember(input.key, input.content, {
            ttlMs: input.ttlMs,
            tags: input.tags,
            importance: input.importance,
        });
    }

    /** The store has no batch write; sequencing here keeps ONE definition of a write. */
    async rememberMany(inputs: RememberInput[]): Promise<void> {
        for (const input of inputs) await this.remember(input);
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
     * Optional text embedder — anything with `embed(text): Promise<Float32Array>`.
     * Omit for lexical-only recall. Pass the agent-runtime's
     * `ssmMemoryService.runtime` to reuse an already-loaded GPU Evermind, or an
     * `EvermindTextEmbedder` (pure CPU) via {@link createEvermindEmbedder} on a
     * headless host.
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
     *
     * The file is also WATCHED: it is a shared artifact (other processes compact it
     * in place), so the store re-hydrates from disk whenever it changed underneath —
     * see {@link DiskPersistedBackend} for the loop guard.
     */
    persistFile?: string;
    /**
     * Hard cap on stored facts. Each write evicts the lowest-value entries
     * (expired first, then lowest importance, then oldest).
     *
     * This was left unset by the local backend, which passed only `dbName` to
     * `MemoryStore` — and `maxEntries` is undefined-by-default there, so the store
     * grew without bound: every `memory_remember` added a fact that was never
     * evicted, and the mirror file grew forever. Unbounded is fine for a
     * short-lived test, not for the long-lived MCP server this is the default
     * backend for. Defaults to {@link DEFAULT_MAX_ENTRIES} rather than requiring
     * every caller to remember to pass one.
     */
    maxEntries?: number;
    /**
     * Set false to keep recall lexical even when a model is configured. The local
     * store otherwise wires in {@link createRecallEmbedder}, which only resolves
     * when `BUILDERFORCE_MEMORY_MODEL` is set, so this is mostly for tests.
     */
    semanticRecall?: false;
    /**
     * Recall ranking signals (importance weighting, half-life recency decay).
     * Default {@link DEFAULT_RECALL_RANKING} — both on; `false` ranks by fusion alone.
     */
    ranking?: RecallRanking | false;
    /**
     * Where compacted originals are archived. Default: beside `persistFile`
     * (`memory.archive.json`) when there is one, else in-process.
     */
    archiveFile?: string;
    /** Diagnostic sink (stderr by default — stdout is the MCP stdio channel). */
    log?: (line: string) => void;
}

/**
 * Default cap on locally-stored facts. Large enough that a normal workspace never
 * notices eviction, small enough that the IndexedDB shim, the embedding cache and
 * the mirrored JSON file stay bounded on a host with no eviction policy of its own.
 */
export const DEFAULT_MAX_ENTRIES = 10_000;

/** The on-disk snapshot shape — a flat array of durable entries. */
interface SnapshotEntry {
    key: string;
    content: string;
    tags?: string[];
    importance?: number;
    /** Unix-ms write time — persisted so recency ranking survives a respawn. */
    timestamp?: number;
    /**
     * Written as true for a stub with no archived original, so dependency-free
     * readers (the Claude Code hook) can label it without reading the archive.
     * Derived on every write — never read back as truth; the archive decides.
     */
    contentLost?: true;
}

type FsLike = SharedFileFs & { mkdirSync(path: string, opts: { recursive: boolean }): void };
type PathLike = { dirname(p: string): string };

/** Parse a snapshot's text into its durable entries, or null when unusable. */
function parseSnapshot(text: string): SnapshotEntry[] | null {
    let parsed: unknown;
    try {
        parsed = JSON.parse(text);
    } catch {
        return null;
    }
    if (!Array.isArray(parsed)) return null;
    return (parsed as SnapshotEntry[]).filter(
        (raw) => !!raw && typeof raw.key === "string" && typeof raw.content === "string",
    );
}

/**
 * Wraps a MemoryStoreBackend so every write is mirrored to a JSON file, and
 * hydrates that file back into the store on boot. This is what turns a respawned
 * stdio subprocess into a persistent memory: the store itself is in-memory, the
 * file is the source of truth across process lifetimes.
 *
 * It also WATCHES the file for external edits. The snapshot is a shared artifact —
 * the BuilderForce VS Code extension compacts absorbed entries by rewriting it
 * directly, and a second server instance may be mirroring the same path. Without a
 * watch this server, still holding the pre-edit bodies, would silently re-snapshot
 * over that work on its very next remember/forget; the edit would only stick once
 * the client respawned the subprocess. So every operation first calls
 * {@link DiskPersistedBackend.ensureFresh}, which re-hydrates the store from disk
 * when — and only when — the file changed underneath us.
 *
 * The freshness check and the loop guard are {@link SharedJsonFile}'s — the one
 * implementation every shared local store (memory, experience) uses.
 */
class DiskPersistedBackend implements MemoryBackend {
    constructor(
        private readonly inner: MemoryStoreBackend,
        private readonly store: MemoryStoreLike,
        private readonly file: SharedJsonFile,
    ) {}

    async recall(query: string, topK: number): Promise<RecallHit[]> {
        await this.ensureFresh();
        return this.inner.recall(query, topK);
    }
    async recallRanked(query: string, topK: number): Promise<RankedRecall> {
        await this.ensureFresh();
        return this.inner.recallRanked(query, topK);
    }
    async get(key: string): Promise<RecallHit | undefined> {
        await this.ensureFresh();
        return this.inner.get(key);
    }
    async recallByTag(tag: string, limit: number): Promise<RecallHit[]> {
        await this.ensureFresh();
        return this.inner.recallByTag(tag, limit);
    }

    async remember(input: RememberInput): Promise<void> {
        await this.ensureFresh();
        await this.inner.remember(input);
        await this.snapshot();
    }

    /** Batch write — ONE snapshot for the whole set, so compacting N keys is not N full rewrites. */
    async rememberMany(inputs: RememberInput[]): Promise<void> {
        await this.ensureFresh();
        await this.inner.rememberMany(inputs);
        await this.snapshot();
    }

    async forget(key: string): Promise<void> {
        await this.ensureFresh();
        await this.inner.forget(key);
        await this.snapshot();
    }

    /**
     * Re-read the snapshot into the store when the file changed since we last
     * touched it. Also the BOOT hydration: with no stamp yet, an existing file is
     * always read in — one routine, so disk and memory can never diverge by taking
     * two different paths.
     */
    async ensureFresh(): Promise<void> {
        const changed = this.file.readIfChanged();
        if (!changed) return;
        const entries = parseSnapshot(changed.text);
        // Corrupt/partial snapshot (a half-written file, say): keep what we have
        // rather than wiping the live store, and leave the stamp alone so the next
        // call re-checks.
        if (!entries) return;
        await this.replaceStore(entries);
        changed.accept();
    }

    /**
     * Make the in-memory store equal the snapshot: drop keys the file no longer has,
     * then replay every entry (a re-remember overwrites, so a body shortened to a
     * stub on disk becomes the stub in memory). Replayed as durable — TTLs are
     * dropped on persist; the snapshot is the long-term tier.
     */
    private async replaceStore(entries: SnapshotEntry[]): Promise<void> {
        const wanted = new Set(entries.map((e) => e.key));
        for (const existing of await this.store.recallAll()) {
            if (!wanted.has(existing.key)) await this.store.forget(existing.key);
        }
        for (const raw of entries) {
            await this.store.remember(raw.key, raw.content, {
                tags: raw.tags,
                importance: raw.importance,
                ...(typeof raw.timestamp === "number" && Number.isFinite(raw.timestamp) ? { timestamp: raw.timestamp } : {}),
            });
        }
    }

    /**
     * Archive first — durable before the caller writes the stubs. When the rows
     * RESTORE lost stubs (memory_restore), the snapshot is rewritten too, so its
     * derived `contentLost` flags stop naming them.
     */
    async archive(facts: ArchivedFact[]): Promise<void> {
        await this.ensureFresh();
        const lostBefore = new Set(await this.inner.lostKeys());
        await this.inner.archive(facts);
        if (facts.some((f) => lostBefore.has(f.key))) await this.snapshot();
    }

    async lostKeys(): Promise<string[]> {
        await this.ensureFresh();
        return this.inner.lostKeys();
    }

    private async snapshot(): Promise<void> {
        const entries = await this.store.recallAll();
        const rows = this.inner.archivedRows();
        const out: SnapshotEntry[] = entries.map((e) => ({
            key: e.key,
            content: e.content,
            tags: e.tags,
            importance: e.importance,
            timestamp: e.timestamp,
            ...(isContentLost(e.content, rows[e.key]) ? { contentLost: true as const } : {}),
        }));
        this.file.write(JSON.stringify(out, null, 2));
    }
}

/**
 * Builds a MemoryStoreBackend over a fresh MemoryStore, wiring fake-indexeddb in
 * Node exactly as SsmMemoryService does. @seanhogg/builderforce-memory and fake-indexeddb
 * are imported indirectly so they remain optional peers — a consumer that only
 * wants a custom backend (or the HTTP thin-client) never has to install them.
 */
export async function createLocalMemoryStoreBackend(opts: LocalBackendOptions = {}): Promise<MemoryBackend> {
    const memoryMod = (await dynamicImport("@seanhogg/builderforce-memory")) as { MemoryStore: new (o: unknown) => MemoryStoreLike };
    const { MemoryStore } = memoryMod;

    // IndexedDB shim for Node. In the browser the global is used automatically.
    let idbFactory: unknown;
    try {
        const fake = (await dynamicImport("fake-indexeddb")) as { IDBFactory: new () => unknown };
        idbFactory = new fake.IDBFactory();
    } catch {
        // Browser or a host that provides global indexedDB — MemoryStore handles it.
    }

    const maxEntries = opts.maxEntries ?? DEFAULT_MAX_ENTRIES;

    // A local store with no embedder degrades to lexical-only recall, which is why
    // recall stopped finding paraphrases on a headless host. Wire the CPU Evermind
    // embedder in when the caller supplied no runtime and a model is configured.
    // `createRecallEmbedder` returns null when no model is set, or when the model
    // fails to stand up — both are correct degrades, not failures to boot.
    let runtime = opts.runtime;
    if (runtime === undefined && opts.semanticRecall !== false) {
        const embed = await import("../embedding/index.js");
        runtime =
            (await embed.createRecallEmbedder({
                memoryFile: opts.persistFile ?? opts.dbName ?? "",
            })) ?? undefined;
    }

    const store = new MemoryStore({ idbFactory, dbName: opts.dbName, maxEntries });
    const log = opts.log ?? ((line: string) => void globalThis.process?.stderr?.write(`${line}\n`));

    if (!opts.persistFile) return new MemoryStoreBackend(store, runtime, { ranking: opts.ranking });

    // Disk-mirror requested. node:fs/path are loaded indirectly so a browser
    // bundle of this module never statically pulls in Node builtins.
    const fs = (await dynamicImport("node:fs")) as FsLike;
    const path = (await dynamicImport("node:path")) as PathLike;
    const dir = path.dirname(opts.persistFile);
    if (dir && !fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    // The archive sits beside the snapshot, OUTSIDE it: it must survive the snapshot
    // being rebuilt, rewritten by another process, or capped — see fact-archive.ts.
    const archive = new FileFactArchive(opts.archiveFile ?? defaultArchiveFile(opts.persistFile), fs);
    const backend = new MemoryStoreBackend(store, runtime, { archive, ranking: opts.ranking });
    const persisted = new DiskPersistedBackend(backend, store, new SharedJsonFile(opts.persistFile, fs));
    // Boot hydration IS the freshness check with no stamp yet — see ensureFresh().
    await persisted.ensureFresh();

    // Facts compacted before archiving existed are stubs whose text is gone. Every
    // read flags them `contentLost` (and so does the next snapshot); say so once at
    // boot so the loss is visible, with the way back.
    const lost = await persisted.lostKeys();
    if (lost.length > 0) {
        log(
            `[builderforce-memory] ${lost.length} memory(ies) are compaction stubs with no archived original ` +
                `(content lost — re-verify before relying on them). Recover the text with memory_restore.`,
        );
    }
    return persisted;
}
