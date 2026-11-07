/**
 * knowledge/stores/localSnapshotStore.ts — a {@link KnowledgeStore} over one JSON snapshot,
 * for a folder not linked to a project (PRD 38 §4.2, §5.1).
 *
 * Persistence is a port (`read`/`write` of the whole snapshot string), so this module has
 * no `fs`: the memory MCP host persists it beside its own snapshot through its shared JSON
 * file, a browser host in IndexedDB. Upserts use the platform's unique keys — a fact by
 * `key`, a relation by `(from, to, kind)`, a scope by `rootId` — so linking a folder later
 * uploads the same rows idempotently. The snapshot is written canonically (sorted), so
 * equal content is equal bytes.
 *
 * Every write serializes the WHOLE snapshot, so a many-mutation run (a directory review)
 * would cost quadratic bytes. {@link LocalSnapshotStore.batched} holds persistence back
 * inside a run; outside one, each awaited mutation has persisted when it resolves.
 *
 * PRD 41 A10: reads come from memory once loaded (never behind the mutation queue); rows live in
 * per-root maps updated incrementally; with the optional `extractedLog` port the extract cache is
 * an append-only JSON Lines file instead of part of the snapshot.
 */

import type {
    Extracted, KnowledgeStore, NodeRef, Scope, ScopeRecord, StoredFact, StoredRelation,
} from '../model.js';
import { canonicalJson, compareStr, pathOfRef } from '../refs.js';

export interface SnapshotPersistence {
    read(): Promise<string | undefined>;
    write(snapshot: string): Promise<void>;
    /** Keeps the bytes of a snapshot that did not parse (e.g. as `<snapshot>.corrupt`) before the store starts empty. Optional: without it they are lost. */
    keepAside?(text: string): Promise<void>;
}

/**
 * Append-only, content-addressed home of the extract cache (PRD 41 A10 rule 7). JSON Lines, one
 * `{"h":"<hash>","e":[...]}` per line; the LAST line for a hash wins, an unparseable (torn) line is ignored.
 */
export interface ExtractedLogPersistence {
    read(): Promise<string | undefined>;
    /** Appends `text` (whole lines, newline-terminated) to the end of the log. */
    append(text: string): Promise<void>;
    /** Replaces the whole log with `text` (atomically where the host can). Optional: without it the log is never compacted. */
    replace?(text: string): Promise<void>;
}

export interface KnowledgeSnapshot {
    version: 1;
    scopes: ScopeRecord[];
    facts: Record<string, StoredFact[]>;
    relations: Record<string, StoredRelation[]>;
    /** Present only when the store has no `extractedLog` (or in a snapshot written before the log existed). */
    extracted?: Record<string, Extracted[]>;
}

export interface LocalSnapshotStoreOptions {
    /** When given, the extract cache lives in this log and the snapshot omits `extracted`. */
    extractedLog?: ExtractedLogPersistence;
    /** ISO clock for `staleAt`. Default `new Date().toISOString()`. */
    now?(): string;
    /** Inside `batched`, a mutation persists when this many ms passed since the last write (a crash loses at most this much). Default 30_000. */
    checkpointMs?: number;
    /** Millisecond clock for the checkpoint. Default `Date.now`. */
    nowMs?(): number;
}

const relKey = (r: Pick<StoredRelation, 'from' | 'to' | 'kind'>) => `${r.from}\u0000${r.to}\u0000${r.kind}`;

function asRecord<T>(v: unknown): Record<string, T[]> {
    return v && typeof v === 'object' && !Array.isArray(v) ? v as Record<string, T[]> : {};
}

const emptySnapshot = (): KnowledgeSnapshot => ({ version: 1, scopes: [], facts: {}, relations: {} });

/** Reads a snapshot's text. THROWS when it does not parse, so the caller can keep the bytes aside before starting empty. */
function parse(text: string | undefined): KnowledgeSnapshot {
    if (!text) return emptySnapshot();
    const s = JSON.parse(text) as Partial<KnowledgeSnapshot>;
    return {
        version: 1,
        scopes: Array.isArray(s.scopes) ? s.scopes : [],
        facts: asRecord(s.facts),
        relations: asRecord(s.relations),
        ...(s.extracted && typeof s.extracted === 'object' ? { extracted: s.extracted } : {}),
    };
}

interface ParsedLog {
    entries: Map<string, Extracted[]>;
    /** The live (last) line of each hash, newline-terminated, in log order. */
    liveLines: Map<string, string>;
    liveBytes: number;
    /** Superseded lines, torn lines and blank lines. */
    deadBytes: number;
}

/**
 * The log's entries, last line per hash winning. A line that does not parse (torn by a crash or a failed
 * append) is skipped on its own: it never costs the other lines.
 */
function parseLog(text: string | undefined): ParsedLog {
    const out: ParsedLog = { entries: new Map(), liveLines: new Map(), liveBytes: 0, deadBytes: 0 };
    if (!text) return out;
    for (const line of text.split('\n')) {
        if (!line.trim()) continue;
        try {
            const row = JSON.parse(line) as { h?: unknown; e?: unknown };
            if (typeof row.h === 'string' && Array.isArray(row.e)) {
                out.entries.set(row.h, row.e as Extracted[]);
                out.liveLines.delete(row.h); // re-set below so the map stays in log order of last occurrence
                out.liveLines.set(row.h, `${line}\n`);
            }
        } catch { /* torn line */ }
    }
    for (const l of out.liveLines.values()) out.liveBytes += l.length;
    out.deadBytes = Math.max(0, text.length - out.liveBytes);
    return out;
}

type Index = Map<string, Set<string>>;

export class LocalSnapshotStore implements KnowledgeStore {
    private loaded = false;
    private loading: Promise<void> | undefined;
    private scopeList: ScopeRecord[] = [];
    /** Per-root rows, keyed by their unique key, updated incrementally (PRD 41 A10 rule 7). */
    private readonly factMaps = new Map<string, Map<string, StoredFact>>();
    private readonly relMaps = new Map<string, Map<string, StoredRelation>>();
    /** Per root: source path → keys of the relations touching it (`from` or `to`). */
    private readonly relByPath = new Map<string, Index>();
    private readonly extracted = new Map<string, Extracted[]>();
    /** Extract entries not yet appended to the log. */
    private pending = new Map<string, Extracted[]>();
    /** The log's last line is torn (no newline), so the next append starts on a fresh line. */
    private logTorn = false;
    private readonly log: ExtractedLogPersistence | undefined;
    /** Serializes mutations so two writers never interleave a read-modify-write. */
    private queue: Promise<unknown> = Promise.resolve();
    private readonly now: () => string;
    private readonly nowMs: () => number;
    private readonly checkpointMs: number;
    /** Open `batched` calls; persistence is held while it is above zero. */
    private depth = 0;
    /** Memory is ahead of the persisted snapshot. */
    private dirty = false;
    private persistedAt = 0;
    /** Serializes snapshot writes; never rejects. */
    private persisting: Promise<void> = Promise.resolve();
    /** A background checkpoint in flight (at most one). */
    private checkpointing: Promise<void> | undefined;

    constructor(private readonly persistence: SnapshotPersistence, opts: LocalSnapshotStoreOptions = {}) {
        this.now = opts.now ?? (() => new Date().toISOString());
        this.nowMs = opts.nowMs ?? Date.now;
        this.checkpointMs = opts.checkpointMs ?? 30_000;
        this.log = opts.extractedLog;
    }

    /** Loads once; every later call is a no-op, so reads never wait on the mutation queue after the first. */
    private load(): Promise<void> {
        if (this.loaded) return Promise.resolve();
        this.loading ??= this.doLoad().catch((err) => { this.loading = undefined; throw err; });
        return this.loading;
    }

    private async doLoad(): Promise<void> {
        const snapText = await this.persistence.read();
        let snap: KnowledgeSnapshot;
        try {
            snap = parse(snapText);
        } catch {
            // Unreadable: never silently replaced. Its bytes are kept aside, then the store starts empty (the next review rebuilds).
            await this.persistence.keepAside?.(snapText!).catch(() => undefined);
            snap = emptySnapshot();
        }
        const logText = this.log ? await this.log.read() : undefined;
        this.scopeList = snap.scopes;
        for (const [rootId, rows] of Object.entries(snap.facts)) {
            const m = this.factsOf(rootId);
            if (Array.isArray(rows)) for (const f of rows) m.set(f.key, f);
        }
        for (const [rootId, rows] of Object.entries(snap.relations)) {
            this.relsOf(rootId);
            if (Array.isArray(rows)) for (const r of rows) this.putRelation(rootId, r);
        }
        if (this.log) {
            const parsed = parseLog(logText);
            for (const [h, e] of parsed.entries) this.extracted.set(h, e);
            this.logTorn = !!logText && !logText.endsWith('\n');
            await this.compactLog(parsed);
            // An old snapshot still carrying the cache: move it into the log once; the next snapshot write omits it.
            const legacy = Object.entries(snap.extracted ?? {});
            if (legacy.length) {
                for (const [h, e] of legacy) {
                    if (this.extracted.has(h)) continue;
                    this.extracted.set(h, e);
                    this.pending.set(h, e);
                }
                this.dirty = true;
            }
        } else {
            for (const [h, e] of Object.entries(snap.extracted ?? {})) this.extracted.set(h, e);
        }
        this.loaded = true;
    }

    /**
     * Rewrites the log to its live lines when dead bytes (superseded and torn lines) exceed live bytes. Needs the
     * port's `replace`; a failure is ignored (the log stays correct, only larger). Another window appending between
     * this read and the replace loses those lines from the cache — a cache miss, re-extracted next review.
     */
    private async compactLog(parsed: ParsedLog): Promise<void> {
        if (!this.log?.replace || parsed.deadBytes <= parsed.liveBytes) return;
        try {
            await this.log.replace([...parsed.liveLines.values()].join(''));
            this.logTorn = false;
        } catch { /* keep the uncompacted log */ }
    }

    private factsOf(rootId: string): Map<string, StoredFact> {
        let m = this.factMaps.get(rootId);
        if (!m) this.factMaps.set(rootId, m = new Map());
        return m;
    }

    private relsOf(rootId: string): Map<string, StoredRelation> {
        let m = this.relMaps.get(rootId);
        if (!m) this.relMaps.set(rootId, m = new Map());
        return m;
    }

    /** Upserts a relation and indexes it under the source paths of both ends. */
    private putRelation(rootId: string, r: StoredRelation): void {
        const key = relKey(r);
        this.relsOf(rootId).set(key, r);
        let idx = this.relByPath.get(rootId);
        if (!idx) this.relByPath.set(rootId, idx = new Map());
        for (const ref of [r.from, r.to]) {
            const p = pathOfRef(ref);
            if (p === undefined) continue;
            let keys = idx.get(p);
            if (!keys) idx.set(p, keys = new Set());
            keys.add(key);
        }
    }

    private enqueue<T>(task: () => Promise<T>): Promise<T> {
        const run = this.queue.then(task);
        this.queue = run.catch(() => undefined);
        return run;
    }

    /**
     * Appends the not-yet-logged extract entries as ONE `append` of complete lines. A failed append keeps them
     * pending and marks the log torn (it may hold a partial line), so the next append starts on a fresh line.
     */
    private async appendPending(): Promise<void> {
        if (!this.log || this.pending.size === 0) return;
        const batch = this.pending;
        this.pending = new Map();
        let text = this.logTorn ? '\n' : '';
        for (const [h, e] of batch) text += `${JSON.stringify({ h, e })}\n`;
        try {
            await this.log.append(text);
            this.logTorn = false;
        } catch (err) {
            this.logTorn = true;
            for (const [h, e] of batch) if (!this.pending.has(h)) this.pending.set(h, e);
            throw err;
        }
    }

    /**
     * Appends new extract entries, then writes the snapshot if memory is ahead of it, one write at a time. It
     * runs OUTSIDE the mutation queue: a write of a large snapshot (serialize + disk) must never hold up reads or
     * the next mutation's in-memory apply (PRD 41 A10 — it did: every `getExtracted` waited out each checkpoint).
     * The log goes first, so a snapshot that dropped a migrated cache never outlives its entries. A failed
     * write leaves the snapshot dirty, so the next one retries it.
     */
    private persistIfDirty(): Promise<void> {
        const run = this.persisting.then(async () => {
            if (!this.loaded) return;
            await this.appendPending();
            if (!this.dirty) return;
            this.dirty = false;
            try {
                await this.persistence.write(this.serialize());
            } catch (err) {
                this.dirty = true;
                throw err;
            }
            this.persistedAt = this.nowMs();
        });
        this.persisting = run.catch(() => undefined);
        return run;
    }

    /**
     * Applies in memory (queued, so reads and writes never interleave), then persists: at once outside a
     * batch, else a checkpoint in the BACKGROUND once `checkpointMs` passed. A plain mutation from another
     * caller during a batch (the live file watcher shares this store) is held like the batch's own and made
     * durable by the next checkpoint, flush or batch end. `snapshotChanged: false` is for a change that only
     * touches the extract log.
     */
    private async mutate(fn: () => void, snapshotChanged = true): Promise<void> {
        await this.enqueue(async () => {
            await this.load();
            fn();
            if (snapshotChanged) this.dirty = true;
        });
        if (this.depth === 0) await this.persistIfDirty();
        else if (!this.checkpointing && this.nowMs() - this.persistedAt >= this.checkpointMs) {
            // A failed checkpoint is not lost: the snapshot stays dirty and the batch's end or flush writes (and reports) it.
            // It backs off by `checkpointMs` (advances `persistedAt`) so a failing disk is not retried on every mutation.
            this.checkpointing = this.persistIfDirty().catch(() => { this.persistedAt = this.nowMs(); }).finally(() => { this.checkpointing = undefined; });
        }
    }

    /**
     * Holds persistence while `work` runs. `flush` persists once if anything is unwritten. When `work`
     * settles the outermost batch persists what is left: `work`'s own error wins over a failed final
     * write, a failed final write fails a successful `work`. Batches may nest or overlap.
     */
    async batched<T>(work: (flush: () => Promise<void>) => Promise<T>): Promise<T> {
        if (this.depth === 0) this.persistedAt = this.nowMs();
        this.depth += 1;
        // The end goes through the queue so every mutation made during the batch is held, then written once.
        const end = async () => {
            await this.enqueue(async () => { this.depth -= 1; });
            if (this.depth === 0) await this.persistIfDirty();
        };
        let result: T;
        try {
            result = await work(async () => { await this.queue; await this.persistIfDirty(); });
        } catch (err) {
            await end().catch(() => undefined);
            throw err;
        }
        await end();
        return result;
    }

    async scopes(): Promise<ScopeRecord[]> {
        await this.load();
        return [...this.scopeList];
    }

    async facts(scope: Scope): Promise<StoredFact[]> {
        await this.load();
        return [...(this.factMaps.get(scope.rootId)?.values() ?? [])];
    }

    async relations(scope: Scope): Promise<StoredRelation[]> {
        await this.load();
        return [...(this.relMaps.get(scope.rootId)?.values() ?? [])];
    }

    write(batch: { scope: ScopeRecord; facts: StoredFact[]; relations: StoredRelation[] }): Promise<void> {
        const bad = [...batch.facts, ...batch.relations].find((x) => (x.origin as string) === 'extracted');
        if (bad) return Promise.reject(new Error('knowledge: extracted output is never stored'));
        return this.mutate(() => {
            const rootId = batch.scope.rootId;
            const prior = this.scopeList.find((x) => x.rootId === rootId);
            const merged: ScopeRecord = { ...(prior ?? {}), ...stripUndefined(batch.scope) } as ScopeRecord;
            this.scopeList = [...this.scopeList.filter((x) => x.rootId !== rootId), merged];

            const facts = this.factsOf(rootId);
            for (const f of batch.facts) facts.set(f.key, { ...f });
            this.relsOf(rootId);
            for (const r of batch.relations) this.putRelation(rootId, { ...r });
        });
    }

    markStale(scope: Scope, sourceRef: NodeRef, currentHash: string): Promise<void> {
        const path = pathOfRef(sourceRef);
        return this.mutate(() => {
            const at = this.now();
            const rels = this.relsOf(scope.rootId);
            const keys = path === undefined ? undefined : this.relByPath.get(scope.rootId)?.get(path);
            for (const key of keys ?? []) {
                const r = rels.get(key);
                if (r && r.origin === 'model' && !r.staleAt && r.sourceHash !== currentHash) rels.set(key, { ...r, staleAt: at });
            }
        });
    }

    async getExtracted(hash: string): Promise<Extracted[] | undefined> {
        await this.load();
        return this.extracted.get(hash);
    }

    /** Every cached entry by content hash (the log's, else the snapshot's `extracted`), for a host that packs the cache. */
    async extractedEntries(): Promise<Record<string, Extracted[]>> {
        await this.load();
        return Object.fromEntries(this.extracted);
    }

    putExtracted(hash: string, extracted: Extracted[]): Promise<void> {
        return this.mutate(() => {
            // The same entry again (a second pass re-puts what the first stored) is skipped: no append, no dirty snapshot.
            const prior = this.extracted.get(hash);
            if (prior && canonicalJson(prior) === canonicalJson(extracted)) return;
            this.extracted.set(hash, extracted);
            if (this.log) this.pending.set(hash, extracted);
            else this.dirty = true;
        }, false);
    }

    /** Canonical: scopes by rootId, facts by key, relations by (from, to, kind), keys sorted. */
    private serialize(): string {
        const rows = <T>(maps: Map<string, Map<string, T>>, by: (a: T, b: T) => number) =>
            Object.fromEntries([...maps.keys()].sort(compareStr).map((k) => [k, [...maps.get(k)!.values()].sort(by)]));
        return canonicalJson({
            version: 1,
            scopes: [...this.scopeList].sort((a, b) => compareStr(a.rootId, b.rootId)),
            facts: rows(this.factMaps, (a, b) => compareStr(a.key, b.key)),
            relations: rows(this.relMaps, (a, b) => compareStr(relKey(a), relKey(b))),
            ...(this.log ? {} : { extracted: Object.fromEntries(this.extracted) }),
        });
    }
}

export function localSnapshotStore(persistence: SnapshotPersistence, opts?: LocalSnapshotStoreOptions): LocalSnapshotStore {
    return new LocalSnapshotStore(persistence, opts);
}

function stripUndefined<T extends object>(o: T): Partial<T> {
    return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as Partial<T>;
}
