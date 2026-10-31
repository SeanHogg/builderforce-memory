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
 */

import type {
    Extracted, KnowledgeStore, NodeRef, Scope, ScopeRecord, StoredFact, StoredRelation,
} from '../model.js';
import { canonicalJson, compareStr, pathOfRef } from '../refs.js';

export interface SnapshotPersistence {
    read(): Promise<string | undefined>;
    write(snapshot: string): Promise<void>;
}

export interface KnowledgeSnapshot {
    version: 1;
    scopes: ScopeRecord[];
    facts: Record<string, StoredFact[]>;
    relations: Record<string, StoredRelation[]>;
    extracted: Record<string, Extracted[]>;
}

export interface LocalSnapshotStoreOptions {
    /** ISO clock for `staleAt`. Default `new Date().toISOString()`. */
    now?(): string;
    /** Inside `batched`, a mutation persists when this many ms passed since the last write (a crash loses at most this much). Default 30_000. */
    checkpointMs?: number;
    /** Millisecond clock for the checkpoint. Default `Date.now`. */
    nowMs?(): number;
}

const relKey = (r: Pick<StoredRelation, 'from' | 'to' | 'kind'>) => `${r.from}\u0000${r.to}\u0000${r.kind}`;

function empty(): KnowledgeSnapshot {
    return { version: 1, scopes: [], facts: {}, relations: {}, extracted: {} };
}

function parse(text: string | undefined): KnowledgeSnapshot {
    if (!text) return empty();
    try {
        const s = JSON.parse(text) as Partial<KnowledgeSnapshot>;
        return {
            version: 1,
            scopes: Array.isArray(s.scopes) ? s.scopes : [],
            facts: s.facts && typeof s.facts === 'object' ? s.facts : {},
            relations: s.relations && typeof s.relations === 'object' ? s.relations : {},
            extracted: s.extracted && typeof s.extracted === 'object' ? s.extracted : {},
        };
    } catch {
        // A corrupt snapshot is rebuilt by the next review rather than failing every read.
        return empty();
    }
}

export class LocalSnapshotStore implements KnowledgeStore {
    private snapshot: KnowledgeSnapshot | undefined;
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

    constructor(private readonly persistence: SnapshotPersistence, opts: LocalSnapshotStoreOptions = {}) {
        this.now = opts.now ?? (() => new Date().toISOString());
        this.nowMs = opts.nowMs ?? Date.now;
        this.checkpointMs = opts.checkpointMs ?? 30_000;
    }

    private async load(): Promise<KnowledgeSnapshot> {
        if (!this.snapshot) this.snapshot = parse(await this.persistence.read());
        return this.snapshot;
    }

    private enqueue<T>(task: () => Promise<T>): Promise<T> {
        const run = this.queue.then(task);
        this.queue = run.catch(() => undefined);
        return run;
    }

    private async persist(s: KnowledgeSnapshot): Promise<void> {
        await this.persistence.write(serialize(s));
        this.dirty = false;
        this.persistedAt = this.nowMs();
    }

    private async persistIfDirty(): Promise<void> {
        if (this.dirty && this.snapshot) await this.persist(this.snapshot);
    }

    /**
     * Applies in memory, then persists: at once outside a batch, else only at a checkpoint. A plain
     * mutation from another caller during a batch (the live file watcher shares this store) is held
     * like the batch's own and made durable by the next checkpoint, flush or batch end.
     */
    private mutate(fn: (s: KnowledgeSnapshot) => void): Promise<void> {
        return this.enqueue(async () => {
            const s = await this.load();
            fn(s);
            this.dirty = true;
            if (this.depth === 0 || this.nowMs() - this.persistedAt >= this.checkpointMs) await this.persist(s);
        });
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
        const end = () => this.enqueue(async () => {
            this.depth -= 1;
            if (this.depth === 0) await this.persistIfDirty();
        });
        let result: T;
        try {
            result = await work(() => this.enqueue(() => this.persistIfDirty()));
        } catch (err) {
            await end().catch(() => undefined);
            throw err;
        }
        await end();
        return result;
    }

    async scopes(): Promise<ScopeRecord[]> {
        await this.queue;
        return [...(await this.load()).scopes];
    }

    async facts(scope: Scope): Promise<StoredFact[]> {
        await this.queue;
        return [...((await this.load()).facts[scope.rootId] ?? [])];
    }

    async relations(scope: Scope): Promise<StoredRelation[]> {
        await this.queue;
        return [...((await this.load()).relations[scope.rootId] ?? [])];
    }

    write(batch: { scope: ScopeRecord; facts: StoredFact[]; relations: StoredRelation[] }): Promise<void> {
        const bad = [...batch.facts, ...batch.relations].find((x) => (x.origin as string) === 'extracted');
        if (bad) return Promise.reject(new Error('knowledge: extracted output is never stored'));
        return this.mutate((s) => {
            const rootId = batch.scope.rootId;
            const prior = s.scopes.find((x) => x.rootId === rootId);
            const merged: ScopeRecord = { ...(prior ?? {}), ...stripUndefined(batch.scope) } as ScopeRecord;
            s.scopes = [...s.scopes.filter((x) => x.rootId !== rootId), merged];

            const facts = new Map((s.facts[rootId] ?? []).map((f) => [f.key, f] as const));
            for (const f of batch.facts) facts.set(f.key, { ...f });
            s.facts[rootId] = [...facts.values()];

            const relations = new Map((s.relations[rootId] ?? []).map((r) => [relKey(r), r] as const));
            for (const r of batch.relations) relations.set(relKey(r), { ...r });
            s.relations[rootId] = [...relations.values()];
        });
    }

    markStale(scope: Scope, sourceRef: NodeRef, currentHash: string): Promise<void> {
        const path = pathOfRef(sourceRef);
        return this.mutate((s) => {
            const at = this.now();
            const onSource = (ref: NodeRef) => path !== undefined && pathOfRef(ref) === path;
            s.relations[scope.rootId] = (s.relations[scope.rootId] ?? []).map((r) =>
                r.origin === 'model' && !r.staleAt && r.sourceHash !== currentHash && (onSource(r.from) || onSource(r.to))
                    ? { ...r, staleAt: at }
                    : r);
        });
    }

    async getExtracted(hash: string): Promise<Extracted[] | undefined> {
        await this.queue;
        return (await this.load()).extracted[hash];
    }

    putExtracted(hash: string, extracted: Extracted[]): Promise<void> {
        return this.mutate((s) => { s.extracted[hash] = extracted; });
    }
}

export function localSnapshotStore(persistence: SnapshotPersistence, opts?: LocalSnapshotStoreOptions): LocalSnapshotStore {
    return new LocalSnapshotStore(persistence, opts);
}

function stripUndefined<T extends object>(o: T): Partial<T> {
    return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as Partial<T>;
}

/** Canonical: scopes by rootId, facts by key, relations by (from, to, kind), keys sorted. */
function serialize(s: KnowledgeSnapshot): string {
    const sortRecord = <T>(rec: Record<string, T[]>, by: (a: T, b: T) => number) =>
        Object.fromEntries(Object.keys(rec).sort(compareStr).map((k) => [k, [...rec[k]!].sort(by)]));
    return canonicalJson({
        version: 1,
        scopes: [...s.scopes].sort((a, b) => compareStr(a.rootId, b.rootId)),
        facts: sortRecord(s.facts, (a, b) => compareStr(a.key, b.key)),
        relations: sortRecord(s.relations, (a, b) => compareStr(relKey(a), relKey(b))),
        extracted: s.extracted,
    });
}
