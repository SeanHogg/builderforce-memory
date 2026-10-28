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

    constructor(private readonly persistence: SnapshotPersistence, opts: LocalSnapshotStoreOptions = {}) {
        this.now = opts.now ?? (() => new Date().toISOString());
    }

    private async load(): Promise<KnowledgeSnapshot> {
        if (!this.snapshot) this.snapshot = parse(await this.persistence.read());
        return this.snapshot;
    }

    private mutate(fn: (s: KnowledgeSnapshot) => void): Promise<void> {
        const run = this.queue.then(async () => {
            const s = await this.load();
            fn(s);
            await this.persistence.write(serialize(s));
        });
        this.queue = run.catch(() => undefined);
        return run;
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
