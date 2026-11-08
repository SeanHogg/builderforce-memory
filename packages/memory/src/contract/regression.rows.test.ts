/**
 * src/contract/regression.rows.test.ts — the row-persisted LocalSnapshotStore (PRD 41 B4, G03), each invariant pinned by one
 * test named `regression: <what broke>`. The host here is a fake `RowPersistence`: a map-backed "database" that records every
 * change it is given and can be held, failed, told another writer changed it, or made to return `reloaded`.
 */

import type { Extracted, NodeRef, ScopeRecord, StoredFact, StoredRelation } from '../knowledge/model.js';
import { canonicalDump, LocalSnapshotStore, type KnowledgeSnapshot, type RowChange, type RowPersistence } from '../knowledge/stores/localSnapshotStore.js';
import { memPersistence } from '../knowledge/testkit.test-util.js';

const SCOPE: ScopeRecord = { rootId: 'a', title: 'A', packs: ['docs'] };
const fact = (key: string, content = 'c'): StoredFact => ({ key, content, source: 's', importance: 0.5, origin: 'model', fileAnchors: [] });
const rel = (from: string, to: string, kind: string): StoredRelation => ({ from: from as NodeRef, to: to as NodeRef, kind, weight: 1, origin: 'model', sourceHash: 'h1' });
const ex = (name: string) => [{ name }] as unknown as Extracted[];
const deferred = () => { let resolve!: () => void; const promise = new Promise<void>((r) => { resolve = r; }); return { promise, resolve }; };
const settle = async () => { for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r)); };

interface Db {
    scopes: Map<string, ScopeRecord>;
    facts: Map<string, StoredFact>;
    relations: Map<string, StoredRelation>;
    extracted: Map<string, Extracted[]>;
    recheck: Set<string>;
}

interface FakeRows extends RowPersistence {
    db: Db;
    applies: RowChange[];
    generation: number;
    loads: number;
    checks: number;
    bodyReads: string[];
    /** Holds every `apply` (after recording it) until resolved. */
    hold?: ReturnType<typeof deferred>;
    failNext: number;
    /** Another writer's change: run after an apply merged, its result is returned as `reloaded`. */
    reload?: () => KnowledgeSnapshot;
    /** `changedElsewhere` answers true once. */
    changed: boolean;
    /** `load` leaves extract bodies out (they load lazily). */
    lazyBodies: boolean;
}

const fkey = (root: string, key: string) => `${root}\u0000${key}`;
const rkey = (root: string, from: string, to: string, kind: string) => `${root}\u0000${from}\u0000${to}\u0000${kind}`;

function snapshotOf(db: Db, withExtracted = true): KnowledgeSnapshot {
    const facts: Record<string, StoredFact[]> = {};
    const relations: Record<string, StoredRelation[]> = {};
    for (const [k, f] of db.facts) (facts[k.split('\u0000')[0]!] ??= []).push(f);
    for (const [k, r] of db.relations) (relations[k.split('\u0000')[0]!] ??= []).push(r);
    return {
        version: 1, scopes: [...db.scopes.values()], facts, relations,
        ...(withExtracted ? { extracted: Object.fromEntries(db.extracted) } : {}),
        ...(db.recheck.size ? { recheck: [...db.recheck] } : {}),
    };
}

function fakeRows(): FakeRows {
    const db: Db = { scopes: new Map(), facts: new Map(), relations: new Map(), extracted: new Map(), recheck: new Set() };
    const f: FakeRows = {
        db, applies: [], generation: 0, loads: 0, checks: 0, bodyReads: [], failNext: 0, changed: false, lazyBodies: false,
        async load() { f.loads += 1; return snapshotOf(db, !f.lazyBodies); },
        async apply(change) {
            f.applies.push(change);
            if (f.hold) await f.hold.promise;
            if (f.failNext > 0) { f.failNext -= 1; throw new Error('disk full'); }
            const { upserts: u, deletes: d } = change;
            for (const s of u.scopes) db.scopes.set(s.rootId, s);
            for (const [root, x] of u.facts) db.facts.set(fkey(root, x.key), x);
            for (const [root, r] of u.relations) db.relations.set(rkey(root, r.from, r.to, r.kind), r);
            for (const [h, e] of u.extracted) db.extracted.set(h, e);
            for (const h of u.recheck) db.recheck.add(h);
            for (const id of d.scopes) db.scopes.delete(id);
            for (const [root, key] of d.facts) db.facts.delete(fkey(root, key));
            for (const [root, from, to, kind] of d.relations) db.relations.delete(rkey(root, from, to, kind));
            for (const h of d.extracted) db.extracted.delete(h);
            for (const h of d.recheck) db.recheck.delete(h);
            f.generation += 1;
            const reload = f.reload;
            f.reload = undefined;
            return { generation: f.generation, ...(reload ? { reloaded: reload() } : {}) };
        },
        async loadAllExtracted() { return Object.fromEntries(db.extracted); },
        async loadExtracted(hash) { f.bodyReads.push(hash); return db.extracted.get(hash); },
        async changedElsewhere() { f.checks += 1; const c = f.changed; f.changed = false; return c; },
    };
    return f;
}


const rowStore = (rows: RowPersistence) => new LocalSnapshotStore(undefined, { rows, now: () => 'T' });
const loadedFrom = (snap: KnowledgeSnapshot) => {
    const p = memPersistence();
    p.value = JSON.stringify(snap);
    return new LocalSnapshotStore(p, { now: () => 'T' });
};

function prng(seed: number): () => number {
    return () => {
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** The seeded op sequence both kinds of store run: facts, relations, scopes, extracted, recheck (with clears), stale marks, reads, some inside `batched`. */
async function driveOps(store: LocalSnapshotStore, seed: number, steps: number, between?: (rand: () => number, pick: (n: number) => number) => void): Promise<void> {
    const rand = prng(seed);
    const pick = (n: number) => Math.floor(rand() * n);
    const op = async () => {
        const kind = pick(8);
        if (kind <= 2) {
            await store.write({
                scope: { ...SCOPE, ...(rand() < 0.5 ? { lastDigest: `d${pick(9)}` } : {}) },
                facts: Array.from({ length: pick(4) }, () => fact(`claim:k${pick(6)}`, `v${pick(99)}`)),
                relations: Array.from({ length: pick(3) }, () => rel(`fact:claim:k${pick(6)}`, `src:a::f${pick(4)}.md#s`, ['about', 'mentions'][pick(2)]!)),
            });
        } else if (kind === 3) await store.markStale({ rootId: 'a' }, `src:a::f${pick(4)}.md#s` as NodeRef, `h${pick(3)}`);
        else if (kind === 4) await store.putExtracted(`x${pick(8)}`, ex(`e${pick(5)}`));
        else if (kind === 5) await store.markRecheck([`h${pick(5)}`, `h${pick(5)}`]);
        else if (kind === 6) await store.clearRecheck([`h${pick(5)}`]);
        else await store.facts({ rootId: 'a' });
    };
    for (let i = 0; i < steps; i++) {
        between?.(rand, pick);
        try {
            if (rand() < 0.3) await store.batched(async () => { for (let j = 0; j < 5; j++) await op(); });
            else await op();
        } catch (err) {
            if (!between) throw err; // only the fuzz injects failures
        }
    }
}

const sameAsHost = async (store: LocalSnapshotStore, rows: FakeRows) => expect(await canonicalDump(store)).toBe(await canonicalDump(loadedFrom(snapshotOf(rows.db))));

describe('regression: row persistence', () => {
    it('regression: after N random mutations the rows the host accumulated equal the canonical dump', async () => {
        const rows = fakeRows();
        const store = rowStore(rows);
        await driveOps(store, 20261108, 90);
        const dump = await canonicalDump(store);
        expect(dump).toContain('"facts"');
        expect(dump).toContain('"recheck"');
        await sameAsHost(store, rows); // the accumulated rows
        expect(await canonicalDump(rowStore(rows))).toBe(dump); // and a second store loading them
    });

    it('regression: parity oracle: the same op sequence on a snapshot-persisted store and a row store gives byte-equal dumps', async () => {
        for (const seed of [1, 2, 3]) {
            const classic = new LocalSnapshotStore(memPersistence(), { now: () => 'T' });
            const rowed = rowStore(fakeRows());
            await driveOps(classic, seed, 60);
            await driveOps(rowed, seed, 60);
            expect(await canonicalDump(rowed)).toBe(await canonicalDump(classic));
        }
    });

    it('regression: fuzz: failed applies, `reloaded` and another writer changing the store, interleaved with the ops, never lose or reorder this window\'s rows', async () => {
        for (const seed of [11, 12, 13, 14]) {
            const rows = fakeRows();
            const store = rowStore(rows);
            let other = 0;
            const theirs = () => { const key = `claim:o${other++}`; rows.db.facts.set(fkey('b', key), fact(key)); };
            await driveOps(store, seed, 80, (rand) => {
                const r = rand();
                if (r < 0.15) rows.failNext = 1;
                else if (r < 0.3) rows.reload = () => { theirs(); return snapshotOf(rows.db); };
                else if (r < 0.45) { theirs(); rows.changed = true; }
            });
            rows.failNext = 0;
            rows.reload = undefined;
            await store.write({ scope: SCOPE, facts: [], relations: [] }); // flush whatever a failure left marked
            rows.changed = true;
            await store.scopes(); // pick up the other writer's last change
            await sameAsHost(store, rows);
        }
    });

    it('regression: a changed-elsewhere read that races an apply cannot drop the rows that apply commits', async () => {
        const rows = fakeRows();
        const store = rowStore(rows);
        await store.facts({ rootId: 'a' });
        rows.hold = deferred();
        const write = store.write({ scope: SCOPE, facts: [fact('claim:mine')], relations: [] });
        await settle();
        rows.db.facts.set(fkey('b', 'claim:theirs'), fact('claim:theirs'));
        rows.changed = true;
        const read = store.facts({ rootId: 'b' });
        await settle();
        rows.hold.resolve();
        await Promise.all([write, read]);
        await store.scopes();
        expect((await store.facts({ rootId: 'a' })).map((x) => x.key)).toEqual(['claim:mine']);
        expect((await store.facts({ rootId: 'b' })).map((x) => x.key)).toEqual(['claim:theirs']);
        await sameAsHost(store, rows);
    });

    it('regression: a read during a held apply resolves without waiting for it', async () => {
        const rows = fakeRows();
        const store = rowStore(rows);
        await store.facts({ rootId: 'a' });
        rows.checks = 0;
        rows.hold = deferred();
        const write = store.write({ scope: SCOPE, facts: [fact('claim:1')], relations: [] });
        await settle();
        rows.changed = true;
        expect((await store.facts({ rootId: 'a' })).map((x) => x.key)).toEqual(['claim:1']); // resolves while the apply is held
        expect(rows.checks).toBe(0); // no check was made behind the apply
        rows.hold.resolve();
        await write;
    });

    it('regression: a reload arriving while a failed apply is restored keeps the restored rows, and the next apply persists them', async () => {
        const rows = fakeRows();
        const store = rowStore(rows);
        await store.facts({ rootId: 'a' });
        rows.hold = deferred();
        rows.failNext = 1;
        const write = store.write({ scope: SCOPE, facts: [fact('claim:mine')], relations: [] });
        await settle();
        rows.db.facts.set(fkey('b', 'claim:theirs'), fact('claim:theirs'));
        rows.changed = true;
        const read = store.facts({ rootId: 'b' });
        expect(await read).toEqual([]); // answered from memory while the apply was in flight
        rows.hold.resolve();
        await expect(write).rejects.toThrow('disk full');
        expect((await store.facts({ rootId: 'b' })).map((x) => x.key)).toEqual(['claim:theirs']); // the next read reloads
        expect((await store.facts({ rootId: 'a' })).map((x) => x.key)).toEqual(['claim:mine']); // restored marks survived the reload
        await store.markRecheck(['x']);
        expect(rows.db.facts.has(fkey('a', 'claim:mine'))).toBe(true);
        await sameAsHost(store, rows);
    });

    it('regression: a dump or extractedEntries with lazy bodies reads them all through loadAllExtracted, and throws without it', async () => {
        const rows = fakeRows();
        rows.lazyBodies = true;
        rows.db.extracted.set('h9', ex('nine'));
        const store = rowStore(rows);
        await store.putExtracted('mem', ex('m'));
        expect(Object.keys(await store.extractedEntries()).sort()).toEqual(['h9', 'mem']);
        expect(await canonicalDump(store)).toContain('"h9"');

        const bare = fakeRows();
        delete bare.loadAllExtracted;
        const partial = rowStore(bare);
        await expect(partial.extractedEntries()).rejects.toThrow('knowledge_dump_needs_all_bodies');
        await expect(canonicalDump(partial)).rejects.toThrow('knowledge_dump_needs_all_bodies');
    });

    it('regression: preloadExtracted reads the missing bodies once; a miss is remembered until the hash is put', async () => {
        const rows = fakeRows();
        rows.lazyBodies = true;
        for (const h of ['h1', 'h2']) rows.db.extracted.set(h, ex(h));
        const store = rowStore(rows);
        await store.preloadExtracted(['h1', 'h2', 'h1', 'zz']);
        expect([...rows.bodyReads].sort()).toEqual(['h1', 'h2', 'zz']);
        expect(await store.getExtracted('h2')).toEqual(ex('h2'));
        expect(await store.getExtracted('zz')).toBeUndefined();
        expect(rows.bodyReads).toHaveLength(3); // nothing read again
        await store.putExtracted('zz', ex('z'));
        expect(await store.getExtracted('zz')).toEqual(ex('z'));
    });

    it('regression: a mutation made while apply is pending is persisted by the next apply', async () => {
        const rows = fakeRows();
        const store = rowStore(rows);
        rows.hold = deferred();
        const first = store.write({ scope: SCOPE, facts: [fact('claim:1')], relations: [] });
        await settle();
        expect(rows.applies).toHaveLength(1);
        const second = store.write({ scope: SCOPE, facts: [fact('claim:2')], relations: [] });
        await settle();
        rows.hold.resolve();
        await Promise.all([first, second]);
        expect(rows.applies).toHaveLength(2);
        expect(rows.applies[0]!.upserts.facts.map(([, x]) => x.key)).toEqual(['claim:1']);
        expect(rows.applies[1]!.upserts.facts.map(([, x]) => x.key)).toEqual(['claim:2']);
        expect([...rows.db.facts.values()].map((x) => x.key).sort()).toEqual(['claim:1', 'claim:2']);
    });

    it('regression: a rejected apply loses nothing: the retry persists the failed rows with the newer value of a row marked since', async () => {
        const rows = fakeRows();
        const store = rowStore(rows);
        rows.failNext = 1;
        await expect(store.write({ scope: SCOPE, facts: [fact('claim:1', 'old'), fact('claim:2')], relations: [rel('fact:claim:1', 'src:a::f.md#s', 'about')] })).rejects.toThrow('disk full');
        expect(rows.db.facts.size).toBe(0);
        await store.markRecheck(['h1']); // a later mutation: persists, and carries the failed rows with it
        await store.write({ scope: SCOPE, facts: [fact('claim:1', 'new')], relations: [] });
        expect([...rows.db.facts.values()].map((x) => [x.key, x.content]).sort()).toEqual([['claim:1', 'new'], ['claim:2', 'c']]);
        expect(rows.db.relations.size).toBe(1);
        expect(rows.db.scopes.has('a')).toBe(true);
        expect([...rows.db.recheck]).toEqual(['h1']);
        expect(await canonicalDump(loadedFrom(snapshotOf(rows.db)))).toBe(await canonicalDump(store));
    });

    it('regression: recheck survives a reload', async () => {
        const rows = fakeRows();
        const store = rowStore(rows);
        await store.markRecheck(['h2', 'h1', 'h3']);
        await store.clearRecheck(['h2']);
        expect(rows.applies.at(-1)!.deletes.recheck).toEqual(['h2']);
        expect(await rowStore(rows).recheckHashes()).toEqual(['h1', 'h3']);
    });

    it('regression: `reloaded` replaces memory, and a mark made while apply was pending survives in memory and is persisted next', async () => {
        const rows = fakeRows();
        const store = rowStore(rows);
        await store.write({ scope: SCOPE, facts: [fact('claim:gone')], relations: [] });
        rows.hold = deferred();
        // Another window: deletes `claim:gone`, adds `claim:other`. Its rows come back with this window's apply.
        rows.reload = () => { rows.db.facts.delete(fkey('a', 'claim:gone')); rows.db.facts.set(fkey('a', 'claim:other'), fact('claim:other')); return snapshotOf(rows.db); };
        const first = store.write({ scope: SCOPE, facts: [fact('claim:1')], relations: [] });
        await settle();
        const second = store.write({ scope: SCOPE, facts: [fact('claim:2')], relations: [] }); // marked during the apply
        await settle();
        rows.hold.resolve();
        await Promise.all([first, second]);
        const keys = (await store.facts({ rootId: 'a' })).map((x) => x.key).sort();
        expect(keys).toEqual(['claim:1', 'claim:2', 'claim:other']); // replaced (gone is gone, other is here), claim:2 not lost
        expect(rows.applies.at(-1)!.upserts.facts.map(([, x]) => x.key)).toEqual(['claim:2']);
        expect([...rows.db.facts.values()].map((x) => x.key).sort()).toEqual(keys);
    });

    it('regression: `reloaded` rebuilds the relations-by-path index, so markStale finds a relation another window wrote', async () => {
        const rows = fakeRows();
        const store = rowStore(rows);
        rows.reload = () => { rows.db.relations.set(rkey('a', 'fact:claim:9', 'src:a::z.md#s', 'about'), rel('fact:claim:9', 'src:a::z.md#s', 'about')); return snapshotOf(rows.db); };
        await store.write({ scope: SCOPE, facts: [], relations: [] });
        await store.markStale({ rootId: 'a' }, 'src:a::z.md#s', 'other-hash');
        expect((await store.relations({ rootId: 'a' }))[0]!.staleAt).toBe('T');
    });

    it('regression: when another writer changed the store, a read reloads first (one check in flight, none inside a batch)', async () => {
        const rows = fakeRows();
        const store = rowStore(rows);
        await store.facts({ rootId: 'a' });
        rows.checks = 0;
        rows.db.facts.set(fkey('a', 'claim:theirs'), fact('claim:theirs'));
        rows.changed = true;
        const reads = await Promise.all([1, 2, 3, 4, 5].map(() => store.facts({ rootId: 'a' })));
        expect(rows.checks).toBe(1);
        expect(reads.every((r) => r.map((x) => x.key).join() === 'claim:theirs')).toBe(true);

        rows.db.facts.set(fkey('a', 'claim:later'), fact('claim:later'));
        rows.changed = true;
        rows.checks = 0;
        await store.batched(async () => { expect((await store.facts({ rootId: 'a' })).map((x) => x.key)).toEqual(['claim:theirs']); });
        expect(rows.checks).toBe(0);
        expect((await store.facts({ rootId: 'a' })).map((x) => x.key).sort()).toEqual(['claim:later', 'claim:theirs']); // the next read outside the batch sees it
    });

    it('regression: an extract body missing from memory is read through loadExtracted once, and kept', async () => {
        const rows = fakeRows();
        rows.lazyBodies = true;
        rows.db.extracted.set('h9', ex('nine'));
        const store = rowStore(rows);
        expect(await store.getExtracted('h9')).toEqual(ex('nine'));
        expect(await store.getExtracted('h9')).toEqual(ex('nine'));
        expect(await store.getExtracted('nope')).toBeUndefined();
        expect(rows.bodyReads).toEqual(['h9', 'nope']);
        await store.putExtracted('h9', ex('nine')); // the same body: no row written
        expect(rows.applies).toHaveLength(0);
    });

    it('regression: without rows the store writes the same bytes as the dump, and needs one of the two ports', async () => {
        const p = memPersistence();
        const store = new LocalSnapshotStore(p, { now: () => 'T' });
        await store.write({ scope: SCOPE, facts: [fact('claim:1')], relations: [rel('fact:claim:1', 'src:a::f.md#s', 'about')] });
        await store.putExtracted('h1', ex('one'));
        await store.markRecheck(['h1']);
        expect(await canonicalDump(store)).toBe(p.value);
        expect(() => new (LocalSnapshotStore as unknown as new (p: undefined, o: object) => unknown)(undefined, {})).toThrow();
    });
});
