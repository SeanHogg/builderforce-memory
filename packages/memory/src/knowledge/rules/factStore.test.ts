/**
 * src/knowledge/rules/factStore.test.ts — the FactStorePort contract (PRD 41 §B4, §B8.6) over the in-memory reference
 * adapter: round trip, delta saves equal a scratch save, a stale `previous` never trusts identity, restored traces are
 * honoured, an interrupted write leaves the previous manifest valid, mixed-generation loads retry, and the sweep.
 */

import type { FileFacts } from '../facts/fileFacts.js';
import { factsHash, factsJson } from '../facts/factsHash.js';
import { derivedRowKey } from './evaluateRules.js';
import { buildFactGraph, factGraphDump, updateFactGraph } from './factGraph.js';
import {
    factStoreDump, loadFactGraph, memoryFactStore, saveFactGraph, sweepFactStore, FACT_STORE_LOAD_ATTEMPTS, type FactStorePort,
} from './factStore.js';
import { emptyFacts, FIXTURE_RULES } from './rules.testkit.test-util.js';

const file = (content: string, over: Partial<FileFacts> = {}): FileFacts => ({ ...emptyFacts(content), ...over });

const tree = (): Array<readonly [string, FileFacts]> => [
    ['src/a.ts', file('ca', { imports: [{ specifier: './b', resolved: 'src/b.ts', names: [] }], exports: [{ name: 'a', kind: 'function', signatureHash: 'a' }] })],
    ['src/b.ts', file('cb', { calls: [{ callee: 'db.get', enclosing: 'b', awaited: true, inLoop: true, count: 1 }] })],
    ['src/auth.ts', file('cauth', { exports: [{ name: 'requireAuth', kind: 'function', signatureHash: 'r' }] })],
    ['src/routes.ts', file('cr', {
        imports: [{ specifier: './a', resolved: 'src/a.ts', names: [] }],
        entryPoints: [{ kind: 'http-route', key: 'GET /x', symbol: 'h', guards: [], inputs: [], chain: ['src/auth.ts'] }],
        mounts: [{ prefix: '/api', target: './a', order: 1 }],
    })],
];

const built = () => buildFactGraph(tree(), FIXTURE_RULES).graph;
const bFacts = () => tree()[1]![1];

describe('FactStorePort: round trip', () => {
    it('save → load restores a graph with the same dump, at the saved version', async () => {
        const graph = built();
        const store = memoryFactStore();
        const saved = await saveFactGraph(store, graph);
        expect(saved.generation).toBe(1);
        const loaded = await loadFactGraph(store);
        expect([loaded.epoch, loaded.generation]).toEqual([saved.epoch, 1]);
        expect(factGraphDump(loaded.graph)).toBe(factGraphDump(graph));
        expect(await factStoreDump(store)).toBe(factGraphDump(graph));
        const cheap = await loadFactGraph(store, { verify: 'manifest' });
        expect(factGraphDump(cheap.graph)).toBe(factGraphDump(graph));
    });

    it('objects are stored as their canonical JSON under their facts hash', async () => {
        const graph = built();
        const store = memoryFactStore();
        await saveFactGraph(store, graph);
        const stored = await store.getObjects([...graph.objects.keys()]);
        for (const [hash, facts] of graph.objects) expect(stored.get(hash)).toBe(factsJson(facts));
    });

    it('an object written before `mounts` / `uses` existed loads with them empty, under the same hash', async () => {
        const store = memoryFactStore();
        const old = file('old');
        const hash = factsHash(old);
        await store.putObjects(new Map([[hash, JSON.stringify(old)]])); // no mounts, no uses
        const kinds = buildFactGraph([['x.ts', old]], []).graph.manifest.get('x.ts')!.kinds;
        const { epoch } = await store.readVersion();
        await store.writeManifest({ epoch, generation: 1, puts: new Map([['src/old.ts', { contentHash: 'old', factsHash: hash, kinds }]]), deletes: [] }, [], []);
        const facts = (await loadFactGraph(store)).graph.objects.get(hash)!;
        expect(facts.mounts).toEqual([]);
        expect(facts.uses).toEqual([]);
        expect(factsHash(facts)).toBe(hash);
    });
});

describe('FactStorePort: delta saves', () => {
    it('an incremental save after an update equals a from-scratch save, and writes only the delta', async () => {
        const g0 = built();
        const incremental = memoryFactStore();
        const s0 = await saveFactGraph(incremental, g0);
        const g1 = updateFactGraph(g0, [{ path: 'src/b.ts', facts: { ...bFacts(), contentHash: 'cb2' } }, { path: 'src/new.ts', facts: file('cn') }], FIXTURE_RULES).graph;
        const delta = await saveFactGraph(incremental, g1, s0);
        expect(delta.generation).toBe(2);
        expect(delta.puts).toEqual(['src/b.ts', 'src/new.ts']);
        expect(delta.deletes).toEqual([]);
        expect(delta.objects).toHaveLength(2);
        expect(delta.derivedPuts.length).toBeLessThan(g1.derived.size);

        const scratch = memoryFactStore();
        await saveFactGraph(scratch, g1);
        expect(await factStoreDump(incremental)).toBe(await factStoreDump(scratch));
        expect(await factStoreDump(incremental)).toBe(factGraphDump(g1));
    });

    it('a delete removes the manifest entry and its derived rows; an unchanged graph saves nothing but the generation', async () => {
        const g0 = built();
        const store = memoryFactStore();
        await saveFactGraph(store, g0);
        const g1 = updateFactGraph(g0, [{ path: 'src/b.ts', facts: null }], FIXTURE_RULES).graph;
        const delta = await saveFactGraph(store, g1);
        expect(delta.deletes).toEqual(['src/b.ts']);
        expect(delta.derivedDeletes.length).toBeGreaterThan(0);
        expect(delta.derivedDeletes.every((k) => k.endsWith('\u0000src/b.ts'))).toBe(true);
        expect(await factStoreDump(store)).toBe(factGraphDump(g1));
        const again = await saveFactGraph(store, g1, delta);
        expect([again.objects, again.puts, again.deletes, again.derivedPuts, again.derivedDeletes]).toEqual([[], [], [], [], []]);
    });

    it('two writers never interleave: a stale generation or a foreign epoch is rejected', async () => {
        const store = memoryFactStore();
        const { epoch } = await saveFactGraph(store, built());
        expect(() => store.writeManifest({ epoch, generation: 1, puts: new Map(), deletes: [] }, [], [])).toThrow(/fact_store_generation_conflict/);
        expect(() => store.writeManifest({ epoch: 'other', generation: 2, puts: new Map(), deletes: [] }, [], [])).toThrow(/fact_store_generation_conflict/);
    });

    it('writeManifest refuses an entry whose object is not stored', async () => {
        const store = memoryFactStore();
        const s = await saveFactGraph(store, built());
        const entry = s.graph.manifest.get('src/a.ts')!;
        expect(() => store.writeManifest({ epoch: s.epoch, generation: 2, puts: new Map([['src/z.ts', { ...entry, factsHash: 'nope' }]]), deletes: [] }, [], []))
            .toThrow('fact_store_object_missing:nope');
    });

    it('regression (H1): another writer changed ONLY derived rows — a stale previous does not trust identity', async () => {
        const g0 = built();
        const store = memoryFactStore();
        const s0 = await saveFactGraph(store, g0);
        // Another writer: same manifest, one derived row rewritten.
        const key = derivedRowKey('export-count', 'src/auth.ts');
        const row = g0.derived.get(key)!;
        await store.writeManifest({ epoch: s0.epoch, generation: 2, puts: new Map(), deletes: [] }, [{ ...row, inputs: 'tampered', findings: [] }], []);
        // Our save, from the stale previous (generation 1), editing a different file.
        const g1 = updateFactGraph(g0, [{ path: 'src/b.ts', facts: { ...bFacts(), contentHash: 'cb2' } }], FIXTURE_RULES).graph;
        expect(g1.derived.get(key)).toBe(row); // identity would say "unchanged"
        await saveFactGraph(store, g1, s0);
        expect(await factStoreDump(store)).toBe(factGraphDump(g1));
    });

    it('a reset store is never mistaken for the old one, even at the same generation (only the epoch tells them apart)', async () => {
        const g0 = built();
        const store = memoryFactStore();
        const s0 = await saveFactGraph(store, g0);
        expect(s0.generation).toBe(1);
        await store.reset();
        // The reset store reaches generation 1 again, with different derived rows than s0 saw.
        const other = updateFactGraph(g0, [{ path: 'src/new.ts', facts: file('cn') }], FIXTURE_RULES).graph;
        const s1 = await saveFactGraph(store, other);
        expect(s1.generation).toBe(s0.generation);
        expect(s1.epoch).not.toBe(s0.epoch);
        // A save from the stale s0 (same generation, old epoch) must not trust identity: g's rows equal s0's by identity,
        // so an identity diff would write nothing for them and leave `other`'s rows behind.
        const g = updateFactGraph(g0, [], FIXTURE_RULES).graph;
        await saveFactGraph(store, g, s0);
        expect(await factStoreDump(store)).toBe(factGraphDump(g));
    });
});

describe('FactStorePort: restored traces are honoured', () => {
    it('load → update with no change reuses every row', async () => {
        const store = memoryFactStore();
        await saveFactGraph(store, built());
        const { graph } = await loadFactGraph(store);
        const update = updateFactGraph(graph, [], FIXTURE_RULES);
        expect(update.trace.length).toBeGreaterThan(0);
        expect(update.trace.filter((t) => t.outcome === 'ran')).toEqual([]);
    });

    it('load → a content-only edit reruns only the content-reading row of that file (early cutoff across a restart)', async () => {
        const store = memoryFactStore();
        await saveFactGraph(store, built());
        const { graph } = await loadFactGraph(store);
        const update = updateFactGraph(graph, [{ path: 'src/b.ts', facts: { ...bFacts(), contentHash: 'cb-comment-edit' } }], FIXTURE_RULES);
        expect(update.trace.filter((t) => t.outcome === 'ran')).toEqual([{ rule: 'content-tag', focus: 'src/b.ts', outcome: 'ran' }]);
    });
});

describe('FactStorePort: crash safety and corruption', () => {
    it('a write interrupted between the object puts and the manifest leaves the previous manifest valid', async () => {
        const g0 = built();
        const store = memoryFactStore();
        const s0 = await saveFactGraph(store, g0);
        const before = await factStoreDump(store);

        const crashing: FactStorePort = { ...store, writeManifest: () => { throw new Error('crash'); } };
        const g1 = updateFactGraph(g0, [{ path: 'src/new.ts', facts: file('cn') }], FIXTURE_RULES).graph;
        await expect(saveFactGraph(crashing, g1, s0)).rejects.toThrow('crash');

        expect(store.objectHashes()).toContain(g1.manifest.get('src/new.ts')!.factsHash);
        expect(await factStoreDump(store)).toBe(before);
        expect(factGraphDump((await loadFactGraph(store)).graph)).toBe(factGraphDump(g0));

        await saveFactGraph(store, g1, s0);
        expect(await factStoreDump(store)).toBe(factGraphDump(g1));
    });

    it('objects are write-once', async () => {
        const store = memoryFactStore();
        await store.putObjects(new Map([['h', '{"a":1}']]));
        await store.putObjects(new Map([['h', '{"a":2}']]));
        expect((await store.getObjects(['h'])).get('h')).toBe('{"a":1}');
    });

    it('an object whose bytes no longer hash to its address is refused by a full load', async () => {
        const g0 = built();
        const store = memoryFactStore();
        const hash = g0.manifest.get('src/auth.ts')!.factsHash;
        await store.putObjects(new Map([[hash, factsJson(file('cauth', { exports: [{ name: 'other', kind: 'function', signatureHash: 'o' }] }))]]));
        await saveFactGraph(store, g0);
        await expect(loadFactGraph(store)).rejects.toThrow(`fact_store_object_corrupt:${hash}`);
        // `manifest` verification trusts the address but still checks the kind hashes.
        await expect(loadFactGraph(store, { verify: 'manifest' })).rejects.toThrow('fact_store_manifest_corrupt:src/auth.ts');
    });

    it('bad JSON is fact_store_object_corrupt; the recovery is reset() and a rebuild', async () => {
        const g0 = built();
        const store = memoryFactStore();
        const hash = g0.manifest.get('src/a.ts')!.factsHash;
        await store.putObjects(new Map([[hash, '{not json']]));
        await saveFactGraph(store, g0);
        await expect(loadFactGraph(store)).rejects.toThrow(`fact_store_object_corrupt:${hash}`);
        await store.reset();
        await saveFactGraph(store, g0);
        expect(await factStoreDump(store)).toBe(factGraphDump(g0));
    });

    it('a derived row stored under the wrong key is fact_store_derived_corrupt', async () => {
        const g0 = built();
        const store = memoryFactStore();
        await saveFactGraph(store, g0);
        const port: FactStorePort = { ...store, derivedRows: () => new Map([['wrong', [...g0.derived.values()][0]!]]) };
        await expect(loadFactGraph(port)).rejects.toThrow('fact_store_derived_corrupt:wrong');
    });
});

describe('FactStorePort: mixed-generation loads', () => {
    it('a write that lands mid-load makes the load retry, and the result is one generation', async () => {
        const g0 = built();
        const store = memoryFactStore();
        const s0 = await saveFactGraph(store, g0);
        const g1 = updateFactGraph(g0, [{ path: 'src/new.ts', facts: file('cn') }], FIXTURE_RULES).graph;
        let fired = false;
        const racing: FactStorePort = {
            ...store,
            derivedRows: async () => {
                if (!fired) { fired = true; await saveFactGraph(store, g1, s0); }
                return store.derivedRows();
            },
        };
        const loaded = await loadFactGraph(racing);
        expect(loaded.generation).toBe(2);
        expect(factGraphDump(loaded.graph)).toBe(factGraphDump(g1));
    });

    it('a store that keeps moving fails the load with a conflict after the bounded attempts', async () => {
        const store = memoryFactStore();
        await saveFactGraph(store, built());
        let reads = 0;
        const moving: FactStorePort = { ...store, readVersion: () => ({ epoch: 'e', generation: ++reads }) };
        await expect(loadFactGraph(moving)).rejects.toThrow('fact_store_generation_conflict:load');
        expect(reads).toBe(FACT_STORE_LOAD_ATTEMPTS);
    });
});

describe('FactStorePort: sweep', () => {
    it('removes only unreferenced objects and bumps the generation', async () => {
        const g0 = built();
        const store = memoryFactStore();
        await saveFactGraph(store, g0);
        await store.putObjects(new Map([['orphan', '{}']]));
        expect(await sweepFactStore(store)).toBe(1);
        expect(store.objectHashes()).not.toContain('orphan');
        expect((await store.readVersion()).generation).toBe(2);
        expect(await factStoreDump(store)).toBe(factGraphDump(g0));
    });

    it('a no-op sweep leaves the version unchanged', async () => {
        const store = memoryFactStore();
        const s0 = await saveFactGraph(store, built());
        expect(await sweepFactStore(store)).toBe(0);
        expect(await store.readVersion()).toEqual({ epoch: s0.epoch, generation: s0.generation });
        // So a writer holding s0 is not disturbed by it.
        const again = await saveFactGraph(store, s0.graph, s0);
        expect(again.generation).toBe(s0.generation + 1);
    });

    it('sweep vs writer:the writer\'s objects are swept, its manifest write conflicts, and the retry re-puts them', async () => {
        const g0 = built();
        const store = memoryFactStore();
        const s0 = await saveFactGraph(store, g0);
        const change = [{ path: 'src/new.ts', facts: file('cn') }];
        const g1 = updateFactGraph(g0, change, FIXTURE_RULES).graph;
        const newHash = g1.manifest.get('src/new.ts')!.factsHash;
        // The sweep runs between the writer's object puts and its manifest write.
        const raced: FactStorePort = {
            ...store,
            writeManifest: async (delta, puts, deletes) => {
                expect(await sweepFactStore(store)).toBe(1);
                expect(store.objectHashes()).not.toContain(newHash);
                return store.writeManifest(delta, puts, deletes);
            },
        };
        await expect(saveFactGraph(raced, g1, s0)).rejects.toThrow(/fact_store_generation_conflict/);
        // The documented recovery: load, re-apply the pending changes, save with the loaded version.
        const loaded = await loadFactGraph(store);
        const retried = updateFactGraph(loaded.graph, change, FIXTURE_RULES).graph;
        const s1 = await saveFactGraph(store, retried, loaded);
        expect(s1.objects).toEqual([newHash]);
        expect(await factStoreDump(store)).toBe(factGraphDump(g1));
    });
});
