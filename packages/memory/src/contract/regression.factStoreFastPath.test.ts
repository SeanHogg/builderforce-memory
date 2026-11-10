/**
 * src/contract/regression.factStoreFastPath.test.ts — `saveFactGraph` with a current `previous` reads only the version
 * (PRD 41 §B4). Before: every save read the whole stored manifest and every derived row, O(files) per edit, which the
 * VS Code fact index cannot afford on each keystroke-driven save. A stale or foreign `previous` still falls back to the
 * full store diff (the H1 and ABA regressions), and an incremental save still equals a from-scratch save.
 */

import type { FileFacts } from '../knowledge/facts/fileFacts.js';
import { buildFactGraph, factGraphDump, updateFactGraph } from '../knowledge/rules/factGraph.js';
import { derivedRowKey } from '../knowledge/rules/evaluateRules.js';
import { factStoreDump, memoryFactStore, saveFactGraph, type FactStorePort, type MemoryFactStore } from '../knowledge/rules/factStore.js';
import { emptyFacts, FIXTURE_RULES } from '../knowledge/rules/rules.testkit.test-util.js';

const file = (content: string, over: Partial<FileFacts> = {}): FileFacts => ({ ...emptyFacts(content), ...over });

const tree = (): Array<readonly [string, FileFacts]> => [
    ['src/a.ts', file('ca', { imports: [{ specifier: './b', resolved: 'src/b.ts', names: [] }], exports: [{ name: 'a', kind: 'function', signatureHash: 'a' }] })],
    ['src/b.ts', file('cb', { calls: [{ callee: 'db.get', enclosing: 'b', awaited: true, inLoop: true, count: 1 }] })],
    ['src/auth.ts', file('cauth', { exports: [{ name: 'requireAuth', kind: 'function', signatureHash: 'r' }] })],
    ['src/routes.ts', file('cr', {
        imports: [{ specifier: './a', resolved: 'src/a.ts', names: [] }],
        entryPoints: [{ kind: 'http-route', key: 'GET /x', symbol: 'h', guards: [], inputs: [], chain: ['src/auth.ts'] }],
    })],
];
const built = () => buildFactGraph(tree(), FIXTURE_RULES).graph;
const bFacts = () => tree()[1]![1];

/** Counts the O(files) reads. */
function counting(store: MemoryFactStore): FactStorePort & { calls: { readManifest: number; derivedRows: number } } {
    const calls = { readManifest: 0, derivedRows: 0 };
    return {
        ...store,
        readManifest: () => { calls.readManifest++; return store.readManifest(); },
        derivedRows: () => { calls.derivedRows++; return store.derivedRows(); },
        calls,
    };
}

describe('regression: saveFactGraph fast path', () => {
    it('a save with a current previous performs no readManifest and no derivedRows call', async () => {
        const store = memoryFactStore();
        const s0 = await saveFactGraph(store, built());
        const port = counting(store);
        const g1 = updateFactGraph(s0.graph, [{ path: 'src/b.ts', facts: { ...bFacts(), contentHash: 'cb2' } }, { path: 'src/new.ts', facts: file('cn') }], FIXTURE_RULES).graph;
        const s1 = await saveFactGraph(port, g1, s0);
        const g2 = updateFactGraph(g1, [{ path: 'src/a.ts', facts: null }], FIXTURE_RULES).graph;
        const s2 = await saveFactGraph(port, g2, s1);
        expect(port.calls).toEqual({ readManifest: 0, derivedRows: 0 });
        expect([s1.puts, s2.deletes]).toEqual([['src/b.ts', 'src/new.ts'], ['src/a.ts']]);
        expect(await factStoreDump(store)).toBe(factGraphDump(g2));
    });

    it('incremental fast-path saves equal a from-scratch save', async () => {
        const g0 = built();
        const incremental = memoryFactStore();
        let s = await saveFactGraph(incremental, g0);
        let g = g0;
        for (const change of [
            [{ path: 'src/b.ts', facts: { ...bFacts(), contentHash: 'cb2' } }],
            [{ path: 'src/new.ts', facts: file('cn') }],
            [{ path: 'src/auth.ts', facts: null }],
        ]) {
            g = updateFactGraph(g, change, FIXTURE_RULES).graph;
            s = await saveFactGraph(incremental, g, s);
        }
        const scratch = memoryFactStore();
        await saveFactGraph(scratch, g);
        expect(await factStoreDump(incremental)).toBe(await factStoreDump(scratch));
        expect(await factStoreDump(incremental)).toBe(factGraphDump(g));
    });

    it('a stale previous (another writer changed only derived rows) falls back to the store diff', async () => {
        const g0 = built();
        const store = memoryFactStore();
        const s0 = await saveFactGraph(store, g0);
        const key = derivedRowKey('export-count', 'src/auth.ts');
        const row = g0.derived.get(key)!;
        await store.writeManifest({ epoch: s0.epoch, generation: 2, puts: new Map(), deletes: [] }, [{ ...row, inputs: 'tampered', findings: [] }], []);
        const port = counting(store);
        const g1 = updateFactGraph(g0, [{ path: 'src/b.ts', facts: { ...bFacts(), contentHash: 'cb2' } }], FIXTURE_RULES).graph;
        await saveFactGraph(port, g1, s0);
        expect(port.calls).toEqual({ readManifest: 1, derivedRows: 1 });
        expect(await factStoreDump(store)).toBe(factGraphDump(g1));
    });

    it('a previous from before a reset (same generation, other epoch) falls back to the store diff', async () => {
        const g0 = built();
        const store = memoryFactStore();
        const s0 = await saveFactGraph(store, g0);
        await store.reset();
        const other = updateFactGraph(g0, [{ path: 'src/new.ts', facts: file('cn') }], FIXTURE_RULES).graph;
        const s1 = await saveFactGraph(store, other);
        expect(s1.generation).toBe(s0.generation);
        const port = counting(store);
        const g = updateFactGraph(g0, [], FIXTURE_RULES).graph;
        await saveFactGraph(port, g, s0);
        expect(port.calls).toEqual({ readManifest: 1, derivedRows: 1 });
        expect(await factStoreDump(store)).toBe(factGraphDump(g));
    });

    it('a writer that lands between the version read and the commit makes the fast-path save conflict', async () => {
        const g0 = built();
        const store = memoryFactStore();
        const s0 = await saveFactGraph(store, g0);
        const racing: FactStorePort = {
            ...store,
            readVersion: async () => {
                const v = await store.readVersion();
                await store.writeManifest({ epoch: v.epoch, generation: v.generation + 1, puts: new Map(), deletes: [] }, [], []);
                return v;
            },
        };
        const g1 = updateFactGraph(g0, [{ path: 'src/new.ts', facts: file('cn') }], FIXTURE_RULES).graph;
        await expect(saveFactGraph(racing, g1, s0)).rejects.toThrow(/fact_store_generation_conflict/);
    });
});
