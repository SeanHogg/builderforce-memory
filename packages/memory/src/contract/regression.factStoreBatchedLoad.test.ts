/**
 * src/contract/regression.factStoreBatchedLoad.test.ts — `loadFactGraph` reads objects in bounded `getObjects` batches.
 *
 * Before: one `getObjects` call with EVERY referenced hash, which the VS Code host's SQLite port serves as one synchronous
 * transaction on the extension host's main thread (seconds on a large workspace). Now the load asks in bounded batches and
 * yields between them; the loaded graph is exactly what the single call produced.
 */

import type { FileFacts } from '../knowledge/facts/fileFacts.js';
import { buildFactGraph, factGraphDump } from '../knowledge/rules/factGraph.js';
import { loadFactGraph, memoryFactStore, saveFactGraph, type FactStorePort } from '../knowledge/rules/factStore.js';
import { emptyFacts, FIXTURE_RULES } from '../knowledge/rules/rules.testkit.test-util.js';

const FILES = 1300;

const tree = (): Array<readonly [string, FileFacts]> =>
    Array.from({ length: FILES }, (_, i) => [
        `src/f${i}.ts`,
        { ...emptyFacts(`c${i}`), exports: [{ name: `e${i}`, kind: 'function', signatureHash: `s${i}` }] },
    ] as const);

describe('regression: batched fact store load', () => {
    it('regression: a load asks getObjects in bounded batches covering every hash once, and returns the same graph as one call', async () => {
        const graph = buildFactGraph(tree(), FIXTURE_RULES).graph;
        const store = memoryFactStore();
        await saveFactGraph(store, graph);

        const calls: string[][] = [];
        const batched: FactStorePort = { ...store, getObjects: (hashes) => { calls.push([...hashes]); return store.getObjects(hashes); } };
        // The reference: a port that answers every hash at once whatever it is asked, as the single call did.
        const all = [...graph.objects.keys()];
        const whole: FactStorePort = { ...store, getObjects: () => store.getObjects(all) };

        const loaded = await loadFactGraph(batched, { verify: 'manifest' });
        const reference = await loadFactGraph(whole, { verify: 'manifest' });
        expect(factGraphDump(loaded.graph)).toBe(factGraphDump(reference.graph));
        expect(factGraphDump(loaded.graph)).toBe(factGraphDump(graph));

        expect(calls.length).toBeGreaterThan(1);
        expect(Math.max(...calls.map((c) => c.length))).toBeLessThanOrEqual(512);
        const asked = calls.flat();
        expect(asked.length).toBe(new Set(asked).size);
        expect(new Set(asked)).toEqual(new Set(all));
    });

    it('regression: an object missing from a later batch is still fact_store_object_missing', async () => {
        const graph = buildFactGraph(tree(), FIXTURE_RULES).graph;
        const store = memoryFactStore();
        await saveFactGraph(store, graph);
        const sorted = [...graph.objects.keys()].sort();
        const gone = sorted[sorted.length - 1]!;
        const port: FactStorePort = {
            ...store,
            getObjects: async (hashes) => {
                const got = new Map(await store.getObjects(hashes));
                got.delete(gone);
                return got;
            },
        };
        await expect(loadFactGraph(port, { verify: 'manifest' })).rejects.toThrow(`fact_store_object_missing:${gone}`);
    });
});
