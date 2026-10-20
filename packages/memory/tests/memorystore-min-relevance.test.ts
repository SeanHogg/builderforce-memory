/**
 * tests/memorystore-min-relevance.test.ts — recall relevance floor (opt-in).
 *
 * Recall always returns its best-ranked hits, even a zero-overlap one; that is
 * the long-standing contract and stays the default (`minRelevance` 0). A store
 * that would rather return nothing than noise opts in with a floor, which then
 * applies on both the lexical and the hybrid path.
 */

import 'fake-indexeddb/auto';
import { MemoryStore } from '../src/memory/MemoryStore.js';

let _db = 0;
const freshStore = (opts: Record<string, unknown> = {}) =>
    new MemoryStore({ dbName: `minrel-${_db++}-${Date.now()}`, ...opts });

describe('recall relevance floor', () => {
    it('is off by default: recall still returns its least-bad hit', async () => {
        const store = freshStore();
        await store.remember('k1', 'the quarterly revenue forecast');
        await store.remember('k2', 'deployment runbook for the api gateway');

        expect(await store.recallSimilar('xylophone zebra quokka', 5)).toHaveLength(2);
        // Hybrid: a lexical match keeps its (small, RRF-scale) fused score unfiltered.
        expect((await store.recallRanked('quarterly revenue', 5)).hits.map((h) => h.entry.key)).toEqual(['k1']);
    });

    it('opted in, returns nothing for a query sharing no token with any fact', async () => {
        const store = freshStore({ minRelevance: 0.05 });
        await store.remember('k1', 'the quarterly revenue forecast');
        await store.remember('k2', 'deployment runbook for the api gateway');

        // No embedding runtime → Jaccard path. Zero overlap scores 0, below the floor.
        expect(await store.recallSimilar('xylophone zebra quokka', 5)).toEqual([]);
        expect((await store.recallRanked('xylophone zebra quokka', 5)).hits).toEqual([]);
    });

    it('opted in, still returns a genuine overlap above the floor', async () => {
        const store = freshStore({ minRelevance: 0.05 });
        await store.remember('k1', 'the quarterly revenue forecast');
        await store.remember('k2', 'deployment runbook for the api gateway');

        const hits = await store.recallSimilar('quarterly revenue', 5);
        expect(hits.map((e) => e.key)).toEqual(['k1']);
    });

    it('honours an explicit floor, and 0 disables it', async () => {
        const floored = freshStore({ minRelevance: 0.9 });
        await floored.remember('k1', 'alpha beta gamma');
        // One shared token of three is 1/5 = 0.2 — a real match, but under 0.9.
        expect(await floored.recallSimilar('alpha delta epsilon', 5)).toEqual([]);

        const open = freshStore({ minRelevance: 0 });
        await open.remember('k1', 'alpha beta gamma');
        expect(await open.recallSimilar('xylophone zebra quokka', 5)).toHaveLength(1);
    });
});
