/**
 * tests/memorystore-min-relevance.test.ts — recall relevance floor.
 *
 * Recall used to always return something: a query sharing no token with any
 * fact scored 0 on the Jaccard fallback and still came back as the "best"
 * match, so an unrelated fact was surfaced as if it were relevant. The store
 * now drops anything below `minRelevance` (default 0.05) on both the lexical
 * and the hybrid path, so a weak match returns nothing rather than noise.
 */

import 'fake-indexeddb/auto';
import { MemoryStore } from '../src/memory/MemoryStore.js';

let _db = 0;
const freshStore = (opts: Record<string, unknown> = {}) =>
    new MemoryStore({ dbName: `minrel-${_db++}-${Date.now()}`, ...opts });

describe('recall relevance floor', () => {
    it('returns nothing for a query sharing no token with any fact', async () => {
        const store = freshStore();
        await store.remember('k1', 'the quarterly revenue forecast');
        await store.remember('k2', 'deployment runbook for the api gateway');

        // No embedding runtime → Jaccard path. Zero overlap scores 0, below the
        // default 0.05 floor, so the old "always return the least-bad fact"
        // behaviour is gone.
        expect(await store.recallSimilar('xylophone zebra quokka', 5)).toEqual([]);
        expect((await store.recallRanked('xylophone zebra quokka', 5)).hits).toEqual([]);
    });

    it('still returns a genuine overlap above the floor', async () => {
        const store = freshStore();
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
        // Floor disabled: even a zero-overlap query returns its least-bad hit.
        expect(await open.recallSimilar('xylophone zebra quokka', 5)).toHaveLength(1);
    });
});
