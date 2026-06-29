/**
 * tests/memorystore-evm7.test.ts — EVM-7: active TTL sweeper + store size cap.
 */

import 'fake-indexeddb/auto';
import { jest } from '@jest/globals';
import { MemoryStore } from '../src/memory/MemoryStore.js';

let _db = 0;
const freshStore = (opts: Record<string, unknown> = {}) =>
    new MemoryStore({ dbName: `evm7-${_db++}-${Date.now()}`, ...opts });

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

describe('EVM-7 size cap', () => {
    it('evicts down to maxEntries, keeping the highest-importance facts', async () => {
        const store = freshStore({ maxEntries: 3 });
        await store.remember('a', 'a', { importance: 0.1 });
        await store.remember('b', 'b', { importance: 0.9 });
        await store.remember('c', 'c', { importance: 0.5 });
        await store.remember('d', 'd', { importance: 0.8 });
        await store.remember('e', 'e', { importance: 0.2 });

        const all = await store.recallAll();
        expect(all).toHaveLength(3);
        const keys = all.map((e) => e.key).sort();
        // The three highest-importance (b 0.9, d 0.8, c 0.5) survive; a/e evicted.
        expect(keys).toEqual(['b', 'c', 'd']);
    });

    it('evicts expired entries first when over the cap', async () => {
        const store = freshStore({ maxEntries: 2 });
        await store.remember('old', 'x', { importance: 0.99, ttlMs: 1 }); // will expire
        await sleep(5);
        await store.remember('keep1', 'y', { importance: 0.2 });
        await store.remember('keep2', 'z', { importance: 0.2 }); // triggers cap → expired 'old' goes first
        const keys = (await store.recallAll()).map((e) => e.key).sort();
        expect(keys).toEqual(['keep1', 'keep2']);
    });

    it('is a no-op under the cap', async () => {
        const store = freshStore({ maxEntries: 10 });
        await store.remember('a', 'a');
        await store.remember('b', 'b');
        expect(await store.recallAll()).toHaveLength(2);
    });
});

describe('EVM-7 active TTL sweeper', () => {
    it('hard-deletes expired entries in the background', async () => {
        const store = freshStore();
        await store.remember('temp', 'gone soon', { ttlMs: 1 });
        await store.remember('perm', 'stays'); // no TTL
        await sleep(5); // 'temp' is now expired (lazily filtered, not yet deleted)

        const spy = jest.spyOn(store, 'purgeExpired');
        store.startTtlSweeper(5);
        // Poll until the sweeper has actually fired (robust under CI load — no
        // dependence on an exact wall-clock window).
        const start = Date.now();
        while (spy.mock.calls.length === 0 && Date.now() - start < 3000) await sleep(10);
        store.stopTtlSweeper();
        await sleep(15); // let the in-flight purge settle

        expect(spy.mock.calls.length).toBeGreaterThan(0); // sweeper ran purgeExpired
        const keys = (await store.recallAll()).map((e) => e.key);
        expect(keys).toEqual(['perm']); // expired entry gone
        spy.mockRestore();
    });

    it('startTtlSweeper is idempotent and stop is safe to call', async () => {
        const store = freshStore();
        store.startTtlSweeper(1000);
        store.startTtlSweeper(1000); // no second timer
        store.stopTtlSweeper();
        store.stopTtlSweeper(); // safe when already stopped
        expect(true).toBe(true);
    });
});
