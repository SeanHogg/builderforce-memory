/**
 * tests/memorystore-evm8.test.ts — EVM-8: bounded LRU embedding cache.
 */

import 'fake-indexeddb/auto';
import { jest } from '@jest/globals';
import { MemoryStore } from '../src/memory/MemoryStore.js';

let _db = 0;
const freshStore = (opts: Record<string, unknown> = {}) =>
    new MemoryStore({ dbName: `evm8-${_db++}-${Date.now()}`, ...opts });

function countingRuntime() {
    const calls: Record<string, number> = {};
    const embed = jest.fn<any>(async (t: string) => {
        calls[t] = (calls[t] ?? 0) + 1;
        return new Float32Array([t.charCodeAt(0), 1]);
    });
    return { runtime: { embed }, calls };
}

test('embedding cache evicts the least-recently-used entry past embedCacheMax', async () => {
    // Cache holds 2 vectors; query + 2 facts = 3 distinct texts → eviction occurs.
    const store = freshStore({ embedCacheMax: 2 });
    await store.remember('a', 'a');
    await store.remember('b', 'b');
    const { runtime, calls } = countingRuntime();

    await store.recallSimilar('q', 2, runtime); // embeds q,a,b — 'q' evicted (LRU)
    await store.recallSimilar('q', 2, runtime); // 'q' must be re-embedded

    expect(calls['q']).toBeGreaterThanOrEqual(2);
});

test('a large cache keeps everything — no re-embedding across calls', async () => {
    const store = freshStore({ embedCacheMax: 100 });
    await store.remember('a', 'a');
    await store.remember('b', 'b');
    const { runtime, calls } = countingRuntime();

    await store.recallSimilar('q', 2, runtime);
    await store.recallSimilar('q', 2, runtime);

    // Every distinct text embedded exactly once (fully cached).
    expect(calls['q']).toBe(1);
    expect(calls['a']).toBe(1);
    expect(calls['b']).toBe(1);
});
