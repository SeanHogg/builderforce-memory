/**
 * tests/memorystore-evm1b.test.ts — EVM-1b: persistent dense vector store + HNSW
 * index. Steady-state recall must embed only NEW/CHANGED facts and reuse the
 * index, not re-embed every candidate per call.
 */

import 'fake-indexeddb/auto';
import { jest } from '@jest/globals';
import { MemoryStore } from '../src/memory/MemoryStore.js';

let _db = 0;
const freshStore = (opts: Record<string, unknown> = {}) =>
    new MemoryStore({ dbName: `evm1b-${_db++}-${Date.now()}`, ...opts });

/** Embedding runtime that counts how many times each content string is embedded. */
function countingRuntime() {
    const calls: Record<string, number> = {};
    const embed = jest.fn<any>(async (t: string) => {
        calls[t] = (calls[t] ?? 0) + 1;
        // Deterministic 2-D unit-ish vector from the first char.
        return new Float32Array([t.charCodeAt(0) || 1, 1]);
    });
    return { runtime: { embed }, calls };
}

test('facts are embedded once, not on every recall (persistent vectors)', async () => {
    const store = freshStore();
    await store.remember('a', 'alpha');
    await store.remember('b', 'beta');
    const { runtime, calls } = countingRuntime();

    await store.recallSimilar('alpha', 2, runtime);
    await store.recallSimilar('alpha', 2, runtime); // 2nd recall must NOT re-embed facts

    expect(calls['alpha']).toBe(1); // query content embedded once (cached)
    expect(calls['beta']).toBe(1);  // fact embedded once across both recalls
});

test('only the NEW fact is embedded when one is added between recalls (delta)', async () => {
    const store = freshStore();
    await store.remember('a', 'alpha');
    const { runtime, calls } = countingRuntime();

    await store.recallSimilar('q1', 5, runtime); // embeds alpha
    await store.remember('b', 'beta');           // new fact
    await store.recallSimilar('q2', 5, runtime); // must embed ONLY beta

    expect(calls['alpha']).toBe(1); // not re-embedded
    expect(calls['beta']).toBe(1);  // embedded once (the delta)
});

test('an EDIT (same key, new content) re-embeds only that fact', async () => {
    const store = freshStore();
    await store.remember('a', 'alpha');
    const { runtime, calls } = countingRuntime();

    await store.recallSimilar('q', 5, runtime);  // embeds alpha
    await store.remember('a', 'alpha-v2');        // edit → seq bumps
    await store.recallSimilar('q', 5, runtime);   // re-embeds alpha-v2 only

    expect(calls['alpha']).toBe(1);
    expect(calls['alpha-v2']).toBe(1);
});

test('changing the runtime re-embeds everything (embeddings are model-specific)', async () => {
    const store = freshStore();
    await store.remember('a', 'alpha');
    const r1 = countingRuntime();
    const r2 = countingRuntime();

    await store.recallSimilar('q', 5, r1.runtime);
    await store.recallSimilar('q', 5, r2.runtime); // different runtime → fresh embeds

    expect(r1.calls['alpha']).toBe(1);
    expect(r2.calls['alpha']).toBe(1); // re-embedded under the new runtime
});

test('above the ANN threshold the persistent HNSW index is reused across recalls', async () => {
    const store = freshStore({ annThreshold: 4 });
    const dirs: Record<string, [number, number]> = {
        query: [1, 0], f0: [1, 0], f1: [0.92, 0.39], f2: [0.71, 0.71],
        f3: [0.39, 0.92], f4: [0, 1], f5: [-0.71, 0.71],
    };
    for (const k of ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']) await store.remember(k, k);
    const embed = jest.fn<any>(async (t: string) => new Float32Array(dirs[t] ?? [0, 0]));
    const runtime = { embed };

    const first = await store.recallSimilar('query', 1, runtime);
    const callsAfterBuild = embed.mock.calls.length;
    const second = await store.recallSimilar('query', 1, runtime);

    expect(first[0]!.key).toBe('f0');
    expect(second[0]!.key).toBe('f0');
    // 2nd recall embeds nothing new — query, facts, and the HNSW index are all reused.
    expect(embed.mock.calls.length).toBe(callsAfterBuild);
});

test('clear() drops the persistent index and recall reflects the empty then rebuilt set', async () => {
    const store = freshStore();
    await store.remember('a', 'alpha');
    const { runtime } = countingRuntime();
    await store.recallSimilar('q', 5, runtime);

    await store.clear();
    expect(await store.recallSimilar('q', 5, runtime)).toEqual([]); // index dropped → no hits

    await store.remember('b', 'beta'); // rebuild from a fresh set
    const hits = await store.recallSimilar('beta', 5, runtime);
    expect(hits.map((e) => e.key)).toEqual(['b']);
});

test('falls back to lexical when a fact cannot be embedded', async () => {
    const store = freshStore();
    await store.remember('k1', 'alpha beta gamma');
    await store.remember('k2', 'delta');
    // Query embeds fine, but facts return empty → _syncDense returns null → Jaccard.
    let n = 0;
    const runtime = { embed: jest.fn<any>(async () => (n++ === 0 ? new Float32Array([1, 1]) : new Float32Array(0))) };
    const [best] = await store.recallSimilar('alpha beta', 1, runtime);
    expect(best.key).toBe('k1');
});
