/**
 * tests/MemoryStore.coverage.test.ts
 * Covers the MemoryStore surface not exercised by MemoryStore.test.ts:
 * recallRecent / recallByTag / recallSimilar (both embedding + Jaccard paths),
 * export/import, purgeExpired, weight persistence, TTL, and the no-IndexedDB
 * environment guard.
 */

import 'fake-indexeddb/auto';
import { jest } from '@jest/globals';
import { MemoryStore } from '../src/memory/MemoryStore.js';

let _db = 0;
const freshStore = (opts: Record<string, unknown> = {}) =>
    new MemoryStore({ dbName: `cov-${_db++}`, ...opts });

// ── recallRecent / recallByTag ────────────────────────────────────────────────

test('recallRecent returns the N most recent facts, newest first', async () => {
    const store = freshStore();
    await store.remember('a', '1');
    await store.remember('b', '2');
    await store.remember('c', '3');

    const recent = await store.recallRecent(2);
    expect(recent.map(e => e.key)).toEqual(['c', 'b']);
});

test('recallByTag returns only entries carrying the tag', async () => {
    const store = freshStore();
    await store.remember('x', 'tagged',   { tags: ['lang', 'fav'] });
    await store.remember('y', 'untagged');
    await store.remember('z', 'other',    { tags: ['fav'] });

    const fav = await store.recallByTag('fav');
    expect(fav.map(e => e.key).sort()).toEqual(['x', 'z']);
    expect(await store.recallByTag('nope')).toEqual([]);
});

// ── recallSimilar ─────────────────────────────────────────────────────────────

test('recallSimilar on an empty store returns []', async () => {
    expect(await freshStore().recallSimilar('q', 3)).toEqual([]);
});

test('recallSimilar falls back to Jaccard word overlap when no runtime is given', async () => {
    const store = freshStore();
    await store.remember('k1', 'the quick brown fox');
    await store.remember('k2', 'completely unrelated content');

    const [best] = await store.recallSimilar('quick brown fox jumps', 1);
    expect(best.key).toBe('k1');
});

test('recallSimilar uses SSM embeddings when the runtime exposes embed()', async () => {
    const store = freshStore();
    await store.remember('match', 'aligned');
    await store.remember('miss',  'orthogonal');

    // Fake embed: "aligned" and the query point the same way; "orthogonal" doesn't.
    const vectors: Record<string, Float32Array> = {
        'query':      new Float32Array([1, 0]),
        'aligned':    new Float32Array([1, 0]),
        'orthogonal': new Float32Array([0, 1]),
    };
    const runtime = { embed: jest.fn<any>(async (t: string) => vectors[t] ?? new Float32Array([0, 0])) };

    const [top] = await store.recallSimilar('query', 1, runtime);
    expect(top.key).toBe('match');
    // Second call hits the per-content embedding cache (no re-embed of cached text).
    const callsAfterFirst = runtime.embed.mock.calls.length;
    await store.recallSimilar('query', 1, runtime);
    expect(runtime.embed.mock.calls.length).toBe(callsAfterFirst); // fully cached
});

test('recallSimilar falls back to Jaccard when embed() returns an unusable vector', async () => {
    const store = freshStore();
    await store.remember('k1', 'alpha beta gamma');
    await store.remember('k2', 'delta');

    // embed returns empty arrays → _embedWithCache yields null → Jaccard fallback.
    const runtime = { embed: jest.fn<any>(async () => new Float32Array(0)) };
    const [best] = await store.recallSimilar('alpha beta', 1, runtime);
    expect(best.key).toBe('k1');
});

test('recallSimilar falls back to Jaccard when embed() throws', async () => {
    const store = freshStore();
    await store.remember('k1', 'one two three');
    await store.remember('k2', 'nine');

    const runtime = { embed: jest.fn<any>(async () => { throw new Error('no gpu'); }) };
    const [best] = await store.recallSimilar('one two', 1, runtime);
    expect(best.key).toBe('k1');
});

// ── export / import ───────────────────────────────────────────────────────────

test('exportAll returns all non-expired entries', async () => {
    const store = freshStore();
    await store.remember('a', '1');
    await store.remember('b', '2');
    expect((await store.exportAll()).map(e => e.key).sort()).toEqual(['a', 'b']);
});

test('importAll overwrite writes every entry unconditionally', async () => {
    const src = freshStore();
    await src.remember('k', 'new');
    const exported = await src.exportAll();

    const dst = freshStore();
    await dst.remember('k', 'old');
    await dst.importAll(exported, 'overwrite');
    expect((await dst.recall('k'))?.content).toBe('new');
});

test('importAll merge only writes when incoming is newer or the key is missing', async () => {
    const dst = freshStore();
    await dst.remember('keep', 'current');
    const existing = await dst.recall('keep');

    await dst.importAll([
        // older timestamp for an existing key → skipped
        { key: 'keep', content: 'stale', timestamp: (existing!.timestamp) - 1000 },
        // brand-new key → written
        { key: 'fresh', content: 'added', timestamp: Date.now() },
    ], 'merge');

    expect((await dst.recall('keep'))?.content).toBe('current');
    expect((await dst.recall('fresh'))?.content).toBe('added');
});

// ── TTL ───────────────────────────────────────────────────────────────────────

test('an expired entry is not returned by recall and is dropped from recallAll', async () => {
    const store = freshStore();
    await store.remember('temp', 'gone', { ttlMs: 1 });
    await new Promise(r => setTimeout(r, 10));
    expect(await store.recall('temp')).toBeUndefined();
    expect(await store.recallAll()).toEqual([]);
});

test('purgeExpired hard-deletes expired rows and returns the count', async () => {
    const store = freshStore();
    await store.remember('live', 'stays');
    await store.remember('dead', 'goes', { ttlMs: 1 });
    await new Promise(r => setTimeout(r, 10));

    expect(await store.purgeExpired()).toBe(1);
    expect(await store.purgeExpired()).toBe(0); // nothing left to purge
    expect((await store.recallAll()).map(e => e.key)).toEqual(['live']);
});

test('defaultTtlMs is applied to new entries when no per-entry ttl is given', async () => {
    const store = freshStore({ defaultTtlMs: 1 });
    await store.remember('k', 'v');
    await new Promise(r => setTimeout(r, 10));
    expect(await store.recall('k')).toBeUndefined();
});

// ── forget / clear ────────────────────────────────────────────────────────────

test('forget removes a single fact and is a no-op for a missing key', async () => {
    const store = freshStore();
    await store.remember('a', '1');
    await store.forget('a');
    await store.forget('never-existed'); // no throw
    expect(await store.recall('a')).toBeUndefined();
});

test('clear removes all facts', async () => {
    const store = freshStore();
    await store.remember('a', '1');
    await store.remember('b', '2');
    await store.clear();
    expect(await store.recallAll()).toEqual([]);
});

// ── weight persistence (delegates to the runtime) ─────────────────────────────

test('saveWeights / loadWeights delegate to the runtime with the configured key', async () => {
    const store = freshStore({ weightsKey: 'wk' });
    const runtime = {
        save: jest.fn<any>(async () => undefined),
        load: jest.fn<any>(async () => true),
    };
    await store.saveWeights(runtime);
    expect(runtime.save).toHaveBeenCalledWith({ storage: 'indexedDB', key: 'wk' });

    expect(await store.loadWeights(runtime)).toBe(true);
    expect(runtime.load).toHaveBeenCalledWith({ key: 'wk' });
});

// ── environment guard ─────────────────────────────────────────────────────────

// ── constructor defaults + injected factory ───────────────────────────────────

test('MemoryStore() uses the default db name and weights key', async () => {
    const store = new MemoryStore(); // exercise the `?? 'ssmjs'` / `?? 'ssmjs-weights'` defaults
    await store.remember('default-key', 'v');
    expect((await store.recall('default-key'))?.content).toBe('v');
    await store.forget('default-key'); // tidy the shared default DB
});

test('an injected idbFactory is used instead of the global indexedDB', async () => {
    const { IDBFactory } = await import('fake-indexeddb');
    const store = new MemoryStore({ dbName: 'injected', idbFactory: new IDBFactory() });
    await store.remember('k', 'v');
    expect((await store.recall('k'))?.content).toBe('v');
});

// ── recallSimilar branch edges ────────────────────────────────────────────────

test('recallSimilar falls back to Jaccard when one entry fails to embed mid-loop', async () => {
    const store = freshStore();
    await store.remember('good', 'embeddable text');
    await store.remember('bad',  'unembeddable');

    // query + 'good' embed fine; 'bad' yields an empty vector → embeddedAll=false → break → Jaccard.
    const vectors: Record<string, Float32Array> = {
        q: new Float32Array([1]),               // the query string embeds fine …
        'embeddable text': new Float32Array([1]), // … and so does 'good'
        // 'unembeddable' (the 'bad' entry) is absent → empty vector → embed returns null mid-loop.
    };
    const runtime = { embed: jest.fn<any>(async (t: string) => vectors[t] ?? new Float32Array(0)) };

    const res = await store.recallSimilar('q', 2, runtime);
    expect(res.map(e => e.key).sort()).toEqual(['bad', 'good']);
});

test('recallSimilar handles a zero embedding (cosine denominator 0 → score 0)', async () => {
    const store = freshStore();
    await store.remember('zero', 'z');
    await store.remember('one',  'o');

    const vectors: Record<string, Float32Array> = {
        query: new Float32Array([1, 0]),
        z: new Float32Array([0, 0]), // zero vector → cosine denom 0
        o: new Float32Array([1, 0]),
    };
    const runtime = { embed: jest.fn<any>(async (t: string) => vectors[t] ?? new Float32Array([0, 0])) };

    const [top] = await store.recallSimilar('query', 1, runtime);
    expect(top.key).toBe('one'); // the aligned, non-zero vector wins
});

test('recallSimilar Jaccard treats two empty token sets as identical (similarity 1)', async () => {
    const store = freshStore();
    await store.remember('blank', '');     // tokenizes to an empty set
    const [only] = await store.recallSimilar('', 1); // empty query too
    expect(only.key).toBe('blank');
});

test('recallSimilar clears the embedding cache once it exceeds its bound', async () => {
    const store = freshStore();
    // EMBED_CACHE_MAX is 2000; 2001 distinct contents in one pass crosses the bound.
    for (let i = 0; i < 2001; i++) await store.remember(`k${i}`, `content number ${i}`);
    const runtime = { embed: jest.fn<any>(async (t: string) => new Float32Array([t.length, 1])) };
    const res = await store.recallSimilar('query', 1, runtime);
    expect(res).toHaveLength(1); // completed the embedding pass without error
}, 30_000);

test('opening without an IndexedDB factory rejects with MEMORY_UNAVAILABLE', async () => {
    const saved = (globalThis as { indexedDB?: unknown }).indexedDB;
    // Force the "no IDB available" branch: no injected factory + no global.
    delete (globalThis as { indexedDB?: unknown }).indexedDB;
    try {
        const store = new MemoryStore({ dbName: 'no-idb' });
        await expect(store.remember('k', 'v')).rejects.toMatchObject({ code: 'MEMORY_UNAVAILABLE' });
    } finally {
        (globalThis as { indexedDB?: unknown }).indexedDB = saved;
    }
});
