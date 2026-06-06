/**
 * tests/SemanticCache.test.ts
 * The embedding-keyed read-through cache: paraphrase hits, the L1/L2 tiers,
 * best-effort L2, thresholds, TTL, LRU, and stats.
 *
 * A deterministic fake embedder maps known strings to fixed vectors so cosine
 * similarity is fully controllable:
 *   A     = [1, 0]
 *   Anear = [0.98, 0.2]   (~0.98 cosine to A — a "paraphrase")
 *   B     = [0, 1]        (0 cosine to A — unrelated)
 */

import { jest } from '@jest/globals';
import { SemanticCache, type SemanticCacheBackend } from '../src/cache/SemanticCache.js';

const VEC: Record<string, Float32Array> = {
    A:     new Float32Array([1, 0]),
    Anear: new Float32Array([0.98, 0.2]),
    B:     new Float32Array([0, 1]),
    C:     new Float32Array([0.3, 0.95]),
};
const embed = async (t: string): Promise<Float32Array> => VEC[t] ?? new Float32Array([0.01, 0.01]);

// ── L1: paraphrase hit / miss ─────────────────────────────────────────────────

test('getOrGenerate misses on first call then serves a paraphrase from L1', async () => {
    const cache = new SemanticCache({ embed });
    const gen = jest.fn<any>(async () => 'the answer');

    const first = await cache.getOrGenerate('A', gen);
    expect(first).toMatchObject({ response: 'the answer', cached: false });
    expect(gen).toHaveBeenCalledTimes(1);

    const second = await cache.getOrGenerate('Anear', gen); // ~0.98 cosine → hit
    expect(second.cached).toBe(true);
    expect(second.tier).toBe('l1');
    expect(second.response).toBe('the answer');
    expect(gen).toHaveBeenCalledTimes(1); // NOT regenerated — tokens saved
});

test('getOrGenerate misses (and regenerates) when nothing is within threshold', async () => {
    const cache = new SemanticCache({ embed });
    await cache.getOrGenerate('A', async () => 'a-answer');

    const gen = jest.fn<any>(async () => 'b-answer');
    const r = await cache.getOrGenerate('B', gen); // 0 cosine → miss
    expect(r.cached).toBe(false);
    expect(gen).toHaveBeenCalledTimes(1);
    expect(cache.stats).toMatchObject({ l1Hits: 0, misses: 2 });
});

test('lookup() and store() work independently of getOrGenerate', async () => {
    const cache = new SemanticCache({ embed });
    expect(await cache.lookup('A')).toBeUndefined(); // empty

    await cache.store('A', 'stored');
    const hit = await cache.lookup('Anear');
    expect(hit?.response).toBe('stored');
    expect(hit?.tier).toBe('l1');
    expect(hit!.score).toBeGreaterThanOrEqual(0.92);
});

test('a stricter threshold rejects a borderline paraphrase', async () => {
    const cache = new SemanticCache({ embed, threshold: 0.999 });
    await cache.store('A', 'x');
    expect(await cache.lookup('Anear')).toBeUndefined(); // 0.98 < 0.999
});

// ── L2 shared tier ────────────────────────────────────────────────────────────

function makeL2(overrides: Partial<SemanticCacheBackend> = {}): SemanticCacheBackend & { lookup: jest.Mock; store: jest.Mock } {
    return {
        lookup: jest.fn<any>(async () => ({ response: 'from-l2', score: 0.99 })),
        store:  jest.fn<any>(async () => undefined),
        ...overrides,
    } as SemanticCacheBackend & { lookup: jest.Mock; store: jest.Mock };
}

test('an L1 miss falls through to L2 and warms L1 for next time', async () => {
    const l2 = makeL2();
    const cache = new SemanticCache({ embed, l2 });

    const gen = jest.fn<any>(async () => 'generated');
    const first = await cache.getOrGenerate('A', gen);
    expect(first).toMatchObject({ cached: true, tier: 'l2', response: 'from-l2' });
    expect(gen).not.toHaveBeenCalled();   // L2 served it
    expect(l2.lookup).toHaveBeenCalledTimes(1);
    expect(cache.size).toBe(1);           // warmed into L1

    const second = await cache.lookup('Anear');
    expect(second?.tier).toBe('l1');      // now a local hit
});

test('warmL1FromL2:false serves from L2 without populating L1', async () => {
    const cache = new SemanticCache({ embed, l2: makeL2(), warmL1FromL2: false });
    const hit = await cache.lookup('A');
    expect(hit?.tier).toBe('l2');
    expect(cache.size).toBe(0); // not warmed
});

test('an L2 entry below threshold is ignored (treated as a miss)', async () => {
    const l2 = makeL2({ lookup: jest.fn<any>(async () => ({ response: 'weak', score: 0.5 })) });
    const cache = new SemanticCache({ embed, l2 });
    const gen = jest.fn<any>(async () => 'fresh');
    const r = await cache.getOrGenerate('A', gen);
    expect(r.cached).toBe(false);
    expect(gen).toHaveBeenCalledTimes(1);
});

test('an L2 lookup that returns undefined is a miss', async () => {
    const l2 = makeL2({ lookup: jest.fn<any>(async () => undefined) });
    const cache = new SemanticCache({ embed, l2 });
    expect(await cache.lookup('A')).toBeUndefined();
});

test('L2 is best-effort: a throwing lookup degrades to a local miss', async () => {
    const l2 = makeL2({ lookup: jest.fn<any>(async () => { throw new Error('gateway down'); }) });
    const cache = new SemanticCache({ embed, l2 });
    const gen = jest.fn<any>(async () => 'local');
    const r = await cache.getOrGenerate('A', gen);
    expect(r.cached).toBe(false);
    expect(gen).toHaveBeenCalledTimes(1);
});

test('L2 is best-effort: a throwing store does not fail the caller', async () => {
    const l2 = makeL2({ store: jest.fn<any>(async () => { throw new Error('gateway down'); }) });
    const cache = new SemanticCache({ embed, l2 });
    const r = await cache.getOrGenerate('A', async () => 'answer');
    expect(r).toMatchObject({ response: 'answer', cached: false });
    expect(cache.size).toBe(1); // local copy still cached
});

test('store() writes through to the L2 backend', async () => {
    const l2 = makeL2();
    const cache = new SemanticCache({ embed, l2 });
    await cache.store('A', 'shared', { model: 'm' });
    expect(l2.store).toHaveBeenCalledTimes(1);
    const [, response, meta] = l2.store.mock.calls[0] as [Float32Array, string, Record<string, unknown>];
    expect(response).toBe('shared');
    expect(meta).toEqual({ model: 'm' });
});

// ── TTL + LRU + maintenance ───────────────────────────────────────────────────

test('an L1 entry older than ttlMs is treated as a miss and dropped', async () => {
    const cache = new SemanticCache({ embed, ttlMs: 1 });
    await cache.store('A', 'x');
    await new Promise(r => setTimeout(r, 10));
    expect(await cache.lookup('A')).toBeUndefined();
    expect(cache.size).toBe(0); // expired entry pruned during the scan
});

test('L1 is bounded by maxEntries (oldest evicted)', async () => {
    const cache = new SemanticCache({ embed, maxEntries: 1 });
    await cache.store('A', 'a');
    await cache.store('B', 'b'); // distinct vector → both stored, but cap is 1
    expect(cache.size).toBe(1);
    expect(await cache.lookup('A')).toBeUndefined(); // 'A' evicted
    expect((await cache.lookup('B'))?.response).toBe('b');
});

test('clear empties L1; stats track l1/l2 hits and misses', async () => {
    const l2 = makeL2();
    const cache = new SemanticCache({ embed, l2 });

    await cache.getOrGenerate('A', async () => 'x'); // L2 hit (warms L1)
    await cache.lookup('Anear');                     // L1 hit
    const l2b = makeL2({ lookup: jest.fn<any>(async () => undefined) });
    const cache2 = new SemanticCache({ embed, l2: l2b });
    await cache2.lookup('B');                         // miss

    expect(cache.stats.l2Hits).toBe(1);
    expect(cache.stats.l1Hits).toBe(1);
    expect(cache2.stats.misses).toBe(1);

    cache.clear();
    expect(cache.size).toBe(0);
});
