/**
 * tests/CachingBridge.test.ts
 * Unit tests for CachingBridge + ResponseCache — read-through caching of
 * transformer completions.
 */

import { jest } from '@jest/globals';
import { CachingBridge } from '../src/bridges/CachingBridge.js';
import { ResponseCache, buildCacheKey } from '../src/bridges/ResponseCache.js';
import type { TransformerBridge, BridgeGenerateOptions } from '../src/bridges/TransformerBridge.js';

// ── Mock inner bridge ───────────────────────────────────────────────────────

function makeInner(
    generate: (p: string, o?: BridgeGenerateOptions) => Promise<string>,
    supportsStreaming = true,
): TransformerBridge & { generate: jest.Mock } {
    return {
        supportsStreaming,
        generate: jest.fn(generate),
        async *stream() { yield 'tok'; },
    } as unknown as TransformerBridge & { generate: jest.Mock };
}

// ── CachingBridge.generate ──────────────────────────────────────────────────

test('identical requests hit the inner bridge only once', async () => {
    let calls = 0;
    const inner  = makeInner(async () => `reply ${++calls}`);
    const bridge = new CachingBridge(inner);

    const first  = await bridge.generate('hello');
    const second = await bridge.generate('hello');

    expect(first).toBe('reply 1');
    expect(second).toBe('reply 1');           // served from cache, not 'reply 2'
    expect(inner.generate).toHaveBeenCalledTimes(1);
});

test('different prompts are cached separately', async () => {
    let calls = 0;
    const inner  = makeInner(async () => `reply ${++calls}`);
    const bridge = new CachingBridge(inner);

    expect(await bridge.generate('a')).toBe('reply 1');
    expect(await bridge.generate('b')).toBe('reply 2');
    expect(inner.generate).toHaveBeenCalledTimes(2);
});

test('a differing option (model) is a separate cache entry', async () => {
    let calls = 0;
    const inner  = makeInner(async () => `reply ${++calls}`);
    const bridge = new CachingBridge(inner);

    await bridge.generate('x', { model: 'm1' });
    await bridge.generate('x', { model: 'm2' });
    await bridge.generate('x', { model: 'm1' }); // hit

    expect(inner.generate).toHaveBeenCalledTimes(2);
});

test('cache stats track hits and misses', async () => {
    const inner  = makeInner(async () => 'r');
    const bridge = new CachingBridge(inner);

    await bridge.generate('p'); // miss
    await bridge.generate('p'); // hit

    expect(bridge.cache.stats).toEqual({ hits: 1, misses: 1 });
});

test('supportsStreaming mirrors the inner bridge', () => {
    expect(new CachingBridge(makeInner(async () => 'r', false)).supportsStreaming).toBe(false);
    expect(new CachingBridge(makeInner(async () => 'r', true)).supportsStreaming).toBe(true);
});

test('stream delegates to the inner bridge', async () => {
    const bridge = new CachingBridge(makeInner(async () => 'r'));
    const tokens: string[] = [];
    for await (const t of bridge.stream('hi')) tokens.push(t);
    expect(tokens).toEqual(['tok']);
});

// ── ResponseCache ───────────────────────────────────────────────────────────

test('ResponseCache evicts least-recently-used entries past maxEntries', () => {
    const cache = new ResponseCache({ maxEntries: 2 });
    cache.set('a', '1', 0);
    cache.set('b', '2', 0);
    cache.get('a');            // 'a' becomes most-recently-used
    cache.set('c', '3', 0);    // evicts 'b' (LRU), not 'a'

    expect(cache.get('a')).toBe('1');
    expect(cache.get('b')).toBeUndefined();
    expect(cache.get('c')).toBe('3');
});

test('buildCacheKey is order-stable and field-sensitive', () => {
    const base = { prompt: 'p', model: 'm', systemPrompt: 's', maxTokens: 10 };
    expect(buildCacheKey(base)).toBe(buildCacheKey({ ...base }));
    expect(buildCacheKey(base)).not.toBe(buildCacheKey({ ...base, systemPrompt: 't' }));
});

test('a shared ResponseCache is reused across bridges', async () => {
    const cache = new ResponseCache();
    let calls = 0;
    const a = new CachingBridge(makeInner(async () => `r${++calls}`), { cache });
    const b = new CachingBridge(makeInner(async () => `r${++calls}`), { cache });

    const first  = await a.generate('shared');
    const second = await b.generate('shared'); // hits the shared entry from `a`

    expect(second).toBe(first);
});
