/**
 * tests/SemanticCachingBridge.test.ts
 * The semantic caching decorator over a TransformerBridge.
 */

import { jest } from '@jest/globals';
import { SemanticCachingBridge } from '../src/bridges/SemanticCachingBridge.js';
import { SemanticCache } from '../src/cache/SemanticCache.js';
import type { TransformerBridge } from '../src/bridges/TransformerBridge.js';

const VEC: Record<string, Float32Array> = {
    'fix the auth bug':  new Float32Array([1, 0]),
    'auth login broken': new Float32Array([0.97, 0.24]), // paraphrase of the above
    'what is 2 + 2':     new Float32Array([0, 1]),
};
const embed = async (t: string): Promise<Float32Array> => VEC[t] ?? new Float32Array([0.01, 0.02]);

function makeInner(generate: (p: string) => Promise<string>, supportsStreaming = true): TransformerBridge & { generate: jest.Mock } {
    return {
        supportsStreaming,
        generate: jest.fn(generate),
        async *stream() { yield 'tok'; },
    } as unknown as TransformerBridge & { generate: jest.Mock };
}

test('a paraphrased prompt is served from cache without re-calling the inner bridge', async () => {
    let n = 0;
    const inner = makeInner(async () => `answer ${++n}`);
    const bridge = new SemanticCachingBridge(inner, { embed });

    const a = await bridge.generate('fix the auth bug');
    const b = await bridge.generate('auth login broken'); // ~0.97 cosine → hit

    expect(a).toBe('answer 1');
    expect(b).toBe('answer 1');                 // reused, not 'answer 2'
    expect(inner.generate).toHaveBeenCalledTimes(1);
});

test('a semantically different prompt calls the inner bridge again', async () => {
    let n = 0;
    const inner = makeInner(async () => `answer ${++n}`);
    const bridge = new SemanticCachingBridge(inner, { embed });

    await bridge.generate('fix the auth bug');
    const b = await bridge.generate('what is 2 + 2'); // 0 cosine → miss
    expect(b).toBe('answer 2');
    expect(inner.generate).toHaveBeenCalledTimes(2);
});

test('the system prompt is part of the cache key (different system → no cross-hit)', async () => {
    let n = 0;
    const inner = makeInner(async () => `answer ${++n}`);
    // Embed system+prompt; distinct system strings yield the fallback vector but
    // differ, so two different systems with the same prompt should not collide.
    const customEmbed = async (t: string): Promise<Float32Array> =>
        t.startsWith('SYS-A') ? new Float32Array([1, 0]) : new Float32Array([0, 1]);
    const bridge = new SemanticCachingBridge(inner, { embed: customEmbed });

    await bridge.generate('hello', { systemPrompt: 'SYS-A' });
    const b = await bridge.generate('hello', { systemPrompt: 'SYS-B' }); // different system → miss
    expect(b).toBe('answer 2');
    expect(inner.generate).toHaveBeenCalledTimes(2);
});

test('the model is stored as cache meta and the answer is reused on a paraphrase', async () => {
    let n = 0;
    const inner = makeInner(async () => `answer ${++n}`);
    const bridge = new SemanticCachingBridge(inner, { embed });

    const a = await bridge.generate('fix the auth bug', { model: 'claude-haiku-4-5' });
    const b = await bridge.generate('auth login broken', { model: 'claude-haiku-4-5' });
    expect(a).toBe('answer 1');
    expect(b).toBe('answer 1');
    expect(inner.generate).toHaveBeenCalledTimes(1);
});

test('supportsStreaming mirrors the inner bridge and stream() delegates', async () => {
    const bridge = new SemanticCachingBridge(makeInner(async () => 'r'), { embed });
    expect(bridge.supportsStreaming).toBe(true);
    const tokens: string[] = [];
    for await (const t of bridge.stream('x')) tokens.push(t);
    expect(tokens).toEqual(['tok']);
});

test('stream() throws when the inner bridge cannot stream', () => {
    const inner = { supportsStreaming: false, generate: jest.fn<any>(async () => 'r') } as unknown as TransformerBridge;
    const bridge = new SemanticCachingBridge(inner, { embed });
    expect(bridge.supportsStreaming).toBe(false);
    expect(() => bridge.stream('x')).toThrow(/does not support streaming/);
});

test('a shared SemanticCache instance is reused across bridges', async () => {
    const cache = new SemanticCache({ embed });
    let n = 0;
    const a = new SemanticCachingBridge(makeInner(async () => `a${++n}`), { embed, cache });
    const b = new SemanticCachingBridge(makeInner(async () => `b${++n}`), { embed, cache });

    const first = await a.generate('fix the auth bug');
    const second = await b.generate('auth login broken'); // hits the shared cache from `a`
    expect(second).toBe(first);
    expect(a.cache).toBe(cache);
});
