/**
 * tests/FetchSemanticCacheBackend.test.ts
 * The shared (L2) tier client — HTTP shape, response parsing, and namespace.
 */

import { jest } from '@jest/globals';
import { FetchSemanticCacheBackend } from '../src/cache/FetchSemanticCacheBackend.js';

function jsonRes(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

// ── lookup ────────────────────────────────────────────────────────────────────

test('lookup POSTs the embedding + threshold and returns a valid hit', async () => {
    const fetchImpl = jest.fn<any>(async () => jsonRes({ hit: { response: 'cached', score: 0.97 } }));
    const backend = new FetchSemanticCacheBackend({ baseUrl: 'https://api.builderforce.ai/', apiKey: 'bfk_x', namespace: 'agent-1', fetchImpl });

    const hit = await backend.lookup(new Float32Array([0.5, 0.5]), 0.9);
    expect(hit).toEqual({ response: 'cached', score: 0.97 });

    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://api.builderforce.ai/v1/semantic-cache/lookup'); // trailing slash trimmed
    expect((init.headers as Record<string, string>)['Authorization']).toBe('Bearer bfk_x');
    const body = JSON.parse(init.body as string);
    expect(body).toEqual({ embedding: [0.5, 0.5], threshold: 0.9, namespace: 'agent-1' });
});

test('lookup returns undefined on a non-OK response', async () => {
    const fetchImpl = jest.fn<any>(async () => new Response('no', { status: 500 }));
    const backend = new FetchSemanticCacheBackend({ baseUrl: 'https://x', apiKey: 'k', fetchImpl });
    expect(await backend.lookup(new Float32Array([1]), 0.9)).toBeUndefined();
});

test('lookup returns undefined when the body has no usable hit', async () => {
    const backend = (body: unknown) => new FetchSemanticCacheBackend({
        baseUrl: 'https://x', apiKey: 'k', fetchImpl: jest.fn<any>(async () => jsonRes(body)),
    });
    expect(await backend({}).lookup(new Float32Array([1]), 0.9)).toBeUndefined();                       // no hit
    expect(await backend({ hit: { score: 0.9 } }).lookup(new Float32Array([1]), 0.9)).toBeUndefined();  // no response
    expect(await backend({ hit: { response: 'x' } }).lookup(new Float32Array([1]), 0.9)).toBeUndefined(); // no score
});

test('lookup returns undefined when the body is not JSON', async () => {
    const fetchImpl = jest.fn<any>(async () => new Response('<html>', { status: 200 }));
    const backend = new FetchSemanticCacheBackend({ baseUrl: 'https://x', apiKey: 'k', fetchImpl });
    expect(await backend.lookup(new Float32Array([1]), 0.9)).toBeUndefined();
});

// ── store ─────────────────────────────────────────────────────────────────────

test('store POSTs the embedding, response, namespace and meta', async () => {
    const fetchImpl = jest.fn<any>(async () => jsonRes({ ok: true }));
    const backend = new FetchSemanticCacheBackend({ baseUrl: 'https://x', apiKey: 'k', namespace: 'ns', fetchImpl });

    // Use Float32-exact values (0.5, 0.25) so Array.from round-trips without
    // floating-point drift.
    await backend.store(new Float32Array([0.5, 0.25]), 'answer', { model: 'claude' });

    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://x/v1/semantic-cache/store');
    expect(JSON.parse(init.body as string)).toEqual({
        embedding: [0.5, 0.25], response: 'answer', namespace: 'ns', meta: { model: 'claude' },
    });
});

test('constructs with the global fetch when no fetchImpl is injected', () => {
    const backend = new FetchSemanticCacheBackend({ baseUrl: 'https://x/', apiKey: 'k' });
    expect(backend).toBeInstanceOf(FetchSemanticCacheBackend);
});

test('store omits namespace and meta when not provided', async () => {
    const fetchImpl = jest.fn<any>(async () => jsonRes({ ok: true }));
    const backend = new FetchSemanticCacheBackend({ baseUrl: 'https://x', apiKey: 'k', fetchImpl });
    await backend.store(new Float32Array([1]), 'r');
    const body = JSON.parse((fetchImpl.mock.calls[0][1] as RequestInit).body as string);
    expect(body).toEqual({ embedding: [1], response: 'r' });
});
