/**
 * tests/bridges.coverage.test.ts
 * Edge branches across the bridge layer:
 *   - Anthropic / OpenAI streaming with a missing response body
 *   - FetchBridge default credentials/model
 *   - CachingBridge.stream guard for non-streaming inner bridges
 *   - ResponseCache TTL expiry, clear, and size
 */

import { jest } from '@jest/globals';
import { AnthropicBridge } from '../src/bridges/AnthropicBridge.js';
import { OpenAIBridge } from '../src/bridges/OpenAIBridge.js';
import { FetchBridge } from '../src/bridges/FetchBridge.js';
import { CachingBridge } from '../src/bridges/CachingBridge.js';
import { ResponseCache } from '../src/bridges/ResponseCache.js';
import type { TransformerBridge } from '../src/bridges/TransformerBridge.js';

function okNoBody(): Response {
    // 200 OK with a null body → exercises the "no body" stream guard.
    return new Response(null, { status: 200 });
}

// ── streaming guards: missing response body ───────────────────────────────────

test('AnthropicBridge.stream throws BRIDGE_RESPONSE_INVALID when the response has no body', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(okNoBody());
    const gen = new AnthropicBridge({ apiKey: 'k' }).stream('hi')[Symbol.asyncIterator]();
    await expect(gen.next()).rejects.toMatchObject({ code: 'BRIDGE_RESPONSE_INVALID' });
    fetchSpy.mockRestore();
});

test('OpenAIBridge.stream throws BRIDGE_RESPONSE_INVALID when the response has no body', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(okNoBody());
    const gen = new OpenAIBridge({ apiKey: 'k' }).stream('hi')[Symbol.asyncIterator]();
    await expect(gen.next()).rejects.toMatchObject({ code: 'BRIDGE_RESPONSE_INVALID' });
    fetchSpy.mockRestore();
});

// ── SSE parsers: malformed / non-text data lines are skipped ──────────────────

function sseResponse(lines: string[]): Response {
    return new Response(
        new ReadableStream({
            start(c) {
                const enc = new TextEncoder();
                for (const l of lines) c.enqueue(enc.encode(l + '\n'));
                c.close();
            },
        }),
        { status: 200 },
    );
}

test('AnthropicBridge.stream skips malformed SSE JSON and still yields valid deltas', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(sseResponse([
        'event: message_start',                                                        // non-`data:` line → skipped
        '',                                                                            // blank line → skipped
        'data: {not valid json',                                                       // malformed → caught + skipped
        'data: ' + JSON.stringify({ type: 'content_block_delta', delta: { text: 'A' } }),
        'data: ' + JSON.stringify({ type: 'content_block_delta', delta: { text: '' } }), // empty text → not yielded
        'data: ' + JSON.stringify({ type: 'message_stop' }),
    ]));

    const tokens: string[] = [];
    for await (const t of new AnthropicBridge({ apiKey: 'k' }).stream('hi')) tokens.push(t);
    expect(tokens).toEqual(['A']);
    fetchSpy.mockRestore();
});

test('OpenAIBridge.stream skips malformed SSE JSON and stops at [DONE]', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(sseResponse([
        ': openrouter comment',                                                           // non-`data:` line → skipped
        'data: {broken',                                                                  // malformed → skipped
        'data: ' + JSON.stringify({ choices: [{ delta: { content: 'X' } }] }),
        'data: [DONE]',
        'data: ' + JSON.stringify({ choices: [{ delta: { content: 'after-done' } }] }),  // never reached
    ]));

    const tokens: string[] = [];
    for await (const t of new OpenAIBridge({ apiKey: 'k' }).stream('hi')) tokens.push(t);
    expect(tokens).toEqual(['X']);
    fetchSpy.mockRestore();
});

test('OpenAIBridge.stream ends cleanly when the upstream closes without a [DONE] sentinel', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(sseResponse([
        'data: ' + JSON.stringify({ choices: [{ delta: { content: 'only' } }] }),
        // no `data: [DONE]` — the reader reaches end-of-stream and breaks on `done`.
    ]));

    const tokens: string[] = [];
    for await (const t of new OpenAIBridge({ apiKey: 'k' }).stream('hi')) tokens.push(t);
    expect(tokens).toEqual(['only']);
    fetchSpy.mockRestore();
});

// ── error paths where the error body itself is unreadable (.text() rejects) ────

function errResUnreadableText(status = 500): Response {
    const r = new Response('body', { status });
    Object.defineProperty(r, 'text', { value: () => Promise.reject(new Error('unreadable')) });
    return r;
}

test('AnthropicBridge.generate degrades gracefully when the error body cannot be read', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(errResUnreadableText());
    await expect(new AnthropicBridge({ apiKey: 'k' }).generate('hi'))
        .rejects.toMatchObject({ code: 'BRIDGE_REQUEST_FAILED' });
    fetchSpy.mockRestore();
});

test('AnthropicBridge.stream degrades gracefully when the error body cannot be read', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(errResUnreadableText(403));
    const gen = new AnthropicBridge({ apiKey: 'k' }).stream('hi')[Symbol.asyncIterator]();
    await expect(gen.next()).rejects.toMatchObject({ code: 'BRIDGE_REQUEST_FAILED' });
    fetchSpy.mockRestore();
});

test('OpenAIBridge.generate degrades gracefully when the error body cannot be read', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(errResUnreadableText());
    await expect(new OpenAIBridge({ apiKey: 'k' }).generate('hi'))
        .rejects.toMatchObject({ code: 'BRIDGE_REQUEST_FAILED' });
    fetchSpy.mockRestore();
});

test('OpenAIBridge.stream degrades gracefully when the error body cannot be read', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(errResUnreadableText(429));
    const gen = new OpenAIBridge({ apiKey: 'k' }).stream('hi')[Symbol.asyncIterator]();
    await expect(gen.next()).rejects.toMatchObject({ code: 'BRIDGE_REQUEST_FAILED' });
    fetchSpy.mockRestore();
});

// ── FetchBridge defaults ──────────────────────────────────────────────────────

test('FetchBridge defaults apiKey to "local" and model to "default"', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), {
            status: 200, headers: { 'Content-Type': 'application/json' },
        }),
    );

    const bridge = new FetchBridge({ baseUrl: 'http://localhost:1234/v1' });
    expect(bridge.supportsStreaming).toBe(true);
    const out = await bridge.generate('hi');
    expect(out).toBe('ok');

    const init = fetchSpy.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)['Authorization']).toBe('Bearer local');
    expect(JSON.parse(init.body as string).model).toBe('default');
    fetchSpy.mockRestore();
});

// ── CachingBridge.stream guard ────────────────────────────────────────────────

test('CachingBridge.stream throws when the wrapped bridge does not support streaming', () => {
    const inner = {
        supportsStreaming: false,
        generate: jest.fn<any>(async () => 'r'),
    } as unknown as TransformerBridge;
    const bridge = new CachingBridge(inner);

    expect(bridge.supportsStreaming).toBe(false);
    expect(() => bridge.stream('hi')).toThrow(/does not support streaming/);
});

// ── ResponseCache TTL + maintenance ───────────────────────────────────────────

test('ResponseCache treats an entry older than ttlMs as a miss and evicts it', () => {
    const cache = new ResponseCache({ ttlMs: 1 });
    cache.set('k', 'v', Date.now() - 1000); // inserted "in the past"
    expect(cache.get('k')).toBeUndefined();  // expired → miss + eviction
    expect(cache.size).toBe(0);
    expect(cache.stats.misses).toBe(1);
});

test('ResponseCache without a TTL keeps entries until evicted by capacity', () => {
    const cache = new ResponseCache(); // no ttl
    cache.set('k', 'v', Date.now());
    expect(cache.get('k')).toBe('v');
    expect(cache.stats.hits).toBe(1);
});

test('ResponseCache.clear empties the store', () => {
    const cache = new ResponseCache();
    cache.set('a', '1', Date.now());
    cache.set('b', '2', Date.now());
    expect(cache.size).toBe(2);
    cache.clear();
    expect(cache.size).toBe(0);
    expect(cache.get('a')).toBeUndefined();
});

test('ResponseCache.set on an existing key updates the value in place', () => {
    const cache = new ResponseCache();
    cache.set('k', 'first', Date.now());
    cache.set('k', 'second', Date.now()); // re-set existing key → delete-then-set branch
    expect(cache.get('k')).toBe('second');
    expect(cache.size).toBe(1);
});
