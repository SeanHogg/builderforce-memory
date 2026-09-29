/**
 * tests/chatClient.test.ts
 * The `/wire` chat-completions client — the request body it sends, its error mapping,
 * and the assembled stream result (text + tool-call deltas + usage).
 */

import { jest } from '@jest/globals';
import { chatComplete, chatStream, ChatCompletionError, type ChatStreamEvent } from '../src/wire/index.js';

const client = { baseUrl: 'https://gw.example/v1/', apiKey: 'k' };

function jsonResponse(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

function sseResponse(frames: string[]): Response {
    const enc = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
        start(controller) {
            for (const f of frames) controller.enqueue(enc.encode(f));
            controller.close();
        },
    });
    return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } });
}

function sentBody(spy: { mock: { calls: unknown[][] } }): Record<string, unknown> {
    const init = spy.mock.calls[0][1] as RequestInit;
    return JSON.parse(init.body as string);
}

afterEach(() => jest.restoreAllMocks());

describe('chatComplete', () => {
    it('posts to {baseUrl}/chat/completions with only what the request states', async () => {
        const spy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(jsonResponse({
            model: 'm-used',
            usage: { prompt_tokens: 3 },
            choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
        }));
        const result = await chatComplete(client, { messages: [{ role: 'user', content: 'q' }] });

        expect(spy.mock.calls[0][0]).toBe('https://gw.example/v1/chat/completions');
        const init = spy.mock.calls[0][1] as RequestInit;
        expect((init.headers as Record<string, string>).authorization).toBe('Bearer k');
        // No sampling defaults, no model, no stream fields.
        expect(sentBody(spy)).toEqual({ messages: [{ role: 'user', content: 'q' }] });
        expect(result).toEqual({
            content: 'hi',
            hasTextContent: true,
            toolCalls: [],
            finishReason: 'stop',
            model: 'm-used',
            usage: { prompt_tokens: 3 },
        });
    });

    it('maps request fields onto the wire and merges vendor extras last', async () => {
        const spy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(jsonResponse({ choices: [{ message: { content: '' } }] }));
        const tools = [{ type: 'function' as const, function: { name: 't', description: 'd', parameters: {} } }];
        await chatComplete(client, {
            messages: [{ role: 'user', content: 'q' }],
            tools,
            model: 'm',
            maxTokens: 10,
            temperature: 0,
            topP: 0.5,
            includeUsage: true, // ignored off-stream
            extra: { provider: 'p', temperature: 0.2 },
        });
        expect(sentBody(spy)).toEqual({
            model: 'm',
            messages: [{ role: 'user', content: 'q' }],
            tools,
            tool_choice: 'auto',
            max_tokens: 10,
            temperature: 0.2,
            top_p: 0.5,
            provider: 'p',
        });
    });

    it('reports a tool-call-only turn as no text content', async () => {
        const calls = [{ id: 'c1', type: 'function', function: { name: 't', arguments: '{}' } }];
        jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(jsonResponse({
            choices: [{ message: { content: null, tool_calls: calls }, finish_reason: 'tool_calls' }],
        }));
        const result = await chatComplete(client, { messages: [] });
        expect(result.hasTextContent).toBe(false);
        expect(result.content).toBe('');
        expect(result.toolCalls).toEqual(calls);
        expect(result.finishReason).toBe('tool_calls');
    });

    it('throws ChatCompletionError with the status and a body capped at 300 chars', async () => {
        jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(new Response('x'.repeat(1000), { status: 429 }));
        const err = await chatComplete(client, { messages: [] }).catch((e: unknown) => e);
        expect(err).toBeInstanceOf(ChatCompletionError);
        expect((err as ChatCompletionError).status).toBe(429);
        expect((err as ChatCompletionError).body).toHaveLength(300);
    });

    it('treats an unparseable 2xx body as an empty reply', async () => {
        jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(new Response('not json', { status: 200 }));
        const result = await chatComplete(client, { messages: [] });
        expect(result).toEqual({ content: '', hasTextContent: false, toolCalls: [], finishReason: undefined });
    });

    it('passes an abort signal combining the caller signal and the client timeout', async () => {
        const spy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(jsonResponse({ choices: [] }));
        const caller = new AbortController();
        await chatComplete({ ...client, timeoutMs: 60_000 }, { messages: [] }, caller.signal);
        const signal = (spy.mock.calls[0][1] as RequestInit).signal as AbortSignal;
        expect(signal).toBeDefined();
        expect(signal.aborted).toBe(false);
        caller.abort();
        expect(signal.aborted).toBe(true);
    });
});

describe('chatStream', () => {
    it('yields text and tool-call deltas, then one done event with the assembled result', async () => {
        const spy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(sseResponse([
            'data: {"model":"m1","choices":[{"delta":{"content":"Hel"}}]}\n',
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"run","arguments":"{\\"a\\""}}]}}]}\n',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]},"finish_reason":"tool_calls"}]}\n',
            'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n',
            'data: [DONE]\n',
        ]));
        const events: ChatStreamEvent[] = [];
        for await (const e of chatStream(client, { messages: [], includeUsage: true })) events.push(e);

        expect(sentBody(spy)).toEqual({ messages: [], stream: true, stream_options: { include_usage: true } });
        expect((spy.mock.calls[0][1] as RequestInit).headers).toMatchObject({ accept: 'text/event-stream' });
        expect(events.filter((e) => e.type === 'text-delta')).toEqual([
            { type: 'text-delta', delta: 'Hel' },
            { type: 'text-delta', delta: 'lo' },
        ]);
        expect(events.filter((e) => e.type === 'tool-call')).toHaveLength(2);
        expect(events[events.length - 1]).toEqual({
            type: 'done',
            result: {
                content: 'Hello',
                hasTextContent: true,
                finishReason: 'tool_calls',
                toolCalls: [{ id: 'c1', type: 'function', function: { name: 'run', arguments: '{"a":1}' } }],
                model: 'm1',
                usage: { prompt_tokens: 5, completion_tokens: 2 },
            },
        });
    });

    it('sends no stream_options unless usage is asked for', async () => {
        const spy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(sseResponse(['data: [DONE]\n']));
        const events: ChatStreamEvent[] = [];
        for await (const e of chatStream(client, { messages: [] })) events.push(e);
        expect(sentBody(spy)).toEqual({ messages: [], stream: true });
        expect(events).toEqual([{ type: 'done', result: { content: '', hasTextContent: false, finishReason: undefined, toolCalls: [] } }]);
    });

    it('throws ChatCompletionError on a non-2xx answer before any event', async () => {
        jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(new Response('overloaded', { status: 503 }));
        const it = chatStream(client, { messages: [] });
        await expect(it.next()).rejects.toMatchObject({ name: 'ChatCompletionError', status: 503, body: 'overloaded' });
    });
});
