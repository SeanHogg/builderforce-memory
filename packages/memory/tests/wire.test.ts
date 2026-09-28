/**
 * tests/wire.test.ts
 * SSE framing and usage reading — the wire rules the bridges and the gateway share.
 */

import { parseSseDataLine, parseSseDataFrames, readSseDataFrames, readUsageFields, AnthropicStreamUsage, finiteNumber } from '../src/wire/index.js';

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
    const enc = new TextEncoder();
    return new ReadableStream({
        start(controller) {
            for (const c of chunks) controller.enqueue(enc.encode(c));
            controller.close();
        },
    });
}

describe('parseSseDataLine', () => {
    it('accepts both the spaced and the spaceless data frame', () => {
        expect(parseSseDataLine('data: {"a":1}')).toEqual({ a: 1 });
        expect(parseSseDataLine('data:{"a":2}')).toEqual({ a: 2 });
    });

    it('skips non-data lines, the DONE sentinel, empty and malformed payloads', () => {
        expect(parseSseDataLine('event: ping')).toBeUndefined();
        expect(parseSseDataLine('data: [DONE]')).toBeUndefined();
        expect(parseSseDataLine('data:   ')).toBeUndefined();
        expect(parseSseDataLine('data: {oops')).toBeUndefined();
    });
});

describe('parseSseDataFrames / readSseDataFrames', () => {
    const body = 'event: x\ndata: {"n":1}\n\ndata:{"n":2}\ndata: [DONE]\n';

    it('yields every usable frame of a buffered body and stops at [DONE]', () => {
        expect([...parseSseDataFrames(body)]).toEqual([{ n: 1 }, { n: 2 }]);
        expect([...parseSseDataFrames('data: {"n":1}\ndata: [DONE]\ndata: {"n":9}\n')]).toEqual([{ n: 1 }]);
    });

    it('yields the same frames from a stream split mid-line, including an unterminated last line', async () => {
        const out: unknown[] = [];
        for await (const f of readSseDataFrames(streamOf(['data: {"n"', ':1}\n\nda', 'ta:{"n":2}\ndata: {"n":3}']))) out.push(f);
        expect(out).toEqual([{ n: 1 }, { n: 2 }, { n: 3 }]);
    });

    it('ends a live stream at [DONE]', async () => {
        const out: unknown[] = [];
        for await (const f of readSseDataFrames(streamOf(['data: {"n":1}\ndata: [DONE]\ndata: {"n":2}\n']))) out.push(f);
        expect(out).toEqual([{ n: 1 }]);
    });
});

describe('readUsageFields', () => {
    it('reads the OpenAI shape, cached reads from prompt_tokens_details', () => {
        expect(readUsageFields({ prompt_tokens: 100, completion_tokens: 20, total_tokens: 120, prompt_tokens_details: { cached_tokens: 60 } }))
            .toEqual({ promptTokens: 100, completionTokens: 20, totalTokens: 120, cacheReadTokens: 60 });
    });

    it('reads the Anthropic shape and keeps absent distinct from zero', () => {
        expect(readUsageFields({ input_tokens: 5, output_tokens: 0, cache_read_input_tokens: 7, cache_creation_input_tokens: 3 }))
            .toEqual({ promptTokens: 5, completionTokens: 0, cacheReadTokens: 7, cacheCreationTokens: 3 });
        expect(readUsageFields(null)).toEqual({});
        expect(finiteNumber('x')).toBeUndefined();
    });
});

describe('AnthropicStreamUsage', () => {
    it('takes input and cache from message_start and the latest cumulative output from message_delta', () => {
        const u = new AnthropicStreamUsage();
        const body = [
            'data: {"type":"message_start","message":{"model":"claude-x","usage":{"input_tokens":12,"cache_read_input_tokens":40,"cache_creation_input_tokens":8,"output_tokens":1}}}',
            'data: {"type":"content_block_delta","delta":{"text":"hi"}}',
            'data: {"type":"message_delta","usage":{"output_tokens":9}}',
            'data: {"type":"message_delta","usage":{"output_tokens":21}}',
        ].join('\n');
        for (const f of parseSseDataFrames(body)) u.observe(f);
        expect(u.result()).toEqual({ inputTokens: 12, outputTokens: 21, cacheReadTokens: 40, cacheCreationTokens: 8, model: 'claude-x' });
    });
});
