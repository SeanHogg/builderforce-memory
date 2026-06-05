/**
 * tests/SSMRuntime.test.ts
 * Validates the runtime's system-prefix threading — the consumption pattern that
 * makes transformer-side prompt caching possible:
 *   - SSM path     : `system` is prepended to `input` (trained single-string format)
 *   - transformer  : `system` is sent as the bridge's `systemPrompt` (cacheable),
 *                    and the user message is NOT prepended (no duplication)
 *
 * SSMRuntime's constructor is private (compile-time only). We cast to bypass it
 * so a fake MambaSession + real InferenceRouter can be injected without a GPU.
 */

import { jest } from '@jest/globals';
import { SSMRuntime } from '../src/runtime/SSMRuntime.js';
import { InferenceRouter } from '../src/router/InferenceRouter.js';
import type { TransformerBridge } from '../src/bridges/TransformerBridge.js';

// ── Fakes ───────────────────────────────────────────────────────────────────

function makeFakeSession() {
    return {
        complete: jest.fn<any>(async () => 'ssm-reply'),
        completeStream: jest.fn<any>(async function* () { yield 'ssm'; yield '-tok'; }),
        evaluate: jest.fn<any>(async () => 10),
        embed: jest.fn<any>(async () => new Float32Array([1])),
        adapt: jest.fn<any>(async () => ({ losses: [], epochCount: 0, durationMs: 0 })),
        save: jest.fn<any>(async () => undefined),
        load: jest.fn<any>(async () => false),
        destroy: jest.fn<any>(),
        get internals() { return {}; },
    };
}

function makeBridge(): TransformerBridge & { generate: jest.Mock; stream: jest.Mock } {
    return {
        supportsStreaming: true,
        generate: jest.fn<any>(async () => 'bridge-reply'),
        stream: jest.fn<any>(async function* () { yield 'b'; yield 'r'; }),
    } as unknown as TransformerBridge & { generate: jest.Mock; stream: jest.Mock };
}

function makeRuntime(opts: { bridge?: TransformerBridge; strategy?: 'ssm' | 'transformer' | 'auto' } = {}) {
    const session = makeFakeSession();
    const router = new InferenceRouter({
        strategy : opts.strategy ?? (opts.bridge ? 'transformer' : 'ssm'),
        hasBridge: !!opts.bridge,
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const runtime = new (SSMRuntime as any)(session, opts.bridge, router) as SSMRuntime;
    return { runtime, session };
}

// ── SSM path ──────────────────────────────────────────────────────────────────

test('generate (SSM path) prepends system to input as the trained single-string format', async () => {
    const { runtime, session } = makeRuntime({ strategy: 'ssm' });

    const reply = await runtime.generate('USER', { system: 'SYS', maxNewTokens: 5 });

    expect(reply).toBe('ssm-reply');
    expect(session.complete).toHaveBeenCalledTimes(1);
    const [input, completeOpts] = session.complete.mock.calls[0] as [string, Record<string, unknown>];
    expect(input).toBe('SYS\nUSER');
    // system + bridgeOpts are stripped from the CompleteOptions forwarded to the session
    expect(completeOpts.system).toBeUndefined();
    expect(completeOpts.bridgeOpts).toBeUndefined();
    expect(completeOpts.maxNewTokens).toBe(5);
});

test('generate (SSM path) leaves input unchanged when no system is provided', async () => {
    const { runtime, session } = makeRuntime({ strategy: 'ssm' });
    await runtime.generate('USER', {});
    expect((session.complete.mock.calls[0] as [string, unknown])[0]).toBe('USER');
});

test('generate with no bridge always routes to SSM regardless of system', async () => {
    const { runtime, session } = makeRuntime(); // no bridge → router returns 'ssm'
    await runtime.generate('hi', { system: 'S' });
    expect(session.complete).toHaveBeenCalledTimes(1);
});

// ── Transformer path ───────────────────────────────────────────────────────────

test('generate (transformer path) sends system as systemPrompt and does NOT prepend it to the user message', async () => {
    const bridge = makeBridge();
    const { runtime, session } = makeRuntime({ bridge, strategy: 'transformer' });

    const reply = await runtime.generate('USER', { system: 'SYS' });

    expect(reply).toBe('bridge-reply');
    expect(bridge.generate).toHaveBeenCalledTimes(1);
    const [prompt, bridgeOpts] = bridge.generate.mock.calls[0] as [string, Record<string, unknown>];
    expect(prompt).toBe('USER');            // NOT 'SYS\nUSER' — no duplication
    expect(bridgeOpts.systemPrompt).toBe('SYS');
    // SSM session is untouched on the transformer path
    expect(session.complete).not.toHaveBeenCalled();
});

test('explicit bridgeOpts.systemPrompt wins over the top-level system', async () => {
    const bridge = makeBridge();
    const { runtime } = makeRuntime({ bridge, strategy: 'transformer' });

    await runtime.generate('USER', { system: 'SYS', bridgeOpts: { systemPrompt: 'OVERRIDE' } });

    const [, bridgeOpts] = bridge.generate.mock.calls[0] as [string, Record<string, unknown>];
    expect(bridgeOpts.systemPrompt).toBe('OVERRIDE');
});

// ── Streaming ──────────────────────────────────────────────────────────────────

test('stream (SSM) prepends system and yields tokens', async () => {
    const { runtime, session } = makeRuntime({ strategy: 'ssm' });

    const tokens: string[] = [];
    for await (const t of runtime.stream('USER', { system: 'SYS' })) tokens.push(t);

    expect(tokens).toEqual(['ssm', '-tok']);
    expect((session.completeStream.mock.calls[0] as [string, unknown])[0]).toBe('SYS\nUSER');
});

test('streamHybrid (transformer) streams via the bridge with systemPrompt set', async () => {
    const bridge = makeBridge();
    const { runtime } = makeRuntime({ bridge, strategy: 'transformer' });

    const tokens: string[] = [];
    for await (const t of runtime.streamHybrid('USER', { system: 'SYS' })) tokens.push(t);

    expect(tokens).toEqual(['b', 'r']);
    const [prompt, bridgeOpts] = bridge.stream.mock.calls[0] as [string, Record<string, unknown>];
    expect(prompt).toBe('USER');
    expect(bridgeOpts.systemPrompt).toBe('SYS');
});
