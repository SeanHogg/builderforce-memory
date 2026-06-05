/**
 * tests/SSMRuntime.coverage.test.ts
 * Covers the runtime's adaptation / persistence pass-throughs, accessors,
 * lifecycle (destroy + post-destroy guards), and the streamHybrid routing
 * branches not exercised by SSMRuntime.test.ts.
 */

import { jest } from '@jest/globals';
import { SSMRuntime } from '../src/runtime/SSMRuntime.js';
import { InferenceRouter } from '../src/router/InferenceRouter.js';
import type { TransformerBridge } from '../src/bridges/TransformerBridge.js';

function makeFakeSession() {
    return {
        complete: jest.fn<any>(async () => 'r'),
        completeStream: jest.fn<any>(async function* () { yield 's'; }),
        evaluate: jest.fn<any>(async () => 42),
        embed: jest.fn<any>(async () => new Float32Array([1, 2])),
        adapt: jest.fn<any>(async () => ({ losses: [0.1], epochCount: 2, durationMs: 1 })),
        save: jest.fn<any>(async () => undefined),
        load: jest.fn<any>(async () => true),
        destroy: jest.fn<any>(),
        get internals() { return { marker: 'internals' }; },
    };
}

function makeRuntime(opts: { bridge?: TransformerBridge; strategy?: 'ssm' | 'transformer' } = {}) {
    const session = makeFakeSession();
    const router = new InferenceRouter({
        strategy : opts.strategy ?? (opts.bridge ? 'transformer' : 'ssm'),
        hasBridge: !!opts.bridge,
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const runtime = new (SSMRuntime as any)(session, opts.bridge, router) as SSMRuntime;
    return { runtime, session };
}

// ── Pass-throughs ─────────────────────────────────────────────────────────────

test('adapt / evaluate / embed / save / load delegate to the session', async () => {
    const { runtime, session } = makeRuntime();

    expect(await runtime.adapt('data', { epochs: 2 })).toMatchObject({ epochCount: 2 });
    expect(session.adapt).toHaveBeenCalledWith('data', { epochs: 2 });

    expect(await runtime.evaluate('text')).toBe(42);
    expect(session.evaluate).toHaveBeenCalledWith('text');

    expect(Array.from(await runtime.embed('text'))).toEqual([1, 2]);
    expect(session.embed).toHaveBeenCalledWith('text');

    await runtime.save({ key: 'w' });
    expect(session.save).toHaveBeenCalledWith({ key: 'w' });

    expect(await runtime.load({ key: 'w' })).toBe(true);
    expect(session.load).toHaveBeenCalledWith({ key: 'w' });
});

// ── Accessors ─────────────────────────────────────────────────────────────────

test('accessors expose bridge, destroyed flag, internals, and the routing audit log', async () => {
    const bridge = { supportsStreaming: true, generate: jest.fn<any>(async () => 'x') } as unknown as TransformerBridge;
    const { runtime } = makeRuntime({ bridge, strategy: 'transformer' });

    expect(runtime.bridge).toBe(bridge);
    expect(runtime.destroyed).toBe(false);
    expect(runtime.internals).toMatchObject({ marker: 'internals' });
    expect(runtime.getDistillationLog()).toEqual([]);

    await runtime.generate('hi'); // produces one routing audit entry
    const audit = runtime.getRoutingAuditLog();
    expect(audit).toHaveLength(1);
    expect(audit[0].decision.target).toBe('transformer');
});

// ── streamHybrid routing ──────────────────────────────────────────────────────

test('streamHybrid uses the SSM path when routing stays on the SSM', async () => {
    const { runtime, session } = makeRuntime({ strategy: 'ssm' });
    const out: string[] = [];
    for await (const t of runtime.streamHybrid('input', { system: 'SYS' })) out.push(t);
    expect(out).toEqual(['s']);
    expect((session.completeStream.mock.calls[0] as [string, unknown])[0]).toBe('SYS\ninput');
});

test('streamHybrid falls back to generate() for a transformer bridge that cannot stream', async () => {
    const bridge = {
        supportsStreaming: false,                       // no stream support
        generate: jest.fn<any>(async () => 'whole-answer'),
    } as unknown as TransformerBridge;
    const { runtime } = makeRuntime({ bridge, strategy: 'transformer' });

    const out: string[] = [];
    for await (const t of runtime.streamHybrid('input', { system: 'SYS' })) out.push(t);

    expect(out).toEqual(['whole-answer']);                       // single yield of the full text
    expect(bridge.generate).toHaveBeenCalledWith('input', expect.objectContaining({ systemPrompt: 'SYS' }));
});

// ── Lifecycle ─────────────────────────────────────────────────────────────────

test('destroy() tears down the session and flips destroyed', () => {
    const { runtime, session } = makeRuntime();
    runtime.destroy();
    expect(session.destroy).toHaveBeenCalledTimes(1);
    expect(runtime.destroyed).toBe(true);
});

test('every inference method throws RUNTIME_DESTROYED after destroy()', async () => {
    const { runtime } = makeRuntime();
    runtime.destroy();

    await expect(runtime.generate('x')).rejects.toMatchObject({ code: 'RUNTIME_DESTROYED' });
    await expect(runtime.adapt('x')).rejects.toMatchObject({ code: 'RUNTIME_DESTROYED' });
    await expect(runtime.evaluate('x')).rejects.toMatchObject({ code: 'RUNTIME_DESTROYED' });
    await expect(runtime.embed('x')).rejects.toMatchObject({ code: 'RUNTIME_DESTROYED' });
    await expect(runtime.save()).rejects.toMatchObject({ code: 'RUNTIME_DESTROYED' });
    await expect(runtime.load()).rejects.toMatchObject({ code: 'RUNTIME_DESTROYED' });

    const streamAfter = runtime.stream('x')[Symbol.asyncIterator]();
    await expect(streamAfter.next()).rejects.toMatchObject({ code: 'RUNTIME_DESTROYED' });

    const hybridAfter = runtime.streamHybrid('x')[Symbol.asyncIterator]();
    await expect(hybridAfter.next()).rejects.toMatchObject({ code: 'RUNTIME_DESTROYED' });
});
