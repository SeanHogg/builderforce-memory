/**
 * tests/SSMRuntime.create.test.ts
 * Covers the SSMRuntime.create() factory and the top-level SSM.create() helper
 * by mocking the WebGPU session module so no GPU is required.
 */

import { jest } from '@jest/globals';

const fakeSession = {
    complete: jest.fn<any>(async () => 'r'),
    completeStream: jest.fn<any>(async function* () { yield 'r'; }),
    evaluate: jest.fn<any>(async () => 5),
    destroy: jest.fn<any>(),
    get internals() { return {}; },
};
const createMock = jest.fn<any>(async () => fakeSession);

// Mock the session module so MambaSession.create resolves to a fake (no GPU).
jest.unstable_mockModule('../src/session/index.js', () => ({
    MambaSession: { create: createMock },
    SessionError: class SessionError extends Error {},
    MODEL_PRESETS: {},
    resolveLayerSchedule: () => [],
    resolveModelConfig: () => ({}),
}));

const { SSMRuntime } = await import('../src/runtime/SSMRuntime.js');
const { SSM } = await import('../src/index.js');

// ── SSMRuntime.create ─────────────────────────────────────────────────────────

test('create() builds a runtime with an attached bridge and an auto router', async () => {
    const bridge = { supportsStreaming: true, generate: jest.fn<any>(async () => 'b') };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const runtime = await SSMRuntime.create({ session: { modelSize: 'nano' } as any, bridge: bridge as any });

    expect(createMock).toHaveBeenCalled();
    expect(runtime.bridge).toBe(bridge);
    expect(runtime.destroyed).toBe(false);

    // A short, non-complex input is inconclusive on the cheap heuristics, so auto
    // routing invokes the perplexity probe — which calls session.evaluate (the
    // wired-up probe arrow). evaluate() returns 5 (< threshold) → stays on SSM.
    await runtime.generate('hello there');
    expect(fakeSession.evaluate).toHaveBeenCalledWith('hello there');
    expect(runtime.getRoutingAuditLog().at(-1)?.decision.reason).toBe('complexity');
});

test('create() builds a bridge-less runtime (router short-circuits to SSM)', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const runtime = await SSMRuntime.create({ session: {} as any });
    expect(runtime.bridge).toBeUndefined();

    const audit = await runtime.generate('hi').then(() => runtime.getRoutingAuditLog());
    expect(audit[0].decision.target).toBe('ssm');
    expect(audit[0].decision.reason).toBe('no_bridge');
});

// ── SSM.create namespace helper ───────────────────────────────────────────────

test('SSM.create is a thin shorthand for SSMRuntime.create', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const runtime = await SSM.create({ session: {} as any });
    expect(runtime).toBeInstanceOf(SSMRuntime);
    expect(runtime.destroyed).toBe(false);
});
