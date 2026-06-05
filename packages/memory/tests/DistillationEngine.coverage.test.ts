/**
 * tests/DistillationEngine.coverage.test.ts
 * Covers the quality-gate branches and the in-memory distillation log not
 * exercised by DistillationEngine.test.ts.
 */

import { jest } from '@jest/globals';
import { DistillationEngine } from '../src/distillation/DistillationEngine.js';
import type { SSMRuntime } from '../src/runtime/SSMRuntime.js';
import type { TransformerBridge } from '../src/bridges/TransformerBridge.js';

function makeRuntime(overrides: Partial<SSMRuntime> = {}): SSMRuntime {
    return {
        adapt   : jest.fn<any>().mockResolvedValue({ losses: [0.2], epochCount: 3, durationMs: 1 }),
        evaluate: jest.fn<any>().mockResolvedValue(50),
        ...overrides,
    } as unknown as SSMRuntime;
}

function makeBridge(output = 'a sufficiently long teacher response'): TransformerBridge {
    return {
        generate        : jest.fn<any>().mockResolvedValue(output),
        supportsStreaming: true,
    } as unknown as TransformerBridge;
}

// ── Quality gate: minLength ───────────────────────────────────────────────────

test('quality gate skips adaptation when teacher output is shorter than minLength', async () => {
    const runtime = makeRuntime();
    const engine  = new DistillationEngine(runtime, makeBridge('short'));

    const result = await engine.distill('prompt', { qualityGate: { minLength: 100 } });

    expect(result.skipped).toBe(true);
    expect(result.skipReason).toBe('low_quality');
    expect(result.adaptResult.epochCount).toBe(0);
    expect(runtime.adapt).not.toHaveBeenCalled();
});

// ── Quality gate: maxPerplexity (already learned) ─────────────────────────────

test('quality gate skips adaptation when SSM perplexity is already below maxPerplexity', async () => {
    const runtime = makeRuntime({ evaluate: jest.fn<any>().mockResolvedValue(10) });
    const engine  = new DistillationEngine(runtime, makeBridge());

    const result = await engine.distill('prompt', { qualityGate: { maxPerplexity: 40 } });

    expect(result.skipped).toBe(true);
    expect(result.skipReason).toBe('already_learned');
    expect(runtime.adapt).not.toHaveBeenCalled();
});

test('quality gate adapts when perplexity is above maxPerplexity', async () => {
    const runtime = makeRuntime({ evaluate: jest.fn<any>().mockResolvedValue(90) });
    const engine  = new DistillationEngine(runtime, makeBridge());

    const result = await engine.distill('prompt', { qualityGate: { maxPerplexity: 40 } });

    expect(result.skipped).toBe(false);
    expect(runtime.adapt).toHaveBeenCalled();
});

test('maxPerplexity gate proceeds with adaptation when evaluate() throws (non-fatal)', async () => {
    const runtime = makeRuntime({ evaluate: jest.fn<any>().mockRejectedValue(new Error('eval failed')) });
    const engine  = new DistillationEngine(runtime, makeBridge());

    const result = await engine.distill('prompt', { qualityGate: { maxPerplexity: 40 } });

    expect(result.skipped).toBe(false);
    expect(runtime.adapt).toHaveBeenCalled();
});

// ── Error wrapping with non-Error throwables ──────────────────────────────────

test('a non-Error thrown by the teacher bridge is stringified into DISTILL_FAILED', async () => {
    const bridge = { generate: jest.fn<any>().mockRejectedValue('plain string failure'), supportsStreaming: true } as unknown as TransformerBridge;
    const engine = new DistillationEngine(makeRuntime(), bridge);
    await expect(engine.distill('p')).rejects.toMatchObject({ code: 'DISTILL_FAILED' });
});

test('a non-Error thrown by runtime.adapt is stringified into DISTILL_FAILED', async () => {
    const runtime = makeRuntime({ adapt: jest.fn<any>().mockRejectedValue(42) });
    const engine = new DistillationEngine(runtime, makeBridge());
    await expect(engine.distill('p')).rejects.toMatchObject({ code: 'DISTILL_FAILED' });
});

// ── Distillation log ──────────────────────────────────────────────────────────

test('getLog records a successful distillation with final loss + epochs', async () => {
    const engine = new DistillationEngine(makeRuntime(), makeBridge());
    await engine.distill('prompt');

    const log = engine.getLog();
    expect(log).toHaveLength(1);
    expect(log[0]).toMatchObject({ skipped: false, epochs: 3, finalLoss: 0.2 });
    expect(typeof log[0].timestamp).toBe('number');
});

test('getLog records a skipped distillation with its reason', async () => {
    const engine = new DistillationEngine(makeRuntime(), makeBridge('x'));
    await engine.distill('p', { qualityGate: { minLength: 100 } });
    expect(engine.getLog()[0]).toMatchObject({ skipped: true, skipReason: 'low_quality', epochs: 0 });
});

test('the distillation log is capped at 200 entries (oldest dropped)', async () => {
    const engine = new DistillationEngine(makeRuntime(), makeBridge());
    for (let i = 0; i < 205; i++) await engine.distill(`p${i}`);

    const log = engine.getLog();
    expect(log).toHaveLength(200);
    // The first five inputs were evicted; the newest is last.
    expect(log[log.length - 1].input).toBe('p204');
    expect(log.some(e => e.input === 'p0')).toBe(false);
});
