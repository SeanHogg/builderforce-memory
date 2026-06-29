/**
 * tests/distillation-rehearsal.test.ts — EVM-5: catastrophic-forgetting guard
 * (experience-replay / rehearsal buffer) in the DistillationEngine.
 */

import { jest } from '@jest/globals';
import { DistillationEngine } from '../src/distillation/DistillationEngine.js';
import type { SSMRuntime } from '../src/runtime/SSMRuntime.js';
import type { TransformerBridge } from '../src/bridges/TransformerBridge.js';

function makeRuntime(adaptSpy: (text: string) => void): SSMRuntime {
    return {
        generate: jest.fn<any>(),
        stream: jest.fn<any>(),
        adapt: jest.fn<any>(async (text: string) => { adaptSpy(text); return { losses: [0.4], epochCount: 1 }; }),
        evaluate: jest.fn<any>(),
        save: jest.fn<any>(),
        load: jest.fn<any>(),
        destroy: jest.fn<any>(),
        get bridge() { return undefined; },
        get destroyed() { return false; },
        get internals() { return {} as never; },
        streamHybrid: jest.fn<any>(),
    } as unknown as SSMRuntime;
}

/** Bridge whose teacher output echoes the input so we can trace it in adapt text. */
function makeBridge(): TransformerBridge {
    return {
        generate: jest.fn<any>(async (input: string) => `answer:${input}`),
        stream: jest.fn<any>(),
        supportsStreaming: true,
    };
}

test('rehearsal is disabled by default — buffer stays empty, nothing rehearsed', async () => {
    const texts: string[] = [];
    const engine = new DistillationEngine(makeRuntime((t) => texts.push(t)), makeBridge());
    const r = await engine.distill('q1');
    expect(r.rehearsed).toBe(0);
    expect(engine.getRehearsalBufferSize()).toBe(0);
    expect(texts[0]).toBe('q1\nanswer:q1'); // no rehearsed pairs prepended
});

test('rehearsal replays past exemplars into later adapt passes', async () => {
    const texts: string[] = [];
    const engine = new DistillationEngine(
        makeRuntime((t) => texts.push(t)),
        makeBridge(),
        { bufferSize: 10, sampleK: 2, seed: 5 },
    );

    await engine.distill('q1'); // buffer empty during this adapt → no rehearsal
    await engine.distill('q2'); // q1 available to rehearse
    const r3 = await engine.distill('q3'); // q1,q2 available

    // First pass had nothing to rehearse.
    expect(texts[0]).toBe('q1\nanswer:q1');
    // Later passes interleave past exemplars before the new one.
    expect(r3.rehearsed).toBeGreaterThanOrEqual(1);
    expect(texts[2]).toContain('q3\nanswer:q3'); // new exemplar still present
    expect(texts[2]).toMatch(/q1\nanswer:q1|q2\nanswer:q2/); // at least one past exemplar replayed
    expect(texts[2]!.includes('\n\n')).toBe(true); // pairs joined with a blank line
});

test('buffer is a ring bounded by bufferSize', async () => {
    const engine = new DistillationEngine(makeRuntime(() => {}), makeBridge(), { bufferSize: 2, sampleK: 1 });
    await engine.distill('a');
    await engine.distill('b');
    await engine.distill('c');
    await engine.distill('d');
    expect(engine.getRehearsalBufferSize()).toBe(2); // capped
});

test('sampleK=0 keeps a buffer but rehearses nothing', async () => {
    const texts: string[] = [];
    const engine = new DistillationEngine(makeRuntime((t) => texts.push(t)), makeBridge(), { bufferSize: 5, sampleK: 0 });
    await engine.distill('x');
    const r = await engine.distill('y');
    expect(r.rehearsed).toBe(0);
    expect(engine.getRehearsalBufferSize()).toBe(2);
    expect(texts[1]).toBe('y\nanswer:y');
});

test('skipped (quality-gated) distills do not populate the rehearsal buffer', async () => {
    const engine = new DistillationEngine(makeRuntime(() => {}), makeBridge(), { bufferSize: 5 });
    const r = await engine.distill('short', { qualityGate: { minLength: 1000 } });
    expect(r.skipped).toBe(true);
    expect(engine.getRehearsalBufferSize()).toBe(0);
});
