/**
 * The two ports a {@link DistillationEngine} runs between.
 *
 * Distillation is one recipe — teacher answers, gate, rehearse, adapt, log — over
 * any teacher and any student. The package's own pairing (a `TransformerBridge`
 * teaching an `SSMRuntime`) is one instance; a host that teaches through a metered
 * gateway and adapts a packaged checkpoint into a mergeable diff is another. Both
 * plug in here instead of restating the recipe.
 */

import type { AdaptOptions, AdaptResult } from '../session/index.js';
import type { SSMRuntime } from '../runtime/SSMRuntime.js';
import type { BridgeGenerateOptions } from '../bridges/TransformerBridge.js';

/** Produces the exemplar for an input. Any `TransformerBridge` is one. */
export interface DistillationTeacher {
    /** The exemplar for `input`. Throws when there is none; the engine records the error. */
    generate(input: string, opts?: BridgeGenerateOptions): Promise<string>;
}

/**
 * Learns from a training text. `R` is whatever one adaptation yields (an `AdaptResult`,
 * a checkpoint diff, …); `O` the per-call options it takes.
 */
export interface DistillationStudent<R, O = undefined> {
    adapt(trainingText: string, opts?: O): Promise<R>;
    /** Perplexity on `text` — enables the `maxPerplexity` gate. Optional. */
    evaluate?(text: string): Promise<number>;
    /** The result recorded when nothing was adapted (the gate skipped the input). */
    skippedResult(): R;
    /** Final loss and epoch count of one result, for the log and batch totals. */
    describe(result: R): { finalLoss?: number; epochs: number };
}

/** The adapt options an {@link SSMRuntime} student uses unless a call overrides them. */
export const SSM_STUDENT_DEFAULT_ADAPT: AdaptOptions = { wsla: true, epochs: 3 };

/**
 * An {@link SSMRuntime} as a student. WSLA by default: it is fast and targets the
 * selective-projection rows, exactly the parameters that encode token routing.
 */
export function ssmRuntimeStudent(runtime: SSMRuntime): DistillationStudent<AdaptResult, AdaptOptions> {
    return {
        adapt: (text, opts) => runtime.adapt(text, { ...SSM_STUDENT_DEFAULT_ADAPT, ...opts }),
        evaluate: (text) => runtime.evaluate(text),
        skippedResult: () => ({ losses: [], epochCount: 0, durationMs: 0 }),
        describe: (r) => ({ finalLoss: r.losses.at(-1), epochs: r.epochCount }),
    };
}

/** True when `x` already speaks the student port (an `SSMRuntime` does not: it has no `describe`). */
export function isDistillationStudent<R, O>(x: unknown): x is DistillationStudent<R, O> {
    return !!x && typeof (x as { describe?: unknown }).describe === 'function'
        && typeof (x as { skippedResult?: unknown }).skippedResult === 'function';
}
