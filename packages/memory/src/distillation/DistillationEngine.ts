/**
 * DistillationEngine – online knowledge distillation.
 *
 * A *teacher* (a frontier model) writes the ideal response; the *student* is adapted
 * on it. Runs with no Python and no full retraining.
 *
 * Distillation flow:
 *   1. teacher.generate(input)                → exemplar
 *   2. quality gate (length / already-learned) → skip, or fall back to the caller's text
 *   3. student.adapt(input + exemplar, + rehearsed past exemplars)
 *   4. log the outcome
 *
 * Teacher and student are ports ({@link ./ports}). Passing an `SSMRuntime` and a
 * `TransformerBridge` — the original pairing — still works unchanged.
 */

import type { AdaptOptions, AdaptResult } from '../session/index.js';
import type { SSMRuntime } from '../runtime/SSMRuntime.js';
import type { BridgeGenerateOptions } from '../bridges/TransformerBridge.js';
import { SSMError } from '../errors/SSMError.js';
import { isDistillationStudent, ssmRuntimeStudent, type DistillationStudent, type DistillationTeacher } from './ports.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface QualityGate {
    /**
     * Minimum character length of the teacher output.
     * Outputs shorter than this are considered low quality and are skipped.
     */
    minLength?     : number;
    /**
     * Maximum student perplexity threshold (needs a student with `evaluate`).
     * When the student already achieves perplexity below this value on the teacher
     * output, the content is considered already learned and adaptation is skipped.
     */
    maxPerplexity? : number;
}

export interface DistillOptions<O = AdaptOptions> {
    /** Options forwarded to the student's `adapt()`. */
    adapt?       : O;

    /** Options forwarded to the teacher's `generate()` (system prompt, max tokens, …). */
    generate?    : BridgeGenerateOptions;

    /**
     * Quality gate filters that can skip adaptation for low-quality or
     * already-learned inputs.
     */
    qualityGate? : QualityGate;

    /**
     * What to learn when the teacher yields nothing usable — it threw, or its output
     * failed the `minLength` gate. Without it such an input is skipped (a teacher
     * error is thrown); with it the student learns this text instead, so the
     * contribution is never lost, and the result says it was not distilled.
     */
    fallbackText?: string;

    /**
     * Cap on the characters of `input` that prefix the exemplar in the training text,
     * so a long input can never crowd the exemplar out of the student's window.
     */
    contextChars?: number;
}

/**
 * Catastrophic-forgetting guard (EVM-5). Online adaptation moves a narrow set of
 * weights toward the newest exemplar, which can erode previously-learned knowledge.
 * A rehearsal (experience-replay) buffer mitigates this: each adapt also trains on a
 * sample of past exemplars, so old knowledge is continually reinforced.
 */
export interface RehearsalOptions {
    /** Ring-buffer capacity of past exemplars. 0 disables rehearsal. Default 0. */
    bufferSize?: number;
    /** How many past exemplars to mix into each adapt pass. Default 2. */
    sampleK?: number;
    /** Deterministic sampling seed. Default 1. */
    seed?: number;
}

/** Why an input was not learned from a teacher exemplar. */
export type DistillSkipReason = 'low_quality' | 'already_learned' | 'teacher_failed';

export interface DistillResult<R = AdaptResult> {
    /** The input prompt that was distilled. */
    input        : string;
    /** The teacher's response to the input ('' when the teacher failed). */
    teacherOutput: string;
    /** The student's result — its skipped result when nothing was adapted. */
    adaptResult  : R;
    /** True when nothing was adapted at all. */
    skipped?     : boolean;
    /** True when the student learned the teacher's exemplar (not a fallback). */
    distilled    : boolean;
    /** Why the exemplar was not learned, when it was not. */
    skipReason?  : DistillSkipReason;
    /** What the teacher threw, when `skipReason` is `teacher_failed`. */
    teacherError?: unknown;
    /** Number of past exemplars rehearsed alongside this one (EVM-5). */
    rehearsed?   : number;
    /** Student's pre-adapt perplexity on the teacher output, when the quality gate
     *  measured it — a "how novel was this exemplar" signal. */
    gatePerplexity?: number;
}

export interface DistillBatchResult<R = AdaptResult> {
    results    : DistillResult<R>[];
    /** Total number of adapt epochs run across all inputs. */
    totalEpochs: number;
    /** Wall-clock time for the entire batch in milliseconds. */
    totalMs    : number;
}

export interface DistillationLog {
    timestamp          : number;
    input              : string;
    teacherOutputLength: number;
    skipped            : boolean;
    /** True when the exemplar (not a fallback) was learned. */
    distilled          : boolean;
    skipReason?        : DistillSkipReason;
    finalLoss?         : number;
    epochs             : number;
    /** Pre-adapt perplexity on the teacher output (when the quality gate measured it). */
    gatePerplexity?    : number;
}

/** Maximum number of distillation log entries to retain in memory. */
const MAX_LOG_ENTRIES = 200;

// ── DistillationEngine ────────────────────────────────────────────────────────

export class DistillationEngine<R = AdaptResult, O = AdaptOptions> {
    private readonly _student  : DistillationStudent<R, O>;
    private readonly _teacher  : DistillationTeacher;
    private readonly _log      : DistillationLog[] = [];

    // ── Rehearsal buffer (EVM-5 catastrophic-forgetting guard) ─────────────────
    private readonly _rehearsalSize : number;
    private readonly _rehearsalK    : number;
    private readonly _rehearsal     : Array<{ input: string; teacherOutput: string }> = [];
    private _rehearsalState         : number;

    /**
     * @param student  What learns: an `SSMRuntime`, or any {@link DistillationStudent}.
     * @param teacher  What teaches: any `TransformerBridge`, or any {@link DistillationTeacher}.
     * @param rehearsal Optional experience-replay config (EVM-5). When `bufferSize > 0`,
     *                 each adapt also trains on a sample of past exemplars.
     */
    constructor(student: SSMRuntime | DistillationStudent<R, O>, teacher: DistillationTeacher, rehearsal: RehearsalOptions = {}) {
        this._student = isDistillationStudent<R, O>(student)
            ? student
            : (ssmRuntimeStudent(student as SSMRuntime) as unknown as DistillationStudent<R, O>);
        this._teacher = teacher;
        this._rehearsalSize = Math.max(0, rehearsal.bufferSize ?? 0);
        this._rehearsalK    = Math.max(0, rehearsal.sampleK ?? 2);
        this._rehearsalState = (rehearsal.seed ?? 1) >>> 0 || 1;
    }

    /** Current number of exemplars held in the rehearsal buffer (EVM-5). */
    getRehearsalBufferSize(): number {
        return this._rehearsal.length;
    }

    /** Deterministic [0,1) draw for reproducible rehearsal sampling. */
    private _rand(): number {
        this._rehearsalState = (Math.imul(1664525, this._rehearsalState) + 1013904223) >>> 0;
        return this._rehearsalState / 0x1_0000_0000;
    }

    /** Sample up to `_rehearsalK` distinct past exemplars (reservoir-free, by index). */
    private _sampleRehearsal(): Array<{ input: string; teacherOutput: string }> {
        if (this._rehearsalSize === 0 || this._rehearsal.length === 0 || this._rehearsalK === 0) return [];
        const k = Math.min(this._rehearsalK, this._rehearsal.length);
        const idxs = this._rehearsal.map((_, i) => i);
        // Partial Fisher–Yates to pick k distinct indices deterministically.
        for (let i = 0; i < k; i++) {
            const j = i + Math.floor(this._rand() * (idxs.length - i));
            const tmp = idxs[i]!; idxs[i] = idxs[j]!; idxs[j] = tmp;
        }
        return idxs.slice(0, k).map((i) => this._rehearsal[i]!);
    }

    private _pushRehearsal(input: string, teacherOutput: string): void {
        if (this._rehearsalSize === 0) return;
        this._rehearsal.push({ input, teacherOutput });
        if (this._rehearsal.length > this._rehearsalSize) this._rehearsal.shift();
    }

    /**
     * Runs a single distillation pass:
     *   1. Teacher generates a response for `input`
     *   2. Quality gate is evaluated (if configured)
     *   3. Student is adapted on `input` + the teacher's output
     *
     * The training signal is the teacher's full response — this teaches the student
     * what a good response to that prompt looks like, without labelled data.
     */
    async distill(input: string, opts: DistillOptions<O> = {}): Promise<DistillResult<R>> {
        let teacherOutput = '';
        let teacherError: unknown;
        try {
            teacherOutput = await this._teacher.generate(input, opts.generate);
        } catch (err) {
            if (opts.fallbackText === undefined) {
                throw new SSMError(
                    'DISTILL_FAILED',
                    `Teacher bridge failed to generate for distillation: ${err instanceof Error ? err.message : String(err)}`,
                    err,
                );
            }
            teacherError = err;
        }
        if (teacherError !== undefined) {
            return this._learnFallback(input, '', 'teacher_failed', opts, teacherError);
        }

        // ── Quality gate ──────────────────────────────────────────────────────

        // Pre-adapt novelty of this exemplar, hoisted so it survives to the TRAINED path.
        let gatePerplexity: number | undefined;
        const gate = opts.qualityGate;

        if (gate?.minLength != null && teacherOutput.length < gate.minLength) {
            if (opts.fallbackText !== undefined) return this._learnFallback(input, teacherOutput, 'low_quality', opts);
            return this._skip(input, teacherOutput, 'low_quality');
        }

        if (gate?.maxPerplexity != null && this._student.evaluate) {
            try {
                gatePerplexity = await this._student.evaluate(teacherOutput);
            } catch {
                // Evaluation failure is non-fatal — proceed with adaptation
            }
            if (gatePerplexity != null && gatePerplexity < gate.maxPerplexity) {
                return this._skip(input, teacherOutput, 'already_learned', gatePerplexity);
            }
        }

        // Train on the (prompt → response) mapping. EVM-5: interleave a sample of past
        // exemplars (experience replay) so this adapt reinforces prior knowledge
        // instead of overwriting it.
        const rehearsed = this._sampleRehearsal();
        const context = (i: string) => (opts.contextChars != null ? i.trim().slice(0, opts.contextChars) : i);
        const trainingText = [...rehearsed, { input, teacherOutput }]
            .map((p) => `${context(p.input)}\n${p.teacherOutput}`)
            .join('\n\n');

        const adaptResult = await this._adapt(trainingText, opts);
        this._pushRehearsal(input, teacherOutput);
        this._appendLog({
            input,
            teacherOutputLength: teacherOutput.length,
            skipped    : false,
            distilled  : true,
            ...this._student.describe(adaptResult),
            gatePerplexity,
        });

        return { input, teacherOutput, adaptResult, skipped: false, distilled: true, rehearsed: rehearsed.length, gatePerplexity };
    }

    /**
     * Runs distillation for each input in sequence.
     * Aggregate statistics are returned alongside individual results.
     */
    async distillBatch(inputs: string[], opts: DistillOptions<O> = {}): Promise<DistillBatchResult<R>> {
        const startMs = Date.now();
        const results: DistillResult<R>[] = [];
        let totalEpochs = 0;

        for (const input of inputs) {
            const result = await this.distill(input, opts);
            results.push(result);
            totalEpochs += this._student.describe(result.adaptResult).epochs;
        }

        return {
            results,
            totalEpochs,
            totalMs: Date.now() - startMs,
        };
    }

    /**
     * Returns a copy of the in-memory distillation log (last 200 entries).
     */
    getLog(): DistillationLog[] {
        return this._log.slice();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private async _adapt(trainingText: string, opts: DistillOptions<O>): Promise<R> {
        try {
            return await this._student.adapt(trainingText, opts.adapt);
        } catch (err) {
            throw new SSMError(
                'DISTILL_FAILED',
                `SSM adaptation failed during distillation: ${err instanceof Error ? err.message : String(err)}`,
                err,
            );
        }
    }

    /** No exemplar worth learning: learn the caller's fallback text instead. */
    private async _learnFallback(
        input: string,
        teacherOutput: string,
        skipReason: DistillSkipReason,
        opts: DistillOptions<O>,
        teacherError?: unknown,
    ): Promise<DistillResult<R>> {
        const adaptResult = await this._adapt(opts.fallbackText!, opts);
        this._appendLog({
            input,
            teacherOutputLength: teacherOutput.length,
            skipped   : false,
            distilled : false,
            skipReason,
            ...this._student.describe(adaptResult),
        });
        return {
            input,
            teacherOutput,
            adaptResult,
            skipped: false,
            distilled: false,
            skipReason,
            ...(teacherError !== undefined ? { teacherError } : {}),
        };
    }

    /** Nothing adapted. */
    private _skip(input: string, teacherOutput: string, skipReason: DistillSkipReason, gatePerplexity?: number): DistillResult<R> {
        this._appendLog({
            input,
            teacherOutputLength: teacherOutput.length,
            skipped   : true,
            distilled : false,
            skipReason,
            epochs    : 0,
            gatePerplexity,
        });
        return {
            input,
            teacherOutput,
            adaptResult: this._student.skippedResult(),
            skipped: true,
            distilled: false,
            skipReason,
            ...(gatePerplexity !== undefined ? { gatePerplexity } : {}),
        };
    }

    private _appendLog(entry: Omit<DistillationLog, 'timestamp'>): void {
        this._log.push({ timestamp: Date.now(), ...entry });
        if (this._log.length > MAX_LOG_ENTRIES) {
            this._log.shift();
        }
    }
}
