/**
 * DistillationEngine – JS-only online knowledge distillation.
 *
 * The core insight: use a transformer as a *teacher* to generate high-quality
 * responses, then adapt the SSM *student* on those responses using WSLA.
 * This runs entirely in the browser with no Python or full-retraining required.
 *
 * Distillation flow:
 *   1. bridge.generate(input)  → teacher output
 *   2. runtime.adapt(teacherOutput, opts.adapt)  → SSM trains on it
 *   3. Return both results for inspection
 */

import type { AdaptOptions, AdaptResult } from '../session/index.js';
import type { SSMRuntime } from '../runtime/SSMRuntime.js';
import type { TransformerBridge, BridgeGenerateOptions } from '../bridges/TransformerBridge.js';
import { SSMError } from '../errors/SSMError.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface QualityGate {
    /**
     * Minimum character length of the teacher output.
     * Outputs shorter than this are considered low quality and are skipped.
     */
    minLength?     : number;
    /**
     * Maximum SSM perplexity threshold.
     * When the SSM already achieves perplexity below this value on the teacher
     * output, the content is considered already learned and adaptation is skipped.
     */
    maxPerplexity? : number;
}

export interface DistillOptions {
    /**
     * Options forwarded to `runtime.adapt()`.
     * Default: { wsla: true, epochs: 3 }
     * WSLA is preferred because it is fast and targets the selective
     * projection rows — exactly the parameters that encode token routing.
     */
    adapt?       : AdaptOptions;

    /**
     * Options forwarded to `bridge.generate()`.
     */
    generate?    : BridgeGenerateOptions;

    /**
     * Quality gate filters that can skip adaptation for low-quality or
     * already-learned inputs.
     */
    qualityGate? : QualityGate;
}

/**
 * Catastrophic-forgetting guard (EVM-5). Online WSLA adapts a narrow set of
 * weights toward the newest exemplar, which can erode previously-learned
 * knowledge. A rehearsal (experience-replay) buffer mitigates this: each adapt
 * also trains on a sample of past exemplars, so old knowledge is continually
 * reinforced instead of overwritten.
 */
export interface RehearsalOptions {
    /** Ring-buffer capacity of past exemplars. 0 disables rehearsal. Default 0. */
    bufferSize?: number;
    /** How many past exemplars to mix into each adapt pass. Default 2. */
    sampleK?: number;
    /** Deterministic sampling seed. Default 1. */
    seed?: number;
}

export interface DistillResult {
    /** The input prompt that was distilled. */
    input        : string;
    /** The teacher's (transformer bridge) response to the input. */
    teacherOutput: string;
    /** The adapt() result from training the SSM on the teacher output. */
    adaptResult  : AdaptResult;
    /** Whether adaptation was skipped by the quality gate. */
    skipped?     : boolean;
    /** Reason adaptation was skipped, if applicable. */
    skipReason?  : string;
    /** Number of past exemplars rehearsed alongside this one (EVM-5). */
    rehearsed?   : number;
    /** Student's pre-adapt perplexity on the teacher output, when the quality gate
     *  measured it — a "how novel was this exemplar" signal. Kept even when the
     *  exemplar was NOT skipped (previously discarded on the trained path). */
    gatePerplexity?: number;
}

export interface DistillBatchResult {
    results    : DistillResult[];
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
    skipReason?        : string;
    finalLoss?         : number;
    epochs             : number;
    /** Pre-adapt perplexity on the teacher output (when the quality gate measured it). */
    gatePerplexity?    : number;
}

/** Maximum number of distillation log entries to retain in memory. */
const MAX_LOG_ENTRIES = 200;

// ── DistillationEngine ────────────────────────────────────────────────────────

export class DistillationEngine {
    private readonly _runtime  : SSMRuntime;
    private readonly _bridge   : TransformerBridge;
    private readonly _log      : DistillationLog[] = [];

    // ── Rehearsal buffer (EVM-5 catastrophic-forgetting guard) ─────────────────
    private readonly _rehearsalSize : number;
    private readonly _rehearsalK    : number;
    private readonly _rehearsal     : Array<{ input: string; teacherOutput: string }> = [];
    private _rehearsalState         : number;

    /**
     * @param runtime The SSMRuntime whose SSM will be trained as the student.
     * @param bridge  The transformer bridge acting as teacher.
     *                A bridge must be provided — distillation requires one.
     * @param rehearsal Optional experience-replay config (EVM-5). When
     *                `bufferSize > 0`, each adapt also trains on a sample of past
     *                exemplars to guard against catastrophic forgetting.
     */
    constructor(runtime: SSMRuntime, bridge: TransformerBridge, rehearsal: RehearsalOptions = {}) {
        this._runtime = runtime;
        this._bridge  = bridge;
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
     *   3. SSM is adapted on the teacher's output (WSLA by default)
     *
     * The training signal is the teacher's full response — this teaches the
     * SSM what a good response to that prompt looks like, without requiring
     * labelled data or a loss function beyond the standard LM objective.
     */
    async distill(input: string, opts: DistillOptions = {}): Promise<DistillResult> {
        const adaptOpts: AdaptOptions = {
            wsla        : true,
            epochs      : 3,
            ...opts.adapt,
        };

        let teacherOutput: string;
        try {
            teacherOutput = await this._bridge.generate(input, opts.generate);
        } catch (err) {
            throw new SSMError(
                'DISTILL_FAILED',
                `Teacher bridge failed to generate for distillation: ${err instanceof Error ? err.message : String(err)}`,
                err,
            );
        }

        // ── Quality gate ──────────────────────────────────────────────────────

        // Pre-adapt novelty of this exemplar (student perplexity on the teacher output).
        // Measured by the gate below; hoisted so it survives to the TRAINED path instead
        // of being dropped for exactly the exemplars we go on to learn from.
        let gatePerplexity: number | undefined;

        if (opts.qualityGate) {
            const gate = opts.qualityGate;

            if (gate.minLength != null && teacherOutput.length < gate.minLength) {
                const result: DistillResult = {
                    input,
                    teacherOutput,
                    adaptResult : { losses: [], epochCount: 0, durationMs: 0 },
                    skipped     : true,
                    skipReason  : 'low_quality',
                };
                this._appendLog({
                    input,
                    teacherOutputLength: teacherOutput.length,
                    skipped    : true,
                    skipReason : 'low_quality',
                    epochs     : 0,
                });
                return result;
            }

            if (gate.maxPerplexity != null) {
                try {
                    gatePerplexity = await this._runtime.evaluate(teacherOutput);
                } catch {
                    // Evaluation failure is non-fatal — proceed with adaptation
                }
                if (gatePerplexity != null && gatePerplexity < gate.maxPerplexity) {
                    const result: DistillResult = {
                        input,
                        teacherOutput,
                        adaptResult : { losses: [], epochCount: 0, durationMs: 0 },
                        skipped     : true,
                        skipReason  : 'already_learned',
                        gatePerplexity,
                    };
                    this._appendLog({
                        input,
                        teacherOutputLength: teacherOutput.length,
                        skipped    : true,
                        skipReason : 'already_learned',
                        epochs     : 0,
                        gatePerplexity,
                    });
                    return result;
                }
            }
        }

        // Train the SSM on the teacher's output, prepending the input so the model
        // learns the (prompt → response) mapping. EVM-5: interleave a sample of
        // past exemplars (experience replay) so this adapt reinforces prior
        // knowledge instead of overwriting it (catastrophic-forgetting guard).
        const rehearsed = this._sampleRehearsal();
        const pairs = [...rehearsed, { input, teacherOutput }].map((p) => `${p.input}\n${p.teacherOutput}`);
        const trainingText = pairs.join('\n\n');

        let adaptResult: AdaptResult;
        try {
            adaptResult = await this._runtime.adapt(trainingText, adaptOpts);
        } catch (err) {
            throw new SSMError(
                'DISTILL_FAILED',
                `SSM adaptation failed during distillation: ${err instanceof Error ? err.message : String(err)}`,
                err,
            );
        }

        // Record the new exemplar for future rehearsal.
        this._pushRehearsal(input, teacherOutput);

        this._appendLog({
            input,
            teacherOutputLength: teacherOutput.length,
            skipped    : false,
            finalLoss  : adaptResult.losses.at(-1),
            epochs     : adaptResult.epochCount,
            gatePerplexity,
        });

        return { input, teacherOutput, adaptResult, skipped: false, rehearsed: rehearsed.length, gatePerplexity };
    }

    /**
     * Runs distillation for each input in sequence.
     * Aggregate statistics are returned alongside individual results.
     */
    async distillBatch(inputs: string[], opts: DistillOptions = {}): Promise<DistillBatchResult> {
        const startMs = Date.now();
        const results: DistillResult[] = [];
        let totalEpochs = 0;

        for (const input of inputs) {
            const result = await this.distill(input, opts);
            results.push(result);
            totalEpochs += result.adaptResult.epochCount;
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

    private _appendLog(entry: Omit<DistillationLog, 'timestamp'>): void {
        this._log.push({ timestamp: Date.now(), ...entry });
        if (this._log.length > MAX_LOG_ENTRIES) {
            this._log.shift();
        }
    }
}
