/**
 * trainer.ts – MambaTrainer class
 */

import {
    createUniformBuffer,
    createStorageBuffer,
    createEmptyStorageBuffer,
    createComputePipeline,
    createBindGroup,
    dispatchKernel,
    readBuffer,
    cdiv,
} from '../utils/gpu_utils.js';

import { crossEntropyLoss, crossEntropyGrad } from './autograd.js';
import { WEIGHT_UPDATE_WGSL, GRAD_CLIP_WGSL } from '../kernels/weight_update.js';
import { HybridMambaModel, MambaModel } from '../model/mamba_model.js';
import { BPETokenizer } from '../tokenizer/bpe.js';
import type { LayerParam as BlockParam } from '../model/sequence_layer.js';

export interface TrainOptions {
  learningRate?: number;
  epochs?: number;
  batchSize?: number;
  seqLen?: number;
  maxGradNorm?: number;
  weightDecay?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  wsla?: boolean;
  /** Trust region: max |Δθ| per optimizer step. 0 disables. Defaults to
   *  {@link WSLA_MAX_DELTA} in WSLA (write-through) mode, else 0 (full training). */
  maxDelta?: number;
  /** Read back the pre-clip gradient L2 norm each step (the `grad_norm_reduce`
   *  kernel already computes it — this just maps it to the CPU). OFF by default: the
   *  readback is a GPU sync point that would slow a long finetune. `adapt()` turns it
   *  ON so the norm reaches {@link AdaptResult} as an instability signal. */
  trackGradNorm?: boolean;
  onEpochEnd?: ((epoch: number, loss: number, gradNorm?: number) => void) | null;
}

/**
 * Default per-step trust region for WSLA / write-through adaptation. Small
 * enough that a single `adapt()` nudges the narrow params without lurching, so
 * repeated adapts stay stable (and any that regress are cheaply rolled back by
 * the session). Full-training callers pass `maxDelta: 0` (or omit it in
 * non-WSLA mode) for unbounded steps.
 */
export const WSLA_MAX_DELTA = 0.05;

interface AdamMoments {
  m: GPUBuffer;
  v: GPUBuffer;
}

interface AdamHyperparams {
  learningRate: number;
  weightDecay: number;
  beta1: number;
  beta2: number;
  eps: number;
  beta1_t: number;
  beta2_t: number;
  maxDelta: number;
}

export class MambaTrainer {
    model: HybridMambaModel;
    tokenizer: BPETokenizer | null;
    device: GPUDevice;
    /**
     * Adam moments keyed by parameter NAME (not array index). Name-keying is what
     * lets WSLA toggle safely: a narrow write-through step updates only the
     * `layer{i}.wXProj/bXProj` subset, a full fine-tune updates everything, and
     * both reuse the SAME `m`/`v` buffers per parameter. Index-keyed moments (the
     * old scheme) silently misaligned the moment with the wrong parameter the
     * moment the trainable set changed shape — corrupting the update.
     */
    private _moments: Map<string, AdamMoments>;
    private _step: number;
    private _adamwPipeline: GPUComputePipeline;
    private _clipReducePipeline: GPUComputePipeline;
    private _clipScalePipeline: GPUComputePipeline;

    constructor(model: HybridMambaModel | MambaModel, tokenizer: BPETokenizer | null = null) {
        this.model     = model;
        this.tokenizer = tokenizer;
        this.device    = model.device;

        this._moments = new Map();
        this._step = 0;

        this._adamwPipeline   = createComputePipeline(this.device, WEIGHT_UPDATE_WGSL, 'adamw_update');
        this._clipReducePipeline = createComputePipeline(this.device, GRAD_CLIP_WGSL, 'grad_norm_reduce');
        this._clipScalePipeline  = createComputePipeline(this.device, GRAD_CLIP_WGSL, 'grad_clip_scale');
    }

    /** Get-or-create the Adam first/second moments for a parameter, by name. */
    private _momentFor(p: BlockParam): AdamMoments {
        let mom = this._moments.get(p.name);
        if (!mom) {
            mom = {
                m: createEmptyStorageBuffer(this.device, p.numel * 4, false),
                v: createEmptyStorageBuffer(this.device, p.numel * 4, false),
            };
            this._moments.set(p.name, mom);
        }
        return mom;
    }

    async train(input: string | number[], opts: TrainOptions = {}): Promise<number[]> {
        const {
            learningRate = 1e-4,
            epochs       = 5,
            batchSize    = 1,
            seqLen       = 512,
            maxGradNorm  = 1.0,
            weightDecay  = 0.01,
            beta1        = 0.9,
            beta2        = 0.999,
            eps          = 1e-8,
            wsla         = false,
            trackGradNorm = false,
            onEpochEnd   = null,
        } = opts;
        // Trust region defaults ON for write-through (WSLA) adapts, OFF otherwise.
        const maxDelta = opts.maxDelta ?? (wsla ? WSLA_MAX_DELTA : 0);

        if (wsla) this.model.setWSLAMode(true);

        let tokenIds: number[];
        if (typeof input === 'string') {
            if (!this.tokenizer) {
                throw new Error(
                    'MambaTrainer requires a tokenizer when input is a string. ' +
                    'Pass a BPETokenizer instance as the second constructor argument.'
                );
            }
            tokenIds = this.tokenizer.encode(input);
        } else {
            tokenIds = Array.from(input);
        }

        if (tokenIds.length < 2) {
            throw new Error('Input must contain at least 2 tokens to form a training pair.');
        }

        const chunks = buildChunks(tokenIds, seqLen);
        if (chunks.length === 0) {
            throw new Error('Input is too short to form any training chunk.');
        }

        // Moments are created lazily per-parameter (name-keyed) on first touch —
        // no upfront allocation needed, and it stays correct as WSLA narrows the
        // trainable set.

        const epochLosses: number[] = [];

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let epochGradNorm = 0;
            let numSteps  = 0;

            for (const { inputs, targets } of chunks) {
                const { loss, gradNorm } = await this._trainStep(
                    inputs, targets, batchSize,
                    { learningRate, maxGradNorm, weightDecay, beta1, beta2, eps, wsla, maxDelta, trackGradNorm }
                );
                epochLoss += loss;
                if (gradNorm != null) epochGradNorm += gradNorm;
                numSteps++;
            }

            const avgLoss = epochLoss / numSteps;
            epochLosses.push(avgLoss);

            // Mean pre-clip grad norm this epoch (only measured when trackGradNorm is on).
            const avgGradNorm = trackGradNorm ? epochGradNorm / numSteps : undefined;
            if (onEpochEnd) onEpochEnd(epoch + 1, avgLoss, avgGradNorm);
        }

        if (wsla) this.model.setWSLAMode(false);
        return epochLosses;
    }

    private async _trainStep(
        inputs: number[],
        targets: number[],
        batch: number,
        hyperparams: TrainOptions & { learningRate: number; maxGradNorm: number; weightDecay: number; beta1: number; beta2: number; eps: number; maxDelta: number; trackGradNorm?: boolean }
    ): Promise<{ loss: number; gradNorm: number | null }> {
        const { learningRate, maxGradNorm, weightDecay, beta1, beta2, eps, maxDelta, trackGradNorm } = hyperparams;

        this._step++;
        const seqLen    = inputs.length;
        const vocabSize = this.model.config.vocabSize;

        const { logits, gpuLogits } = await this.model.forward(
            new Uint32Array(inputs), batch, seqLen
        );

        let totalLoss = 0;
        const dLogits = new Float32Array(batch * seqLen * vocabSize);

        for (let i = 0; i < seqLen; i++) {
            const offset = i * vocabSize;
            const logitSlice = logits.slice(offset, offset + vocabSize);
            const target = targets[i]!;
            totalLoss += crossEntropyLoss(logitSlice, target);
            const grad  = crossEntropyGrad(logitSlice, target);
            for (let v = 0; v < vocabSize; v++) {
                dLogits[offset + v] = grad[v]! / seqLen;
            }
        }
        const loss = totalLoss / seqLen;

        const dLogitsBuf = createStorageBuffer(this.device, dLogits, false);

        const gradNorm = await this._clipGradients(dLogitsBuf, dLogits.length, maxGradNorm, !!trackGradNorm);

        // Only the trainable set is updated. Under WSLA (write-through) that is
        // the narrow per-layer subset and the backbone (incl. every A_log) stays
        // frozen — the guarantee that repeated adapts can't destabilise the SSM.
        const params  = this.model.getTrainableParams();
        const beta1_t = Math.pow(beta1, this._step);
        const beta2_t = Math.pow(beta2, this._step);

        await this._adamwStep(
            params, [dLogitsBuf],
            { learningRate, weightDecay, beta1, beta2, eps, beta1_t, beta2_t, maxDelta }
        );

        dLogitsBuf.destroy();
        gpuLogits.destroy();

        return { loss, gradNorm };
    }

    private async _adamwStep(
        params: BlockParam[],
        gradBufs: GPUBuffer[],
        hp: AdamHyperparams
    ): Promise<void> {
        const { learningRate, weightDecay, beta1, beta2, eps, beta1_t, beta2_t, maxDelta } = hp;

        for (let i = 0; i < params.length; i++) {
            const p       = params[i]!;
            const gradBuf = gradBufs[Math.min(i, gradBufs.length - 1)]!;

            if (!gradBuf || gradBuf.size < p.numel * 4) continue;

            const mom = this._momentFor(p);
            const paramsBuf = createUniformBuffer(this.device, packAdamParams(
                p.numel, learningRate, beta1, beta2, eps, weightDecay, beta1_t, beta2_t, maxDelta
            ));

            const bg = createBindGroup(this.device, this._adamwPipeline, [
                paramsBuf,
                p.buf,
                gradBuf,
                mom.m,
                mom.v,
            ]);

            dispatchKernel(this.device, this._adamwPipeline, bg,
                [cdiv(p.numel, 256), 1, 1]);

            paramsBuf.destroy();
        }
    }

    /**
     * Clip gradients to `maxNorm` in place. The reduce kernel writes Σg² into
     * `normSqBuf`; the scale kernel then rescales the grads. When `trackNorm` is set
     * we read Σg² back and return the true pre-clip L2 norm (√Σg²) — the kernel
     * already computed it, so this is only a small buffer readback (a GPU sync,
     * hence opt-in). Returns null when not tracking.
     */
    private async _clipGradients(gradBuf: GPUBuffer, numel: number, maxNorm: number, trackNorm = false): Promise<number | null> {
        const normSqBuf = createEmptyStorageBuffer(this.device, 4, true);
        this.device.queue.writeBuffer(normSqBuf, 0, new Float32Array([0.0]));

        const clipParams = new ArrayBuffer(8);
        new Uint32Array(clipParams, 0, 1).set([numel]);
        new Float32Array(clipParams, 4, 1).set([maxNorm * maxNorm]);
        const pBuf = createUniformBuffer(this.device, clipParams);

        const bg1 = createBindGroup(this.device, this._clipReducePipeline,
            [pBuf, gradBuf, normSqBuf]);
        dispatchKernel(this.device, this._clipReducePipeline, bg1,
            [cdiv(numel, 256), 1, 1]);

        // Read Σg² AFTER the reduce (the scale kernel only reads it, never overwrites),
        // so the returned norm is the pre-clip magnitude — the instability signal.
        let gradNorm: number | null = null;
        if (trackNorm) {
            const out = await readBuffer(this.device, normSqBuf, 4);
            gradNorm = Math.sqrt(Math.max(0, out[0] ?? 0));
        }

        const bg2 = createBindGroup(this.device, this._clipScalePipeline,
            [pBuf, gradBuf, normSqBuf]);
        dispatchKernel(this.device, this._clipScalePipeline, bg2,
            [cdiv(numel, 256), 1, 1]);

        pBuf.destroy();
        normSqBuf.destroy();
        return gradNorm;
    }

    async evaluate(input: string | number[]): Promise<number> {
        let tokenIds: number[];
        if (typeof input === 'string') {
            if (!this.tokenizer) throw new Error('Tokenizer required for string input.');
            tokenIds = this.tokenizer.encode(input);
        } else {
            tokenIds = Array.from(input);
        }

        const seqLen    = tokenIds.length;
        const vocabSize = this.model.config.vocabSize;

        const { logits } = await this.model.forward(
            new Uint32Array(tokenIds.slice(0, -1)), 1, seqLen - 1
        );

        let totalLoss = 0;
        for (let i = 0; i < seqLen - 1; i++) {
            const offset = i * vocabSize;
            totalLoss += crossEntropyLoss(
                logits.slice(offset, offset + vocabSize),
                tokenIds[i + 1]!
            );
        }

        const avgLoss = totalLoss / (seqLen - 1);
        return Math.exp(avgLoss);
    }
}

function buildChunks(ids: number[], seqLen: number): Array<{inputs: number[], targets: number[]}> {
    const chunks: Array<{inputs: number[], targets: number[]}> = [];
    for (let start = 0; start + seqLen < ids.length; start += seqLen) {
        chunks.push({
            inputs : ids.slice(start, start + seqLen),
            targets: ids.slice(start + 1, start + seqLen + 1),
        });
    }
    const rem = ids.length % seqLen;
    if (rem > 1) {
        const start = ids.length - rem;
        chunks.push({
            inputs : ids.slice(start, -1),
            targets: ids.slice(start + 1),
        });
    }
    return chunks;
}

function packAdamParams(
    numElements: number, lr: number, beta1: number, beta2: number,
    eps: number, weightDecay: number, beta1_t: number, beta2_t: number,
    maxDelta: number
): ArrayBuffer {
    // 48 bytes: u32 + 8×f32 = 36, padded to a 16-byte multiple for the uniform
    // layout (matches AdamParams' _pad0.._pad2 in WEIGHT_UPDATE_WGSL).
    const buf = new ArrayBuffer(48);
    new Uint32Array(buf, 0, 1).set([numElements]);
    new Float32Array(buf, 4, 8).set([lr, beta1, beta2, eps, weightDecay, beta1_t, beta2_t, maxDelta]);
    return buf;
}
