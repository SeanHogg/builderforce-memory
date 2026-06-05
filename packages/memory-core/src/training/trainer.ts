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
  onEpochEnd?: ((epoch: number, loss: number) => void) | null;
}

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
}

export class MambaTrainer {
    model: HybridMambaModel;
    tokenizer: BPETokenizer | null;
    device: GPUDevice;
    private _moments: AdamMoments[] | null;
    private _step: number;
    private _adamwPipeline: GPUComputePipeline;
    private _clipReducePipeline: GPUComputePipeline;
    private _clipScalePipeline: GPUComputePipeline;

    constructor(model: HybridMambaModel | MambaModel, tokenizer: BPETokenizer | null = null) {
        this.model     = model;
        this.tokenizer = tokenizer;
        this.device    = model.device;

        this._moments = null;
        this._step = 0;

        this._adamwPipeline   = createComputePipeline(this.device, WEIGHT_UPDATE_WGSL, 'adamw_update');
        this._clipReducePipeline = createComputePipeline(this.device, GRAD_CLIP_WGSL, 'grad_norm_reduce');
        this._clipScalePipeline  = createComputePipeline(this.device, GRAD_CLIP_WGSL, 'grad_clip_scale');
    }

    private _initMoments(): void {
        if (this._moments) return;
        this._moments = this.model.parameters().map(p => ({
            m: createEmptyStorageBuffer(this.device, p.numel * 4, false),
            v: createEmptyStorageBuffer(this.device, p.numel * 4, false),
        }));
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
            onEpochEnd   = null,
        } = opts;

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

        this._initMoments();

        const epochLosses: number[] = [];

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let numSteps  = 0;

            for (const { inputs, targets } of chunks) {
                const loss = await this._trainStep(
                    inputs, targets, batchSize,
                    { learningRate, maxGradNorm, weightDecay, beta1, beta2, eps, wsla }
                );
                epochLoss += loss;
                numSteps++;
            }

            const avgLoss = epochLoss / numSteps;
            epochLosses.push(avgLoss);

            if (onEpochEnd) onEpochEnd(epoch + 1, avgLoss);
        }

        if (wsla) this.model.setWSLAMode(false);
        return epochLosses;
    }

    private async _trainStep(
        inputs: number[],
        targets: number[],
        batch: number,
        hyperparams: TrainOptions & { learningRate: number; maxGradNorm: number; weightDecay: number; beta1: number; beta2: number; eps: number }
    ): Promise<number> {
        const { learningRate, maxGradNorm, weightDecay, beta1, beta2, eps } = hyperparams;

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

        await this._clipGradients(dLogitsBuf, dLogits.length, maxGradNorm);

        const params  = this.model.parameters();
        const beta1_t = Math.pow(beta1, this._step);
        const beta2_t = Math.pow(beta2, this._step);

        await this._adamwStep(
            params, [dLogitsBuf],
            { learningRate, weightDecay, beta1, beta2, eps, beta1_t, beta2_t }
        );

        dLogitsBuf.destroy();
        gpuLogits.destroy();

        return loss;
    }

    private async _adamwStep(
        params: BlockParam[],
        gradBufs: GPUBuffer[],
        hp: AdamHyperparams
    ): Promise<void> {
        const { learningRate, weightDecay, beta1, beta2, eps, beta1_t, beta2_t } = hp;

        for (let i = 0; i < params.length; i++) {
            const p       = params[i]!;
            const gradBuf = gradBufs[Math.min(i, gradBufs.length - 1)]!;

            if (!gradBuf || gradBuf.size < p.numel * 4) continue;

            const paramsBuf = createUniformBuffer(this.device, packAdamParams(
                p.numel, learningRate, beta1, beta2, eps, weightDecay, beta1_t, beta2_t
            ));

            const bg = createBindGroup(this.device, this._adamwPipeline, [
                paramsBuf,
                p.buf,
                gradBuf,
                this._moments![i]!.m,
                this._moments![i]!.v,
            ]);

            dispatchKernel(this.device, this._adamwPipeline, bg,
                [cdiv(p.numel, 256), 1, 1]);

            paramsBuf.destroy();
        }
    }

    private async _clipGradients(gradBuf: GPUBuffer, numel: number, maxNorm: number): Promise<void> {
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

        const bg2 = createBindGroup(this.device, this._clipScalePipeline,
            [pBuf, gradBuf, normSqBuf]);
        dispatchKernel(this.device, this._clipScalePipeline, bg2,
            [cdiv(numel, 256), 1, 1]);

        pBuf.destroy();
        normSqBuf.destroy();
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
    eps: number, weightDecay: number, beta1_t: number, beta2_t: number
): ArrayBuffer {
    const buf = new ArrayBuffer(32);
    new Uint32Array(buf, 0, 1).set([numElements]);
    new Float32Array(buf, 4, 7).set([lr, beta1, beta2, eps, weightDecay, beta1_t, beta2_t]);
    return buf;
}
