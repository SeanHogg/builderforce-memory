/**
 * limbic_trainer.ts – LimbicTrainer: gradient-based training for the LimbicModel.
 *
 * Trains the affect head to predict affect deltas and reward from
 * (experience embedding, current state) pairs, using full-batch gradient
 * descent with AdamW. When a GPUDevice is supplied the optimiser step runs on
 * the GPU via the shared WEIGHT_UPDATE_WGSL kernel (real WebGPU training); with
 * no device it uses a numerically-equivalent CPU AdamW so training works
 * everywhere (CI, Node without @webgpu/node, etc.).
 *
 * The objective is MSE (regression), not the cross-entropy used by the
 * language-model {@link MambaTrainer} — a limbic experience has continuous
 * targets, not a next-token distribution.
 */

import {
  createUniformBuffer,
  createStorageBuffer,
  createComputePipeline,
  createBindGroup,
  dispatchKernel,
  readBuffer,
  cdiv,
} from "../utils/gpu_utils.js";
import { WEIGHT_UPDATE_WGSL } from "../kernels/weight_update.js";
import type { LimbicModel, LimbicParam } from "./limbic_model.js";

/** One training example: an experience and the affect change it should produce. */
export interface LimbicSample {
  /** Experience embedding (length = model.inputDim). */
  input: ArrayLike<number>;
  /** Affective state at the time of the experience (length = model.stateDim). */
  state: ArrayLike<number>;
  /** Observed affect delta target in (-1, 1) per state dim (length = model.stateDim). */
  deltaTarget: ArrayLike<number>;
  /** Observed scalar reward for the experience. */
  reward: number;
}

export interface LimbicTrainOptions {
  learningRate?: number;
  epochs?: number;
  weightDecay?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  /** Max global gradient L2 norm before the optimiser step. Default 1.0. */
  maxGradNorm?: number;
  onEpochEnd?: ((epoch: number, loss: number) => void) | null;
}

interface AdamMoment {
  m: Float32Array;
  v: Float32Array;
}

function packAdamParams(
  numElements: number,
  lr: number,
  beta1: number,
  beta2: number,
  eps: number,
  weightDecay: number,
  beta1_t: number,
  beta2_t: number,
): ArrayBuffer {
  const buf = new ArrayBuffer(32);
  new Uint32Array(buf, 0, 1).set([numElements]);
  new Float32Array(buf, 4, 7).set([lr, beta1, beta2, eps, weightDecay, beta1_t, beta2_t]);
  return buf;
}

export class LimbicTrainer {
  readonly model: LimbicModel;
  readonly device: GPUDevice | null;
  private _moments: AdamMoment[] | null = null;
  private _step = 0;
  private readonly _adamwPipeline: GPUComputePipeline | null;

  constructor(model: LimbicModel, device: GPUDevice | null = null) {
    this.model = model;
    this.device = device;
    this._adamwPipeline = device
      ? createComputePipeline(device, WEIGHT_UPDATE_WGSL, "adamw_update")
      : null;
  }

  /** Whether the optimiser step runs on the GPU. */
  get gpuTraining(): boolean {
    return this.device != null && this._adamwPipeline != null;
  }

  private _initMoments(): void {
    if (this._moments) return;
    this._moments = this.model.parameters().map((p) => ({
      m: new Float32Array(p.numel),
      v: new Float32Array(p.numel),
    }));
  }

  /**
   * Train on a batch of samples for `epochs` passes. Returns the per-epoch mean
   * loss (monotonically decreasing on a learnable mapping). Full-batch: grads
   * accumulate across the whole sequence (recurrent hidden carried, reset per
   * epoch), are averaged, clipped, then applied once per epoch.
   */
  async train(samples: LimbicSample[], opts: LimbicTrainOptions = {}): Promise<number[]> {
    if (samples.length === 0) throw new Error("LimbicTrainer.train: no samples");
    const {
      learningRate = 0.05,
      epochs = 50,
      weightDecay = 0.0,
      beta1 = 0.9,
      beta2 = 0.999,
      eps = 1e-8,
      maxGradNorm = 1.0,
      onEpochEnd = null,
    } = opts;

    this._initMoments();
    const losses: number[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      this.model.zeroGrad();
      let epochLoss = 0;

      // Each sample is one experience appraisal: cross-experience memory flows
      // through the affective state `s` (fed back by the runtime), not through
      // the hidden scratch `h`, so the hidden state resets per sample. This also
      // makes BPTT(1) exact — there is no truncated carry.
      for (const s of samples) {
        const { loss } = this.model.backwardStep(
          s.input,
          this.model.initHidden(),
          s.state,
          s.deltaTarget,
          s.reward,
        );
        epochLoss += loss;
      }

      // Average gradients over the batch.
      const grads = this.model.gradients();
      const invN = 1 / samples.length;
      for (const g of grads) {
        for (let i = 0; i < g.data.length; i++) g.data[i]! *= invN;
      }

      this._clipGradients(grads, maxGradNorm);

      this._step++;
      const beta1_t = Math.pow(beta1, this._step);
      const beta2_t = Math.pow(beta2, this._step);
      const hp = { learningRate, weightDecay, beta1, beta2, eps, beta1_t, beta2_t };
      if (this.gpuTraining) {
        await this._adamwStepGpu(grads, hp);
      } else {
        this._adamwStepCpu(grads, hp);
      }

      const avg = epochLoss / samples.length;
      losses.push(avg);
      if (onEpochEnd) onEpochEnd(epoch + 1, avg);
    }
    return losses;
  }

  /** Mean MSE loss over samples (no weight update). Hidden resets per sample. */
  evaluate(samples: LimbicSample[]): number {
    if (samples.length === 0) return 0;
    let total = 0;
    const { rewardWeight } = this.model.config;
    for (const s of samples) {
      const f = this.model.forward(s.input, this.model.initHidden(), s.state);
      let loss = 0;
      for (let k = 0; k < f.delta.length; k++) {
        const d = f.delta[k]! - (s.deltaTarget[k] ?? 0);
        loss += 0.5 * d * d;
      }
      const rd = f.reward - s.reward;
      loss += 0.5 * rewardWeight * rd * rd;
      total += loss;
    }
    return total / samples.length;
  }

  private _clipGradients(grads: LimbicParam[], maxNorm: number): void {
    let normSq = 0;
    for (const g of grads) for (let i = 0; i < g.data.length; i++) normSq += g.data[i]! * g.data[i]!;
    const norm = Math.sqrt(normSq);
    if (norm > maxNorm && norm > 0) {
      const scale = maxNorm / norm;
      for (const g of grads) for (let i = 0; i < g.data.length; i++) g.data[i]! *= scale;
    }
  }

  private _adamwStepCpu(
    grads: LimbicParam[],
    hp: { learningRate: number; weightDecay: number; beta1: number; beta2: number; eps: number; beta1_t: number; beta2_t: number },
  ): void {
    const params = this.model.parameters();
    const { learningRate: lr, weightDecay: wd, beta1, beta2, eps, beta1_t, beta2_t } = hp;
    for (let pi = 0; pi < params.length; pi++) {
      const p = params[pi]!.data;
      const g = grads[pi]!.data;
      const mom = this._moments![pi]!;
      for (let i = 0; i < p.length; i++) {
        const gi = g[i]!;
        mom.m[i] = beta1 * mom.m[i]! + (1 - beta1) * gi;
        mom.v[i] = beta2 * mom.v[i]! + (1 - beta2) * gi * gi;
        const mHat = mom.m[i]! / (1 - beta1_t);
        const vHat = mom.v[i]! / (1 - beta2_t);
        p[i] = p[i]! * (1 - lr * wd) - (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    }
  }

  /** AdamW on the GPU via the shared WEIGHT_UPDATE_WGSL kernel. Awaited per step. */
  private async _adamwStepGpu(
    grads: LimbicParam[],
    hp: { learningRate: number; weightDecay: number; beta1: number; beta2: number; eps: number; beta1_t: number; beta2_t: number },
  ): Promise<void> {
    const device = this.device!;
    const pipeline = this._adamwPipeline!;
    const params = this.model.parameters();
    const { learningRate: lr, weightDecay: wd, beta1, beta2, eps, beta1_t, beta2_t } = hp;

    for (let pi = 0; pi < params.length; pi++) {
      const p = params[pi]!;
      const mom = this._moments![pi]!;
      const paramBuf = createStorageBuffer(device, p.data, true);
      const gradBuf = createStorageBuffer(device, grads[pi]!.data, false);
      const mBuf = createStorageBuffer(device, mom.m, true);
      const vBuf = createStorageBuffer(device, mom.v, true);
      const uni = createUniformBuffer(
        device,
        packAdamParams(p.numel, lr, beta1, beta2, eps, wd, beta1_t, beta2_t),
      );
      const bg = createBindGroup(device, pipeline, [uni, paramBuf, gradBuf, mBuf, vBuf]);
      dispatchKernel(device, pipeline, bg, [cdiv(p.numel, 256), 1, 1]);

      p.data.set((await readBuffer(device, paramBuf, p.numel * 4)).subarray(0, p.numel));
      mom.m.set((await readBuffer(device, mBuf, p.numel * 4)).subarray(0, p.numel));
      mom.v.set((await readBuffer(device, vBuf, p.numel * 4)).subarray(0, p.numel));

      paramBuf.destroy();
      gradBuf.destroy();
      mBuf.destroy();
      vBuf.destroy();
      uni.destroy();
    }
  }
}
