/**
 * ml/head_trainer.ts — HeadTrainer: the ONE gradient trainer for a small head over
 * an embedding (extracted from the limbic affect trainer, which is now a thin
 * registration over it).
 *
 * Full-batch gradient descent with AdamW. When a GPUDevice is supplied the
 * optimiser step runs on the GPU via the shared WEIGHT_UPDATE_WGSL kernel; with
 * no device it uses a numerically-equivalent CPU AdamW, so training works
 * everywhere (CI, Node without @webgpu/node, a laptop with no GPU).
 *
 * The head owns its forward/backward and its loss (MSE for regression, BCE /
 * softmax cross-entropy for classification); this file owns only the loop:
 * zero → accumulate per sample → average → clip → AdamW step.
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

/** A named trainable tensor (flat row-major Float32Array). */
export interface HeadParam {
  name: string;
  data: Float32Array;
  numel: number;
}

/** What a head must expose to be trained by {@link HeadTrainer}. */
export interface TrainableHead<S> {
  /** Trainable parameters, in a stable order. */
  parameters(): HeadParam[];
  /** Gradient buffers, index-aligned with {@link parameters}. */
  gradients(): HeadParam[];
  zeroGrad(): void;
  /** Accumulate gradients for one sample into {@link gradients}; returns its loss. */
  accumulate(sample: S): number;
  /** Forward-only loss of one sample. */
  loss(sample: S): number;
}

export interface HeadTrainOptions {
  learningRate?: number;
  epochs?: number;
  weightDecay?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  /** Max global gradient L2 norm before the optimiser step. Default 1.0. */
  maxGradNorm?: number;
  /** Per-epoch hook. `gradNorm` is the pre-clip gradient L2 norm this epoch (the
   *  instability early-warning) — computed CPU-side so it's always available. */
  onEpochEnd?: ((epoch: number, loss: number, gradNorm?: number) => void) | null;
}

interface AdamMoment {
  m: Float32Array;
  v: Float32Array;
}

interface AdamHyper {
  learningRate: number;
  weightDecay: number;
  beta1: number;
  beta2: number;
  eps: number;
  beta1_t: number;
  beta2_t: number;
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

export class HeadTrainer<S> {
  readonly head: TrainableHead<S>;
  readonly device: GPUDevice | null;
  private _moments: AdamMoment[] | null = null;
  private _step = 0;
  private readonly _adamwPipeline: GPUComputePipeline | null;

  constructor(
    head: TrainableHead<S>,
    device: GPUDevice | null = null,
    /** Name used in error messages (the registration that owns this trainer). */
    private readonly label: string = "HeadTrainer",
  ) {
    this.head = head;
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
    this._moments = this.head.parameters().map((p) => ({
      m: new Float32Array(p.numel),
      v: new Float32Array(p.numel),
    }));
  }

  /**
   * Train on `samples` for `epochs` passes; returns the per-epoch mean loss.
   * Full-batch: grads accumulate over every sample, are averaged, clipped, then
   * applied once per epoch.
   */
  async train(samples: S[], opts: HeadTrainOptions = {}): Promise<number[]> {
    if (samples.length === 0) throw new Error(`${this.label}.train: no samples`);
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
      this.head.zeroGrad();
      let epochLoss = 0;
      for (const s of samples) epochLoss += this.head.accumulate(s);

      const grads = this.head.gradients();
      const invN = 1 / samples.length;
      for (const g of grads) {
        for (let i = 0; i < g.data.length; i++) g.data[i]! *= invN;
      }

      const gradNorm = clipGradients(grads, maxGradNorm);

      this._step++;
      const beta1_t = Math.pow(beta1, this._step);
      const beta2_t = Math.pow(beta2, this._step);
      const hp: AdamHyper = { learningRate, weightDecay, beta1, beta2, eps, beta1_t, beta2_t };
      if (this.gpuTraining) {
        await this._adamwStepGpu(grads, hp);
      } else {
        this._adamwStepCpu(grads, hp);
      }

      const avg = epochLoss / samples.length;
      losses.push(avg);
      if (onEpochEnd) onEpochEnd(epoch + 1, avg, gradNorm);
    }
    return losses;
  }

  /** Mean loss over samples (no weight update). 0 for an empty set. */
  evaluate(samples: S[]): number {
    if (samples.length === 0) return 0;
    let total = 0;
    for (const s of samples) total += this.head.loss(s);
    return total / samples.length;
  }

  private _adamwStepCpu(grads: HeadParam[], hp: AdamHyper): void {
    const params = this.head.parameters();
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
  private async _adamwStepGpu(grads: HeadParam[], hp: AdamHyper): Promise<void> {
    const device = this.device!;
    const pipeline = this._adamwPipeline!;
    const params = this.head.parameters();
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

/** Clip grads to `maxNorm` in place; returns the PRE-clip L2 norm (the signal). */
function clipGradients(grads: HeadParam[], maxNorm: number): number {
  let normSq = 0;
  for (const g of grads) for (let i = 0; i < g.data.length; i++) normSq += g.data[i]! * g.data[i]!;
  const norm = Math.sqrt(normSq);
  if (norm > maxNorm && norm > 0) {
    const scale = maxNorm / norm;
    for (const g of grads) for (let i = 0; i < g.data.length; i++) g.data[i]! *= scale;
  }
  return norm;
}

/** Concatenate a head's parameters into one flat vector (the federation unit). */
export function flattenParams(params: readonly HeadParam[]): Float32Array {
  const out = new Float32Array(params.reduce((n, p) => n + p.numel, 0));
  let o = 0;
  for (const p of params) {
    out.set(p.data, o);
    o += p.numel;
  }
  return out;
}

/** Write a flat vector back into a head's parameters (inverse of {@link flattenParams}). */
export function assignParams(params: readonly HeadParam[], flat: ArrayLike<number>): void {
  const total = params.reduce((n, p) => n + p.numel, 0);
  if (flat.length !== total) throw new Error(`assignParams: expected ${total} values, got ${flat.length}`);
  let o = 0;
  for (const p of params) {
    for (let i = 0; i < p.numel; i++) p.data[i] = flat[o + i]!;
    o += p.numel;
  }
}
