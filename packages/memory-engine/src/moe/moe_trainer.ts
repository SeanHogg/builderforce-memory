/**
 * moe_trainer.ts — AdamW training loop for {@link SharedExpertMoE}.
 *
 * Makes "train your own Evermind AI" real: given labelled (input → target)
 * samples, runs minibatch AdamW over the flat parameters with the load-balancing
 * auxiliary loss mixed in (so the router spreads load instead of collapsing onto
 * a few experts). Pure CPU, deterministic given a seeded model — the same loop a
 * WebGPU optimiser kernel would accelerate.
 */

import { SharedExpertMoE } from "./moe_model.js";
import { AdamW } from "../optim/adamw.js";

export interface MoESample {
  input: ArrayLike<number>;
  target: ArrayLike<number>;
}

export interface MoETrainOptions {
  /** Learning rate. Default 0.01. */
  lr?: number;
  /** AdamW β1. Default 0.9. */
  beta1?: number;
  /** AdamW β2. Default 0.999. */
  beta2?: number;
  /** AdamW ε. Default 1e-8. */
  eps?: number;
  /** Decoupled weight decay. Default 0. */
  weightDecay?: number;
  /** Weight of the load-balancing auxiliary loss. Default 0.01. */
  auxWeight?: number;
  /** Minibatch size. Default = all samples (full batch). */
  batchSize?: number;
  /** Passes over the dataset. Default 1. */
  epochs?: number;
}

export interface MoEEpochResult {
  /** Mean per-sample task (MSE·½) loss over the epoch. */
  loss: number;
  /** Load-balancing auxiliary loss at the end of the epoch (≈1 balanced … E collapsed). */
  auxLoss: number;
}

/**
 * AdamW optimiser over a model's flat parameter list. State (m, v) is keyed by
 * parameter index and persists across {@link step} calls.
 */
export class MoETrainer {
  private readonly adam: AdamW;
  private readonly opt: Required<MoETrainOptions>;

  constructor(
    private readonly model: SharedExpertMoE,
    options: MoETrainOptions = {},
  ) {
    this.opt = {
      lr: options.lr ?? 0.01,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
      auxWeight: options.auxWeight ?? 0.01,
      batchSize: options.batchSize ?? 0,
      epochs: options.epochs ?? 1,
    };
    this.adam = new AdamW(model, this.opt);
  }

  /** Train for the configured epochs. Returns the per-epoch loss history. */
  fit(samples: MoESample[]): MoEEpochResult[] {
    const history: MoEEpochResult[] = [];
    for (let e = 0; e < this.opt.epochs; e++) history.push(this.runEpoch(samples));
    return history;
  }

  private runEpoch(samples: MoESample[]): MoEEpochResult {
    const batchSize = this.opt.batchSize > 0 ? this.opt.batchSize : samples.length;
    const { numExperts, modelDim } = this.model.config;
    let epochLoss = 0;
    let lastAux = 0;

    for (let start = 0; start < samples.length; start += batchSize) {
      const batch = samples.slice(start, start + batchSize);
      this.model.zeroGrad();

      // Forward + task backward, retaining (x, probs) for the batch aux gradient.
      const xs: Float32Array[] = [];
      const probsList: Float32Array[] = [];
      const counts = new Float32Array(numExperts);
      let batchLoss = 0;

      for (const s of batch) {
        const f = this.model.forward(s.input);
        const dOut = new Float32Array(modelDim);
        for (let d = 0; d < modelDim; d++) {
          const diff = f.output[d]! - (s.target[d] ?? 0);
          dOut[d] = diff;
          batchLoss += 0.5 * diff * diff;
        }
        this.model.backward(dOut, f.cache);
        xs.push(f.cache.x);
        probsList.push(f.route.probs);
        for (const ex of f.route.experts) counts[ex] = counts[ex]! + 1;
      }

      // Load-balancing aux gradient (batch-level): f = dispatch fractions.
      const dispatched = counts.reduce((a, b) => a + b, 0) || 1;
      const fVec = Float32Array.from(counts, (c) => c / dispatched);
      const scale = (this.opt.auxWeight * numExperts) / batch.length;
      for (let i = 0; i < xs.length; i++) {
        this.model.auxGradStep(xs[i]!, probsList[i]!, fVec, scale);
      }

      // Average the task gradient over the batch, then AdamW step.
      this.scaleGradients(1 / batch.length);
      this.adam.step();

      epochLoss += batchLoss;
      lastAux = numExperts * fVec.reduce((sum, f, e) => sum + f * (probsList.length
        ? probsList.reduce((s, p) => s + p[e]!, 0) / probsList.length
        : 0), 0);
    }

    return { loss: epochLoss / Math.max(1, samples.length), auxLoss: lastAux };
  }

  private scaleGradients(k: number): void {
    if (k === 1) return;
    for (const g of this.model.gradients()) {
      for (let i = 0; i < g.data.length; i++) g.data[i] = g.data[i]! * k;
    }
  }
}
