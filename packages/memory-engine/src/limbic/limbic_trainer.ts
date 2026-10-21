/**
 * limbic_trainer.ts – LimbicTrainer: gradient-based training for the LimbicModel.
 *
 * Trains the affect head to predict affect deltas and reward from
 * (experience embedding, current state) pairs, using full-batch gradient
 * descent with AdamW — through the shared {@link HeadTrainer} (`ml/head_trainer.ts`),
 * which runs the optimiser step on the GPU via WEIGHT_UPDATE_WGSL when a device is
 * supplied and on a numerically-equivalent CPU AdamW otherwise. This file is only
 * the limbic registration: the head adapter and the public trainer surface.
 *
 * The objective is MSE (regression), not the cross-entropy used by the
 * language-model {@link MambaTrainer} — a limbic experience has continuous
 * targets, not a next-token distribution.
 */

import { HeadTrainer, type HeadTrainOptions, type TrainableHead } from "../ml/head_trainer.js";
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

/** The limbic trainer's options are the shared head trainer's. */
export type LimbicTrainOptions = HeadTrainOptions;

/**
 * The limbic model as a {@link TrainableHead}. Each sample is one experience
 * appraisal: cross-experience memory flows through the affective state `s` (fed
 * back by the runtime), not through the hidden scratch `h`, so the hidden state
 * resets per sample. This also makes BPTT(1) exact — there is no truncated carry.
 */
export function limbicHead(model: LimbicModel): TrainableHead<LimbicSample> {
  return {
    parameters: (): LimbicParam[] => model.parameters(),
    gradients: (): LimbicParam[] => model.gradients(),
    zeroGrad: () => model.zeroGrad(),
    accumulate: (s) =>
      model.backwardStep(s.input, model.initHidden(), s.state, s.deltaTarget, s.reward).loss,
    loss: (s) => {
      const { rewardWeight } = model.config;
      const f = model.forward(s.input, model.initHidden(), s.state);
      let loss = 0;
      for (let k = 0; k < f.delta.length; k++) {
        const d = f.delta[k]! - (s.deltaTarget[k] ?? 0);
        loss += 0.5 * d * d;
      }
      const rd = f.reward - s.reward;
      loss += 0.5 * rewardWeight * rd * rd;
      return loss;
    },
  };
}

/** Thin registration of the limbic head over the shared {@link HeadTrainer}. */
export class LimbicTrainer {
  readonly model: LimbicModel;
  readonly device: GPUDevice | null;
  private readonly _trainer: HeadTrainer<LimbicSample>;

  constructor(model: LimbicModel, device: GPUDevice | null = null) {
    this.model = model;
    this.device = device;
    this._trainer = new HeadTrainer(limbicHead(model), device, "LimbicTrainer");
  }

  /** Whether the optimiser step runs on the GPU. */
  get gpuTraining(): boolean {
    return this._trainer.gpuTraining;
  }

  /**
   * Train on a batch of samples for `epochs` passes. Returns the per-epoch mean
   * loss (monotonically decreasing on a learnable mapping).
   */
  train(samples: LimbicSample[], opts: LimbicTrainOptions = {}): Promise<number[]> {
    return this._trainer.train(samples, opts);
  }

  /** Mean MSE loss over samples (no weight update). Hidden resets per sample. */
  evaluate(samples: LimbicSample[]): number {
    return this._trainer.evaluate(samples);
  }
}
