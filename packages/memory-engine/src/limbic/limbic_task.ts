/**
 * limbic/limbic_task.ts — `limbic.affect`, the ML catalog's first row, registered
 * by limbic/ itself.
 *
 * Nothing about the affect head changes: the row describes it (input = experience
 * embedding + affective state, output = bounded affect delta + reward), the
 * predictor adapter lets the ONE runner answer with an existing `LimbicModel`, and
 * training stays `LimbicTrainer` (itself a thin registration over `HeadTrainer`).
 * Its flat parameters are what federate when the head travels with the model.
 */
import { assignParams, flattenParams } from "../ml/head_trainer.js";
import type { MlPrediction, MlPredictor, MlTask } from "../ml/types.js";
import type { LimbicModel } from "./limbic_model.js";

export interface LimbicAffectInput {
  /** Experience embedding (length = model.inputDim). */
  embedding: ArrayLike<number>;
  /** Current affective state (length = model.stateDim). */
  state: ArrayLike<number>;
}

export interface LimbicAffectOutput {
  /** Bounded affect delta in (-1, 1) per state dim. */
  delta: Float32Array;
  /** Reward prediction. */
  reward: number;
}

export const LIMBIC_AFFECT_TASK: MlTask<LimbicAffectInput, LimbicAffectOutput> = {
  kind: "regress",
  id: "limbic.affect",
  description: "How an experience moves the agent's affective state, and how rewarding it was.",
  features: { numeric: (i) => [...Array.from(i.embedding), ...Array.from(i.state)] },
  engines: [{ tier: "local", engine: "head", head: "mlp" }],
  train: { minSamples: 1, trainedBy: "LimbicTrainer" },
  output: {
    decode: (p: MlPrediction) => ({
      delta: Float32Array.from(p.scores.slice(0, -1)),
      reward: p.scores[p.scores.length - 1] ?? 0,
    }),
  },
};

/**
 * Answer `limbic.affect` with `model`. The feature vector is `[embedding, state]`;
 * the hidden scratch resets per call, exactly as the trainer treats each sample.
 * Confidence is 1: a deterministic regression head has no calibrated uncertainty.
 */
export function limbicAffectPredictor(model: LimbicModel): MlPredictor {
  const { inputDim, stateDim } = model.config;
  return {
    predict(x: Float32Array): MlPrediction {
      if (x.length !== inputDim + stateDim) {
        throw new Error(`limbic.affect: expected ${inputDim + stateDim} features (embedding ${inputDim} + state ${stateDim}), got ${x.length}`);
      }
      const f = model.forward(x.subarray(0, inputDim), model.initHidden(), x.subarray(inputDim));
      return { scores: [...Array.from(f.delta), f.reward], confidence: 1 };
    },
    exportParams: () => flattenParams(model.parameters()),
    importParams: (params) => assignParams(model.parameters(), params),
  };
}
