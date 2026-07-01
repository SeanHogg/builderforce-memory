/**
 * Limbic system – trainable affective/motivational dynamics for agents.
 *
 * The dynamic counterpart to the (static) psychometric personality: where
 * personality sets the homeostatic setpoints, the limbic model learns — in
 * WebGPU — how an agent's affective state moves in response to experience.
 */

export {
  REGION,
  LIMBIC_DIM,
  LIMBIC_DIM_NAMES,
  LIMBIC_STATE_DIM,
  LIMBIC_BOUNDS,
  NEUTRAL_STATE,
  clampDim,
  clampState,
  neutralState,
  stateToRecord,
  recordToState,
  personalitySetpoint,
} from "./regions.js";
export type { Region, LimbicDimName, PersonalityTraits } from "./regions.js";

export { LimbicModel, DEFAULT_LIMBIC_CONFIG, DEFAULT_LIMBIC_SEED } from "./limbic_model.js";
export type { LimbicModelConfig, LimbicForward, LimbicParam } from "./limbic_model.js";

export { LimbicTrainer } from "./limbic_trainer.js";
export type { LimbicSample, LimbicTrainOptions } from "./limbic_trainer.js";
