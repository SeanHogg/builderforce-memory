/**
 * Limbic system – trainable affective/motivational dynamics for agents.
 *
 * The dynamic counterpart to the (static) psychometric personality: where
 * personality sets the homeostatic setpoints, the heuristic regions
 * (`affect.ts`) move the live state in response to experience and the limbic
 * model learns — in WebGPU — how it moves. The one limbic implementation; the
 * dependency-free half ships alone as `@seanhogg/builderforce-memory-engine/limbic`.
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
  limbicSetpoints,
  personalitySetpoint,
} from "./regions.js";
export type { Region, LimbicDimName, PersonalityTraits } from "./regions.js";

export {
  clampLimbicDim,
  neutralLimbicState,
  meanLimbicSetpoints,
  applyLimbicDelta,
  appraiseAmygdala,
  homeostasis,
  thalamusGate,
  basalGangliaExploreBias,
  basalGangliaSelect,
  compileLimbicState,
  buildLimbicBlock,
  LIMBIC_BLOCK_HEADER,
  appraiseTask,
} from "./affect.js";
export type {
  LimbicState,
  LimbicSetpoints,
  LimbicDelta,
  LimbicEvent,
  LimbicExecParams,
  CompiledLimbic,
} from "./affect.js";

export { LimbicModel, DEFAULT_LIMBIC_CONFIG, DEFAULT_LIMBIC_SEED } from "./limbic_model.js";
export type { LimbicModelConfig, LimbicForward, LimbicParam } from "./limbic_model.js";

export { LimbicTrainer } from "./limbic_trainer.js";
export type { LimbicSample, LimbicTrainOptions } from "./limbic_trainer.js";
