/**
 * Mixture-of-Experts — shared-expert hybrid sparsity for the Evermind generator.
 */

export {
  SharedExpertMoE,
  LoadBalanceAccumulator,
  DEFAULT_MOE_CONFIG,
  DEFAULT_MOE_SEED,
} from "./moe_model.js";
export type { MoEConfig, MoEParam, RouteResult } from "./moe_model.js";

export { MoETrainer } from "./moe_trainer.js";
export type { MoESample, MoETrainOptions, MoEEpochResult } from "./moe_trainer.js";

export { EvermindModelPackage } from "./moe_package.js";
export type {
  EvermindModelManifest,
  EvermindModelCard,
  PackageMeta,
  ValidationResult,
} from "./moe_package.js";
