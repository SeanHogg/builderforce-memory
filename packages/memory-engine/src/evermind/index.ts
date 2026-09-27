/**
 * Evermind — the engine half of the Evermind module: the portable `.evermind`
 * artifact plus the learning pipeline every learner shares (adapt, diff wire
 * contract, FedAvg merge, pre/post-merge eval).
 */
export { EvermindModelPackage, PACKAGE_HEADER_BYTES } from "./package.js";
export type {
  EvermindModelManifest,
  EvermindModelCard,
  EvermindModelType,
  EvermindModality,
  PackageMeta,
  ValidationResult,
} from "./package.js";

export { adaptAndDiff, adaptPackage, EVERMIND_ADAPT_MAX_CHARS } from "./adapt.js";
export type { AdaptResult, AdaptedPackage, AdaptOptions, AdaptTokenizer } from "./adapt.js";

export {
  parseDeltaLearnPayload,
  buildDeltaLearnPayload,
  decodeDeltaPayload,
  deltaUnusableReason,
  MAX_DELTA_B64_CHARS,
} from "./delta_wire.js";
export type { DeltaLearnPayload, DeltaParseResult } from "./delta_wire.js";

export { mergeCheckpointDiffs } from "./merge.js";
export type { MergeResult } from "./merge.js";

export { meanEvalLoss } from "./eval.js";
export type { EvalExample, EvalTokenizer } from "./eval.js";
