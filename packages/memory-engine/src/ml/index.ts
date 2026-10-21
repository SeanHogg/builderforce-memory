/**
 * ml — the ML task catalog (docs/ml-task-catalog.md): small, purpose-built models
 * for bounded questions, answered on the person's own machine without a GPU or an
 * LLM call. ONE registry, ONE runner, ONE trainer seam; trained heads federate on
 * the existing delta wire. Pure and portable: no vscode, no DOM, no network.
 */
export type {
  MlTask,
  MlTaskKind,
  MlEngineRef,
  ClassicalAlgorithm,
  InferenceMode,
  MlPrediction,
  MlPredictor,
  FeatureSpec,
  MlSample,
  TrainSpec,
  OutputSpec,
} from "./types.js";
export { engineKey, ML_TASK_ID_RE } from "./types.js";

export { MlTaskRegistry } from "./registry.js";
export {
  runMlTask,
  candidateEngines,
  InMemoryMlModels,
  MlDecisionLog,
  MlEngineUnavailableError,
} from "./run.js";
export type { MlRunContext, MlResult, MlModelStore, MlCloudPort, MlDecision, MlDecisionSink, MlSkipReason } from "./run.js";

export { trainMlTask, mlEvalLoss } from "./train.js";
export type { MlTrainExample, MlTrainOptions, MlTrainResult } from "./train.js";

export { HeadTrainer, flattenParams, assignParams } from "./head_trainer.js";
export type { HeadParam, TrainableHead, HeadTrainOptions } from "./head_trainer.js";
export { DenseHead } from "./dense_head.js";
export type { DenseHeadConfig } from "./dense_head.js";

export { extractFeatures, extractFeatureMatrix, logScale, clampRatio } from "./features.js";

export * from "./classical/index.js";

export {
  headCheckpoint,
  headParams,
  mlHeadEntry,
  mlHeadParams,
  mlHeadUnusableReason,
  packageMlHead,
  withPackageMlHead,
  buildMlHeadDelta,
  mergeMlHeadDeltas,
} from "./federate.js";
export type { EvermindMlHead, MlHeadMergeOutcome } from "./federate.js";

export * from "./tasks/index.js";
