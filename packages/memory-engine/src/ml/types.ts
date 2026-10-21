/**
 * ml/types.ts — the ML task catalog contract (docs/ml-task-catalog.md §3).
 *
 * One bounded question Evermind answers without an LLM is ONE data row: how its
 * input becomes a vector, which engines may answer it (in preference order), how
 * it trains, and what its output means. Pure types — no vscode, no DOM, no network.
 */

/** The shape of the question. */
export type MlTaskKind = "classify" | "regress" | "cluster" | "anomaly" | "rank";

/** Pure-TypeScript algorithms that run everywhere (`ml/classical/`). */
export type ClassicalAlgorithm =
  | "knn"
  | "naive-bayes"
  | "logistic"
  | "linear"
  | "kmeans"
  | "robust-z"
  | "ewma"
  | "isolation-forest"
  | "gbm-small";

/** An engine that can answer a task. Cloud engines are a port the HOST supplies. */
export type MlEngineRef =
  | { tier: "local"; engine: "classical"; algorithm: ClassicalAlgorithm }
  | { tier: "local"; engine: "head"; head: "mlp" | "linear" }
  | { tier: "local"; engine: "onnx"; model: string }
  | { tier: "cloud"; engine: "workers-ai"; model: string };

/**
 * Which engines a run may use — the person's inference setting.
 *  - `local-only`: local tiers only, never the network.
 *  - `hybrid`: local first, the cloud port when no local engine can answer (or the
 *    local answer is below the caller's confidence floor).
 *  - `specific`: exactly the engine the caller names.
 */
export type InferenceMode = "local-only" | "hybrid" | "specific";

/** What an engine returns before the task decodes it. */
export interface MlPrediction {
  /**
   * classify: one probability per label; regress: the predicted value(s);
   * cluster: `[clusterId]`; anomaly: `[score]` where ≥ 1 is anomalous;
   * rank: one score per candidate.
   */
  scores: number[];
  /** Calibrated confidence in [0, 1]. */
  confidence: number;
}

/** A fitted model that answers on a feature vector. */
export interface MlPredictor {
  predict(x: Float32Array): MlPrediction;
  /** Trainable state as one flat vector — present on models that federate. */
  exportParams?(): Float32Array;
  /** Load a flat vector produced by {@link exportParams} (e.g. after a merge). */
  importParams?(params: ArrayLike<number>): void;
}

/** How an input becomes a vector: an embedded text part, numeric fields, or both. */
export interface FeatureSpec<I> {
  /** Text to embed (needs an embedder in the run context). */
  text?: (input: I) => string;
  /** Numeric fields, already on a stable scale (so federated weights mean the same thing everywhere). */
  numeric?: (input: I) => number[];
  /** Names of the numeric fields, index-aligned — documentation and Insights labels. */
  numericNames?: readonly string[];
}

/** One labelled training example in feature space. */
export interface MlSample {
  x: Float32Array;
  /** classify: class index; regress: target value; cluster/anomaly: ignored. */
  y: number;
  /** Sample weight (default 1). */
  weight?: number;
}

/** How a task trains. Absent on the task = rule-based or pre-trained only. */
export interface TrainSpec {
  /** Fewer labelled examples than this and the task stays untrained. */
  minSamples: number;
  /** Fraction held out for the eval gate (default 0.2). */
  holdoutFraction?: number;
  /**
   * A task-specific fit (e.g. a composite of two classical detectors). When absent
   * the trainer fits the engine named by the run (`ml/train.ts`).
   */
  fit?: (samples: readonly MlSample[], engine: MlEngineRef) => MlPredictor;
  /** Who trains it when not `ml/train.ts` (e.g. `LimbicTrainer` for `limbic.affect`). */
  trainedBy?: string;
}

/** What a task's answer means. */
export interface OutputSpec<O> {
  /** Label set for classify tasks (index-aligned with `MlPrediction.scores`). */
  labels?: readonly string[];
  /** Decode an engine prediction into the task's output. */
  decode(p: MlPrediction): O;
}

/** One bounded question Evermind can answer without an LLM. DATA, not a class tree. */
export interface MlTask<I = unknown, O = unknown> {
  kind: MlTaskKind;
  /** Stable id, e.g. `experience.quality`. */
  id: string;
  /** One line on what it answers. */
  description: string;
  features: FeatureSpec<I>;
  /** In preference order; filtered by the inference mode. */
  engines: readonly MlEngineRef[];
  train?: TrainSpec;
  output: OutputSpec<O>;
}

/** Task ids: lowercase dotted segments, e.g. `run.local-success`. */
export const ML_TASK_ID_RE = /^[a-z][a-z0-9-]*(\.[a-z][a-z0-9-]*)+$/;

/** A stable string key for an engine ref, e.g. `local:classical:logistic`. */
export function engineKey(ref: MlEngineRef): string {
  switch (ref.engine) {
    case "classical":
      return `local:classical:${ref.algorithm}`;
    case "head":
      return `local:head:${ref.head}`;
    case "onnx":
      return `local:onnx:${ref.model}`;
    case "workers-ai":
      return `cloud:workers-ai:${ref.model}`;
  }
}
