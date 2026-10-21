/**
 * ml/tasks/local_success.ts — `run.local-success`: will this turn succeed on the
 * local model? Exported for the Hybrid router (Builderforce.ai), which keeps a turn
 * local when it can and sends it to the cloud when it can't.
 *
 * It LEARNS FROM THE CLOUD (docs/ml-task-catalog.md §6): in Hybrid, a turn that
 * fell back to — or was also served by — the cloud is a training label for the
 * local predictor, alongside local outcomes. See {@link localSuccessLabel}.
 */
import { clampRatio, logScale } from "../features.js";
import type { MlTrainExample } from "../train.js";
import type { MlPrediction, MlTask } from "../types.js";

/** What the router knows about a turn before it runs, plus the local model's served context. */
export interface LocalTurnFeatures {
  promptTokens: number;
  /** Tokens of context (history, files, memory) the turn carries. */
  contextTokens: number;
  /** The local model's served context window, in tokens. */
  servedContextTokens: number;
  /** Tools advertised to the turn. */
  tools: number;
  /** Position of the turn in its conversation (0 = first). */
  turnIndex: number;
  attachments?: number;
  /** Recent local success rate for this person/workload in [0, 1], when known. */
  priorLocalSuccessRate?: number;
  /** The turn is mostly code (edits, diffs, stack traces). */
  codeHeavy?: boolean;
}

export const LOCAL_SUCCESS_FEATURES = [
  "promptTokens",
  "contextTokens",
  "contextFill",
  "overflows",
  "tools",
  "turnIndex",
  "attachments",
  "priorLocalSuccessRate",
  "codeHeavy",
] as const;

/** Fixed scales — the same on every machine, so the weights federate. */
export function localSuccessFeatures(t: LocalTurnFeatures): number[] {
  const fill = clampRatio(t.promptTokens + t.contextTokens, t.servedContextTokens, 4);
  return [
    logScale(t.promptTokens, 100_000),
    logScale(t.contextTokens, 100_000),
    fill,
    fill > 1 ? 1 : 0,
    logScale(t.tools, 64),
    logScale(t.turnIndex, 100),
    logScale(t.attachments ?? 0, 16),
    Math.max(0, Math.min(1, t.priorLocalSuccessRate ?? 0.5)),
    t.codeHeavy ? 1 : 0,
  ];
}

export interface LocalSuccess {
  succeeds: boolean;
  /** P(the local model succeeds). */
  probability: number;
}

export const LOCAL_SUCCESS_TASK: MlTask<LocalTurnFeatures, LocalSuccess> = {
  kind: "classify",
  id: "run.local-success",
  description: "Will this turn succeed on the local model?",
  features: { numeric: localSuccessFeatures, numericNames: LOCAL_SUCCESS_FEATURES },
  engines: [
    { tier: "local", engine: "classical", algorithm: "logistic" },
    { tier: "local", engine: "classical", algorithm: "gbm-small" },
  ],
  train: { minSamples: 8 },
  output: {
    labels: ["fails", "succeeds"],
    decode: (p: MlPrediction) => ({ succeeds: (p.scores[1] ?? 0) >= 0.5, probability: p.scores[1] ?? 0 }),
  },
};

export type TurnResult = "succeeded" | "failed";

/** How one routed turn actually went. */
export interface LocalSuccessOutcome {
  turn: LocalTurnFeatures;
  /** The local model's result, when it ran. */
  local?: TurnResult;
  /** The cloud's result, when it ran. */
  cloud?: TurnResult;
  /** The local model ran first and the turn was handed to the cloud. */
  fellBack?: boolean;
}

/** Weight of a cloud failure on a turn the local model never tried — weak evidence the turn was hard. */
export const CLOUD_ONLY_FAILURE_WEIGHT = 0.5;

/**
 * The training label of one outcome, or null when it says nothing about the local model.
 *  - local ran: its result (1 succeeded / 0 failed);
 *  - fell back to the cloud: 0 — the local model could not finish it;
 *  - cloud only, and the cloud failed: 0 at {@link CLOUD_ONLY_FAILURE_WEIGHT};
 *  - cloud only, and the cloud succeeded: no evidence about local → null.
 */
export function localSuccessLabel(o: LocalSuccessOutcome): { label: 0 | 1; weight: number } | null {
  if (o.local) return { label: o.local === "succeeded" ? 1 : 0, weight: 1 };
  if (o.fellBack) return { label: 0, weight: 1 };
  if (o.cloud === "failed") return { label: 0, weight: CLOUD_ONLY_FAILURE_WEIGHT };
  return null;
}

/** Labelled examples from routed outcomes (local AND cloud), oldest first. */
export function localSuccessExamples(outcomes: readonly LocalSuccessOutcome[]): MlTrainExample<LocalTurnFeatures>[] {
  return outcomes.flatMap((o) => {
    const l = localSuccessLabel(o);
    return l ? [{ input: o.turn, label: l.label, weight: l.weight }] : [];
  });
}
