/**
 * ml/tasks/check_scheduler.ts — `run.check-scheduler`: will running this optional
 * expensive check (a benchmark, diff coverage, mutation testing) on this run be
 * DECISIVE for a verdict? The question behind Builderforce.ai's dynamic-tier
 * scheduler (PRD 41 D9.1), which runs the check when P × value ≥ cost.
 *
 * Model versions:
 *  - `beta-cell.1` (v1, the host's own): a hierarchical Beta–Bernoulli posterior per
 *    (check × the run's sorted dimensions-touched cell) of one repository, shrunk
 *    toward the check's repository rate, sampled by Thompson (the builderforce-memory
 *    `stats` module). Conjugate and exact, and stable at the 20–500 runs a device
 *    holds; it needs no engine here, so the row lists the upgrade's.
 *  - the later upgrade, pooled across consenting tenants: the logistic engine below
 *    over {@link CHECK_SCHEDULER_FEATURES}. At per-device sizes it would be
 *    under-determined, which is why v1 is the Beta cell.
 *
 * A host row: not in `BUILTIN_ML_TASKS` (no local runner asks it yet); a host that
 * trains the upgrade registers it (`createMlRegistry().register(CHECK_SCHEDULER_TASK)`).
 *
 * Also the package's light `./ml/tasks/check-scheduler` subpath: browser and Worker bundles read the card from here
 * without the engine root. Keep its imports to leaves (`scale.ts`, types) — `tests/regression.checkSchedulerSubpath`.
 */
import { logScale } from "../scale.js";
import type { MlPrediction, MlTask } from "../types.js";

/** The optional checks the scheduler decides. */
export type ScheduledCheck = "benchmark" | "diff-coverage" | "mutation";

/** What the scheduler knows about a run before any optional check runs. */
export interface CheckSchedulerInput {
  check: ScheduledCheck;
  /** The run's dimensions whose static verdict was not `same` (`security`, `performance`, `quality`, `ilities.*`). */
  dimensionsTouched: readonly string[];
  blastRadiusBand: "low" | "medium" | "high";
  /** The run-risk probability in [0, 1], when a risk model exists. */
  risk?: number;
  changedFiles: number;
}

export const CHECK_SCHEDULER_FEATURES = [
  "check.benchmark",
  "check.diffCoverage",
  "check.mutation",
  "touched.security",
  "touched.performance",
  "touched.quality",
  "touched.other",
  "blast.medium",
  "blast.high",
  "risk",
  "changedFiles",
] as const;

/**
 * Fixed scales — the same on every machine, so pooled weights mean the same thing everywhere. The remaining budget is
 * NOT a feature: it is a constraint on the decision (a check that cannot finish is not run), not a predictor of whether
 * the check would be decisive. A missing risk is imputed at 0.5 (see the card's known limits).
 */
export function checkSchedulerFeatures(i: CheckSchedulerInput): number[] {
  const touched = new Set(i.dimensionsTouched);
  const other = [...touched].some((d) => d !== "security" && d !== "performance" && d !== "quality");
  return [
    i.check === "benchmark" ? 1 : 0,
    i.check === "diff-coverage" ? 1 : 0,
    i.check === "mutation" ? 1 : 0,
    touched.has("security") ? 1 : 0,
    touched.has("performance") ? 1 : 0,
    touched.has("quality") ? 1 : 0,
    other ? 1 : 0,
    i.blastRadiusBand === "medium" ? 1 : 0,
    i.blastRadiusBand === "high" ? 1 : 0,
    Math.max(0, Math.min(1, i.risk ?? 0.5)),
    logScale(i.changedFiles, 1000),
  ];
}

export interface CheckSchedulerOutput {
  decisive: boolean;
  /** P(the check is decisive for a verdict). */
  probability: number;
}

export const CHECK_SCHEDULER_TASK: MlTask<CheckSchedulerInput, CheckSchedulerOutput> = {
  kind: "classify",
  id: "run.check-scheduler",
  description: "Will running this optional check be decisive for a verdict of the run?",
  features: { numeric: checkSchedulerFeatures, numericNames: CHECK_SCHEDULER_FEATURES },
  engines: [{ tier: "local", engine: "classical", algorithm: "logistic" }],
  // The pooled upgrade's gate (mirrors the run-risk gate): far beyond one device's history, which v1 serves.
  train: { minSamples: 2000 },
  output: {
    labels: ["not-decisive", "decisive"],
    decode: (p: MlPrediction) => ({ decisive: (p.scores[1] ?? 0) >= 0.5, probability: p.scores[1] ?? 0 }),
  },
};

/**
 * The model card of the version a host runs today (PRD 41 D12): what `scheduledBy.modelId` /
 * `modelVersion` name on every decision.
 */
export const CHECK_SCHEDULER_MODEL_CARD = {
  taskId: CHECK_SCHEDULER_TASK.id,
  modelVersion: "beta-cell.1",
  intendedUse: "Decide whether an opted-in, optional dynamic check runs; never a verdict (D0).",
  dataWindow: "The newest 500 runs of one repository, on the device.",
  /** THE source: observations of a check in a repository below which the host's opt-in default decides. */
  minObservations: 20,
  /** THE source: the seeded share of sampled decisions that run regardless (and record what v1 would have done). */
  explorationShare: 0.1,
  label: "Leave-one-out: the check changed a dimension's verdict, not counting a move out of not_measured.",
  knownLimits: [
    "Context is the dimensions-touched cell only; risk, blast radius and files changed wait for the pooled logistic upgrade.",
    "Per-device history: a new repository runs on the opt-in default until 20 observations.",
    "Upgrade features: a run without a risk model has its risk imputed at 0.5 (risk ?? 0.5), which reads as 'average', not 'unknown'.",
  ],
} as const;
