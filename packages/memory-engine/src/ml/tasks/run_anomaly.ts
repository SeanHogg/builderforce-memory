/**
 * ml/tasks/run_anomaly.ts — `run.anomaly`: a run that cost, took or failed far
 * outside its normal range. Flags runs to leave out of training and to surface as
 * Insights.
 *
 * ONE engine, `anomaly-composite`: robust z (median/MAD over the whole history —
 * "far from normal") and EWMA (the recent trend — "drifting away from lately"),
 * scored as the larger of the two so either one can flag. The row lists exactly
 * that engine, so the key it is stored under is the key the runner asks for.
 */
import type { Run } from "../../experience/types.js";
import { AnomalyCompositeModel, type AnomalyCompositeOptions } from "../classical/anomaly_composite.js";
import { fitEwma } from "../classical/ewma.js";
import { fitRobustZ } from "../classical/robust_z.js";
import { logScale } from "../features.js";
import type { MlPrediction, MlSample, MlTask } from "../types.js";

/** What a run cost and how it went — the host supplies cost; the rest is in the run log. */
export interface RunMetrics {
  costUsd: number;
  durationMs: number;
  toolFailures: number;
}

export interface RunAnomaly {
  anomalous: boolean;
  /** Normalised score: ≥ 1 is anomalous. */
  score: number;
}

export const RUN_ANOMALY_METRICS = ["cost", "duration", "toolFailures"] as const;

/** The fixed log references: cost in cents ($10), duration in ms (one hour), tool failures (50). */
const COST_REF_CENTS = 1000;
const DURATION_REF_MS = 3_600_000;
const FAILURES_REF = 50;

/** Fixed log scales (a $10 run, a one-hour run, 50 failures map to 1). */
export function runAnomalyFeatures(m: RunMetrics): number[] {
  return [
    logScale(m.costUsd * 100, COST_REF_CENTS),
    logScale(m.durationMs, DURATION_REF_MS),
    logScale(m.toolFailures, FAILURES_REF),
  ];
}

/**
 * The multiplicative jump in cost or duration that flags against a constant history.
 * A constant history has MAD = mean-AD = 0 and EWMA σ = 0, so the floor below IS the
 * scale both detectors divide by.
 */
export const RUN_ANOMALY_FLAG_RATIO = 3;

/**
 * The smallest per-metric scale (σ) both detectors use, in the fixed log units above —
 * ONE PER METRIC, because each metric is divided by a different reference: a feature
 * unit is log1p(ref) nats, i.e. log1p(1000) ≈ 6.909 nats for cost but log1p(3.6e6) ≈
 * 15.096 nats for duration. One scalar floor (it was 0.1) therefore meant ~0.69 nats
 * (2×) on cost but ~1.51 nats (4.5×) on duration, and a 10 s → 15 min run (4.5 nats,
 * z ≈ 3.0) slipped under EWMA's 3.
 *
 * Derivation, cost and duration: for values well above 1, a k× jump moves log1p by
 * ≈ ln k nats = ln k / log1p(ref) feature units. The composite flags at the smaller of
 * the two thresholds (EWMA 3; robust z is 3.5), so the floor that flags a 3× jump is
 *   floor = ln 3 / (3 · log1p(ref))
 *   cost:     1.0986 / (3 · 6.9088)  ≈ 0.0530   (robust z alone flags from ≈ 3.6×)
 *   duration: 1.0986 / (3 · 15.0964) ≈ 0.0243   (robust z alone flags from ≈ 3.6×)
 * A 1.5× wobble (0.405 nats) scores z ≈ 1.1 on either — normal.
 *
 * Tool failures keep the absolute 0.1: from a clean history one or two failures
 * (log1p(1|2)/log1p(50) ≈ 0.18 | 0.28) stay normal (z ≈ 1.8 / 2.8) and three or more
 * (≈ 0.35) flag. (0.05 was too tight: it flagged a single failure.)
 */
export const RUN_ANOMALY_MIN_SCALE: readonly number[] = [
  Math.log(RUN_ANOMALY_FLAG_RATIO) / (3 * Math.log1p(COST_REF_CENTS)),
  Math.log(RUN_ANOMALY_FLAG_RATIO) / (3 * Math.log1p(DURATION_REF_MS)),
  0.1,
];

/** The composite detector with `run.anomaly`'s metric names on its explanation. */
export class RunAnomalyDetector extends AnomalyCompositeModel {
  /** Which metrics are out of range, per detector (z / threshold, ≥ 1 = out of range). */
  explain(x: ArrayLike<number>): { metric: (typeof RUN_ANOMALY_METRICS)[number]; population: number; trend: number }[] {
    const per = this.featureScores(x);
    return RUN_ANOMALY_METRICS.map((metric, i) => ({ metric, ...per[i]! }));
  }
}

/** Fit both detectors on a chronological history of runs (as feature vectors), floored per metric at {@link RUN_ANOMALY_MIN_SCALE}. */
export function fitRunAnomaly(samples: readonly MlSample[], opts: AnomalyCompositeOptions = {}): RunAnomalyDetector {
  return new RunAnomalyDetector(
    fitRobustZ(samples, { ...opts.robustZ, minScale: opts.robustZ?.minScale ?? RUN_ANOMALY_MIN_SCALE }),
    fitEwma(samples, { ...opts.ewma, minScale: opts.ewma?.minScale ?? RUN_ANOMALY_MIN_SCALE }),
  );
}

export const RUN_ANOMALY_TASK: MlTask<RunMetrics, RunAnomaly> = {
  kind: "anomaly",
  id: "run.anomaly",
  description: "Did this run cost, take or fail far outside its normal range?",
  features: { numeric: runAnomalyFeatures, numericNames: RUN_ANOMALY_METRICS },
  engines: [{ tier: "local", engine: "classical", algorithm: "anomaly-composite" }],
  // The task's own fit: the same composite, with the metric floor and named explanation.
  train: { minSamples: 5, fit: (samples) => fitRunAnomaly(samples) },
  output: { decode: (p: MlPrediction) => ({ anomalous: (p.scores[0] ?? 0) >= 1, score: p.scores[0] ?? 0 }) },
};

/** Duration and tool-failure count of an experience run, with the host's cost. */
export function experienceRunMetrics(run: Run, costUsd = 0): RunMetrics {
  return {
    costUsd,
    durationMs: run.endedAt != null ? Math.max(0, run.endedAt - run.startedAt) : 0,
    toolFailures: run.steps.filter((s) => s.outcome === "failed").length,
  };
}
