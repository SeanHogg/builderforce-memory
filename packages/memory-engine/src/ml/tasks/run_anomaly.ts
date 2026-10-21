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

/** Fixed log scales (a $10 run, a one-hour run, 50 failures map to 1). */
export function runAnomalyFeatures(m: RunMetrics): number[] {
  return [logScale(m.costUsd * 100, 1000), logScale(m.durationMs, 3_600_000), logScale(m.toolFailures, 50)];
}

/**
 * The smallest per-metric scale (σ) both detectors use, in the fixed log units above.
 * A history with no tool failures has MAD = mean-AD = 0 and EWMA σ = 0, so without an
 * absolute floor ONE failure (log1p(1)/log1p(50) ≈ 0.18) scored as a huge outlier and
 * every run with any failure was flagged. At 0.1, one or two failures after a clean
 * history stay normal (z ≈ 1.8 / 2.8) and three or more flag (z ≥ 3.5); on cost, 0.1
 * is ~0.7 nats, so a constant-cost history flags a ~10× jump. (0.05 was too tight: it
 * still flagged a single failure.)
 */
export const RUN_ANOMALY_MIN_SCALE = 0.1;

/** The composite detector with `run.anomaly`'s metric names on its explanation. */
export class RunAnomalyDetector extends AnomalyCompositeModel {
  /** Which metrics are out of range, per detector (z / threshold, ≥ 1 = out of range). */
  explain(x: ArrayLike<number>): { metric: (typeof RUN_ANOMALY_METRICS)[number]; population: number; trend: number }[] {
    const per = this.featureScores(x);
    return RUN_ANOMALY_METRICS.map((metric, i) => ({ metric, ...per[i]! }));
  }
}

/** Fit both detectors on a chronological history of runs (as feature vectors), floored at {@link RUN_ANOMALY_MIN_SCALE}. */
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
