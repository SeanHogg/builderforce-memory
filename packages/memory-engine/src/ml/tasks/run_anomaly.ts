/**
 * ml/tasks/run_anomaly.ts — `run.anomaly`: a run that cost, took or failed far
 * outside its normal range. Flags runs to leave out of training and to surface as
 * Insights.
 *
 * Two detectors, one answer: robust z (median/MAD over the whole history — "far
 * from normal") and EWMA (the recent trend — "drifting away from lately"). The
 * score is the larger of the two, so either one can flag.
 */
import type { Run } from "../../experience/types.js";
import { EwmaModel, fitEwma, type EwmaOptions } from "../classical/ewma.js";
import { anomalyConfidence, fitRobustZ, RobustZModel, type RobustZOptions } from "../classical/robust_z.js";
import { logScale } from "../features.js";
import type { MlPrediction, MlPredictor, MlSample, MlTask } from "../types.js";

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

/** Robust z over the history, combined with EWMA over its recent trend. */
export class RunAnomalyDetector implements MlPredictor {
  constructor(
    readonly population: RobustZModel,
    readonly trend: EwmaModel,
  ) {}

  /** Which metrics are out of range, per detector (z / threshold, ≥ 1 = out of range). */
  explain(x: ArrayLike<number>): { metric: (typeof RUN_ANOMALY_METRICS)[number]; population: number; trend: number }[] {
    const pz = this.population.zScores(x);
    const tz = this.trend.zScores(x);
    return RUN_ANOMALY_METRICS.map((metric, i) => ({
      metric,
      population: pz[i]! / this.population.threshold,
      trend: tz[i]! / this.trend.threshold,
    }));
  }

  /** Fold a finished, non-anomalous run into the trend. */
  observe(x: ArrayLike<number>): void {
    this.trend.update(x);
  }

  predict(x: Float32Array): MlPrediction {
    const score = Math.max(this.population.predict(x).scores[0]!, this.trend.predict(x).scores[0]!);
    return { scores: [score], confidence: anomalyConfidence(score) };
  }
}

/** Fit both detectors on a chronological history of runs (as feature vectors). */
export function fitRunAnomaly(samples: readonly MlSample[], opts: { robustZ?: RobustZOptions; ewma?: EwmaOptions } = {}): RunAnomalyDetector {
  return new RunAnomalyDetector(fitRobustZ(samples, opts.robustZ), fitEwma(samples, opts.ewma));
}

export const RUN_ANOMALY_TASK: MlTask<RunMetrics, RunAnomaly> = {
  kind: "anomaly",
  id: "run.anomaly",
  description: "Did this run cost, take or fail far outside its normal range?",
  features: { numeric: runAnomalyFeatures, numericNames: RUN_ANOMALY_METRICS },
  engines: [
    { tier: "local", engine: "classical", algorithm: "robust-z" },
    { tier: "local", engine: "classical", algorithm: "ewma" },
  ],
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
