/**
 * ml/classical/robust_z.ts — robust (median/MAD) z-scores for anomaly detection.
 *
 * Per feature: z = |x − median| / (1.4826 · MAD). Median and MAD are not dragged
 * by the very outliers being looked for, unlike mean and standard deviation. When
 * MAD is 0 (more than half the history identical) the mean absolute deviation
 * stands in, then two floors: a fraction of |median| (relative) and `minScale`
 * (absolute, in the feature's own units). Without the absolute one a history that
 * is all zeros — no tool failures, ever — gets a scale of ~1e-9, and the first
 * small non-zero value scores astronomically.
 *
 * Prediction: `scores[0] = max_i z_i / threshold` — ≥ 1 is anomalous.
 */
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { featureDim, median } from "./math.js";

/** MAD → standard deviation for normally-distributed data. */
const MAD_TO_SD = 1.4826;
/** Mean absolute deviation → standard deviation for normally-distributed data. */
const MEANAD_TO_SD = 1.2533;

export interface RobustZOptions {
  /** z at which a feature counts as anomalous. Default 3.5 (Iglewicz & Hoaglin). */
  threshold?: number;
  /** Smallest scale per feature, as a fraction of |median| (plus 1e-9). Default 0.01. */
  minScaleFraction?: number;
  /**
   * Smallest scale per feature in absolute feature units. Default 0 — the algorithm
   * cannot know the units; a task on a fixed scale sets it (`run.anomaly` does).
   */
  minScale?: number;
}

export class RobustZModel implements MlPredictor {
  constructor(
    readonly center: Float64Array,
    readonly scale: Float64Array,
    readonly threshold: number,
  ) {}

  /** Per-feature robust z-scores (absolute). */
  zScores(x: ArrayLike<number>): number[] {
    return Array.from(this.center, (c, i) => Math.abs((x[i] ?? 0) - c) / this.scale[i]!);
  }

  predict(x: Float32Array): MlPrediction {
    const score = Math.max(0, ...this.zScores(x)) / this.threshold;
    return { scores: [score], confidence: anomalyConfidence(score) };
  }
}

/** Confidence of an anomaly score normalised so 1 is the threshold: distance from it, capped. */
export function anomalyConfidence(score: number): number {
  return Math.max(0, Math.min(1, Math.abs(score - 1)));
}

export function fitRobustZ(samples: readonly MlSample[], opts: RobustZOptions = {}): RobustZModel {
  const X = samples.map((s) => s.x);
  const d = featureDim(X);
  const minFrac = opts.minScaleFraction ?? 0.01;
  const minAbs = opts.minScale ?? 0;
  const center = new Float64Array(d);
  const scale = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    const col = X.map((x) => x[i]!);
    const med = median(col);
    const dev = col.map((v) => Math.abs(v - med));
    let s = MAD_TO_SD * median(dev);
    if (!(s > 0)) s = MEANAD_TO_SD * (dev.reduce((a, b) => a + b, 0) / dev.length);
    center[i] = med;
    scale[i] = Math.max(s, minFrac * Math.abs(med) + 1e-9, minAbs);
  }
  return new RobustZModel(center, scale, opts.threshold ?? 3.5);
}
