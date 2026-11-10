/**
 * ml/classical/anomaly_composite.ts — robust z, EWMA and (opt-in) Mahalanobis as ONE
 * anomaly engine.
 *
 * Up to three detectors, one answer: robust z (median/MAD over the whole history —
 * "far from normal" per feature), EWMA (the recent trend — "drifting away from
 * lately") and, when `mahalanobis` is set and the history is long enough, the
 * Mahalanobis distance from the history's mean (PRD 41 §D2.5) — "an unusual
 * COMBINATION of features", which no per-feature detector sees. The score is the
 * largest of the normalised scores, so any one can flag. It is one engine because it
 * is one fitted thing: a task that lists it gets back exactly what it stores under
 * `local:classical:anomaly-composite`.
 *
 * The Mahalanobis term:
 *   - features are STANDARDIZED (z-scores over the history) first, so the covariance is
 *     a correlation matrix and a small-scale feature is not drowned by a large-scale one;
 *   - the covariance is the Ledoit–Wolf shrinkage estimate (`covariance.ts`), inverted by
 *     one Cholesky factorization;
 *   - the threshold is the history's own EMPIRICAL quantile (default 99th) of
 *     leave-one-out distances — no χ² assumption, which skewed count features break;
 *   - below `minSamples` past rows (default 30, never below d + 2) there is no term.
 *   Count features should be log-transformed by the task before they reach here.
 */
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { choleskyInverse, ledoitWolfCovariance } from "./covariance.js";
import { EwmaModel, fitEwma, type EwmaOptions } from "./ewma.js";
import { featureDim, quantile } from "./math.js";
import { anomalyConfidence, fitRobustZ, RobustZModel, type RobustZOptions } from "./robust_z.js";

export interface MahalanobisOptions {
  /** Past rows needed before the term exists. Default 30; never below d + 2. */
  minSamples?: number;
  /** Empirical quantile of the leave-one-out distances that scores 1. Default 0.99. */
  quantile?: number;
  /** Newest rows whose leave-one-out distance is computed (each refits on all others). Default 200. */
  looRows?: number;
  /** Absolute variance floor added to the standardized covariance's diagonal. Default 1e-9. */
  minVariance?: number;
  /** An explicit distance threshold instead of the empirical one. */
  threshold?: number;
}

export interface AnomalyCompositeOptions {
  robustZ?: RobustZOptions;
  ewma?: EwmaOptions;
  /** Present = the Mahalanobis term is fitted (once the history is long enough). Absent = robust z and EWMA only. */
  mahalanobis?: MahalanobisOptions;
}

export const MAHALANOBIS_MIN_SAMPLES = 30;
const DEFAULT_QUANTILE = 0.99;
const DEFAULT_LOO_ROWS = 200;
const DEFAULT_MIN_VARIANCE = 1e-9;
/** A threshold of 0 (an all-identical history) would score every deviation infinite. */
const MIN_THRESHOLD = 1e-9;
/** A column whose standard deviation is below this (relative to its mean) is constant. */
const CONSTANT_COLUMN_EPS = 1e-12;

/** The standardized-space fit: column means and scales, and the precision of the standardized rows. */
interface StandardizedFit {
  mean: Float64Array;
  scale: Float64Array;
  precision: Float64Array;
}

function fitStandardized(rows: readonly ArrayLike<number>[], d: number, minVariance: number): StandardizedFit {
  const n = rows.length;
  const mean = new Float64Array(d);
  for (const x of rows) for (let j = 0; j < d; j++) mean[j] = mean[j]! + x[j]!;
  for (let j = 0; j < d; j++) mean[j] = mean[j]! / n;
  const scale = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    let ss = 0;
    for (const x of rows) ss += (x[j]! - mean[j]!) ** 2;
    const sd = Math.sqrt(ss / n);
    // Rounding noise on a constant column is not a scale: such a column standardizes in its own units.
    scale[j] = sd > CONSTANT_COLUMN_EPS * Math.max(1, Math.abs(mean[j]!)) ? sd : 1;
  }
  const z = rows.map((x) => Array.from({ length: d }, (_, j) => (x[j]! - mean[j]!) / scale[j]!));
  const { covariance } = ledoitWolfCovariance(z);
  const flat = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) flat[i * d + j] = covariance[i]![j]! + (i === j ? minVariance : 0);
  }
  const precision = choleskyInverse(flat, d);
  if (!precision) throw new Error("fitMahalanobis: covariance is not positive definite");
  return { mean, scale, precision };
}

function distanceOf(fit: StandardizedFit, x: ArrayLike<number>): number {
  const d = fit.mean.length;
  const z = new Float64Array(d);
  for (let j = 0; j < d; j++) z[j] = ((x[j] ?? 0) - fit.mean[j]!) / fit.scale[j]!;
  let q = 0;
  for (let i = 0; i < d; i++) {
    let row = 0;
    for (let j = 0; j < d; j++) row += fit.precision[i * d + j]! * z[j]!;
    q += z[i]! * row;
  }
  return Math.sqrt(Math.max(0, q));
}

/** Mahalanobis distance (in standardized space) from a fitted history, scored against its own threshold. */
export class MahalanobisDetector {
  constructor(
    private readonly fit: StandardizedFit,
    readonly threshold: number,
  ) {}

  /** √(zᵀ Σ̂⁻¹ z), z the standardized deviation from the history's mean. */
  distance(x: ArrayLike<number>): number {
    return distanceOf(this.fit, x);
  }

  /** distance / threshold (≥ 1 = out of range). */
  score(x: ArrayLike<number>): number {
    return this.distance(x) / this.threshold;
  }
}

/** The rows needed before the term exists: `minSamples` (default 30), never below d + 2. */
export function mahalanobisMinSamples(d: number, opts: MahalanobisOptions = {}): number {
  return Math.max(opts.minSamples ?? MAHALANOBIS_MIN_SAMPLES, d + 2);
}

/**
 * The empirical leave-one-out threshold: for each of the newest `looRows` rows, its
 * distance under a fit on ALL OTHER rows; the `q`-quantile of those distances. A row is
 * never scored against a fit that saw it, so the threshold is what a NEW run of this
 * history's kind scores.
 */
export function leaveOneOutMahalanobisThreshold(
  rows: readonly ArrayLike<number>[],
  opts: MahalanobisOptions = {},
): number {
  const d = featureDim(rows);
  const minVariance = opts.minVariance ?? DEFAULT_MIN_VARIANCE;
  const looRows = Math.min(rows.length, Math.max(1, opts.looRows ?? DEFAULT_LOO_ROWS));
  const distances: number[] = [];
  for (let i = rows.length - looRows; i < rows.length; i++) {
    const others = rows.filter((_, r) => r !== i);
    distances.push(distanceOf(fitStandardized(others, d, minVariance), rows[i]!));
  }
  return Math.max(MIN_THRESHOLD, quantile(distances, opts.quantile ?? DEFAULT_QUANTILE));
}

/**
 * Fit the Mahalanobis term on a chronological history, or null below the minimum
 * history ({@link mahalanobisMinSamples}): a handful of runs has no covariance worth
 * scoring against, and a term fitted on one would flag any deviation at all.
 */
export function fitMahalanobis(samples: readonly MlSample[], opts: MahalanobisOptions = {}): MahalanobisDetector | null {
  const rows = samples.map((s) => s.x);
  const d = featureDim(rows);
  if (rows.length < mahalanobisMinSamples(d, opts)) return null;
  const fit = fitStandardized(rows, d, opts.minVariance ?? DEFAULT_MIN_VARIANCE);
  const threshold = opts.threshold ?? leaveOneOutMahalanobisThreshold(rows, opts);
  return new MahalanobisDetector(fit, threshold);
}

export class AnomalyCompositeModel implements MlPredictor {
  constructor(
    readonly population: RobustZModel,
    readonly trend: EwmaModel,
    readonly mahalanobis: MahalanobisDetector | null = null,
  ) {}

  /** Per feature, per detector: z / threshold (≥ 1 = out of range). */
  featureScores(x: ArrayLike<number>): { population: number; trend: number }[] {
    const pz = this.population.zScores(x);
    const tz = this.trend.zScores(x);
    return pz.map((z, i) => ({ population: z / this.population.threshold, trend: tz[i]! / this.trend.threshold }));
  }

  /** The Mahalanobis term (distance / threshold), or null when it was not fitted. */
  mahalanobisScore(x: ArrayLike<number>): number | null {
    return this.mahalanobis ? this.mahalanobis.score(x) : null;
  }

  /** Fold a finished, non-anomalous observation into the trend. */
  observe(x: ArrayLike<number>): void {
    this.trend.update(x);
  }

  predict(x: Float32Array): MlPrediction {
    let score = Math.max(this.population.predict(x).scores[0]!, this.trend.predict(x).scores[0]!);
    if (this.mahalanobis) score = Math.max(score, this.mahalanobis.score(x));
    return { scores: [score], confidence: anomalyConfidence(score) };
  }
}

/** Fit the detectors on a chronological history (feature vectors, oldest first). */
export function fitAnomalyComposite(samples: readonly MlSample[], opts: AnomalyCompositeOptions = {}): AnomalyCompositeModel {
  return new AnomalyCompositeModel(
    fitRobustZ(samples, opts.robustZ),
    fitEwma(samples, opts.ewma),
    opts.mahalanobis ? fitMahalanobis(samples, opts.mahalanobis) : null,
  );
}
