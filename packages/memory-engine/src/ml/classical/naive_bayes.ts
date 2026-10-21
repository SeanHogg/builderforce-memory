/**
 * ml/classical/naive_bayes.ts — Gaussian naive Bayes.
 *
 * Per class: a weighted prior, and an independent Gaussian per feature. Variance
 * is smoothed by a fraction of the largest feature variance (as scikit-learn's
 * `var_smoothing`), so a constant feature never divides by zero.
 */
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { classCount, featureDim, softmax, topProbability } from "./math.js";

export interface NaiveBayesOptions {
  classes?: number;
  /** Fraction of the largest feature variance added to every variance. Default 1e-9. */
  varSmoothing?: number;
}

export class NaiveBayesModel implements MlPredictor {
  constructor(
    readonly logPrior: Float64Array,
    readonly mean: Float64Array[],
    readonly variance: Float64Array[],
  ) {}

  /** Per-class joint log-likelihood log P(c) + Σ log N(x_i | μ, σ²). */
  jointLogLikelihood(x: ArrayLike<number>): number[] {
    return this.mean.map((mu, c) => {
      const v = this.variance[c]!;
      let ll = this.logPrior[c]!;
      for (let i = 0; i < mu.length; i++) {
        const d = (x[i] ?? 0) - mu[i]!;
        ll += -0.5 * Math.log(2 * Math.PI * v[i]!) - (d * d) / (2 * v[i]!);
      }
      return ll;
    });
  }

  predict(x: Float32Array): MlPrediction {
    const scores = Array.from(softmax(this.jointLogLikelihood(x)));
    return { scores, confidence: topProbability(scores) };
  }
}

export function fitNaiveBayes(samples: readonly MlSample[], opts: NaiveBayesOptions = {}): NaiveBayesModel {
  const d = featureDim(samples.map((s) => s.x));
  const k = opts.classes ?? classCount(samples.map((s) => s.y));
  const wSum = new Float64Array(k);
  const mean = Array.from({ length: k }, () => new Float64Array(d));
  const variance = Array.from({ length: k }, () => new Float64Array(d));

  for (const s of samples) {
    const w = s.weight ?? 1;
    wSum[s.y] = wSum[s.y]! + w;
    for (let i = 0; i < d; i++) mean[s.y]![i] = mean[s.y]![i]! + w * s.x[i]!;
  }
  for (let c = 0; c < k; c++) if (wSum[c]! > 0) for (let i = 0; i < d; i++) mean[c]![i] = mean[c]![i]! / wSum[c]!;
  for (const s of samples) {
    const w = s.weight ?? 1;
    for (let i = 0; i < d; i++) {
      const diff = s.x[i]! - mean[s.y]![i]!;
      variance[s.y]![i] = variance[s.y]![i]! + w * diff * diff;
    }
  }
  let maxVar = 0;
  for (let c = 0; c < k; c++) {
    for (let i = 0; i < d; i++) {
      variance[c]![i] = wSum[c]! > 0 ? variance[c]![i]! / wSum[c]! : 0;
      maxVar = Math.max(maxVar, variance[c]![i]!);
    }
  }
  const eps = (opts.varSmoothing ?? 1e-9) * Math.max(maxVar, 1e-12) + 1e-12;
  for (let c = 0; c < k; c++) for (let i = 0; i < d; i++) variance[c]![i] = variance[c]![i]! + eps;

  const total = wSum.reduce((a, b) => a + b, 0);
  // An unseen class keeps a vanishing (not -∞) prior so softmax stays defined.
  const logPrior = Float64Array.from(wSum, (w) => Math.log(Math.max(w, 1e-12) / total));
  return new NaiveBayesModel(logPrior, mean, variance);
}
