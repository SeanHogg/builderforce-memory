/**
 * ml/classical/ewma.ts — exponentially-weighted moving average + variance, for
 * drift anomalies in an ordered series (a run's cost creeping up over the week).
 *
 * Per feature, over the history IN ORDER: diff = x − μ; μ += α·diff;
 * σ² = (1 − α)(σ² + α·diff²). A new point scores z = |x − μ| / σ against the
 * state BEFORE it is folded in. Prediction: `scores[0] = max_i z_i / threshold`.
 * σ is floored like robust z's scale: a fraction of |μ|, and `minScale` in absolute
 * units — σ² starts at 0, so a constant history would otherwise flag any change.
 */
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { featureDim } from "./math.js";
import { anomalyConfidence, minScaleAt, type MinScale } from "./robust_z.js";

export interface EwmaOptions {
  /** Smoothing factor in (0, 1]. Default 0.3. */
  alpha?: number;
  /** z at which a feature counts as anomalous. Default 3. */
  threshold?: number;
  /** Smallest σ per feature, as a fraction of |μ| (plus 1e-9). Default 0.01. */
  minScaleFraction?: number;
  /** Smallest σ in absolute feature units, one or per feature. Default 0 (see `RobustZOptions.minScale`). */
  minScale?: MinScale;
}

export class EwmaModel implements MlPredictor {
  constructor(
    readonly mean: Float64Array,
    readonly variance: Float64Array,
    readonly alpha: number,
    readonly threshold: number,
    readonly minScaleFraction: number,
    readonly minScale: MinScale = 0,
  ) {
    if (!(alpha > 0 && alpha <= 1)) throw new Error(`EwmaModel: alpha must be in (0, 1] (got ${alpha})`);
  }

  zScores(x: ArrayLike<number>): number[] {
    return Array.from(this.mean, (m, i) => {
      const sd = Math.max(Math.sqrt(this.variance[i]!), this.minScaleFraction * Math.abs(m) + 1e-9, minScaleAt(this.minScale, i));
      return Math.abs((x[i] ?? 0) - m) / sd;
    });
  }

  /** Fold one observation into the running state. */
  update(x: ArrayLike<number>): void {
    for (let i = 0; i < this.mean.length; i++) {
      const diff = (x[i] ?? 0) - this.mean[i]!;
      const incr = this.alpha * diff;
      this.mean[i] = this.mean[i]! + incr;
      this.variance[i] = (1 - this.alpha) * (this.variance[i]! + diff * incr);
    }
  }

  predict(x: Float32Array): MlPrediction {
    const score = Math.max(0, ...this.zScores(x)) / this.threshold;
    return { scores: [score], confidence: anomalyConfidence(score) };
  }
}

/** Fit over `samples` in the given (chronological) order. */
export function fitEwma(samples: readonly MlSample[], opts: EwmaOptions = {}): EwmaModel {
  const d = featureDim(samples.map((s) => s.x));
  const model = new EwmaModel(
    Float64Array.from(samples[0]!.x),
    new Float64Array(d),
    opts.alpha ?? 0.3,
    opts.threshold ?? 3,
    opts.minScaleFraction ?? 0.01,
    opts.minScale ?? 0,
  );
  for (let i = 1; i < samples.length; i++) model.update(samples[i]!.x);
  return model;
}
