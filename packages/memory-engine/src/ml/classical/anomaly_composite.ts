/**
 * ml/classical/anomaly_composite.ts — robust z and EWMA as ONE anomaly engine.
 *
 * Two detectors, one answer: robust z (median/MAD over the whole history — "far
 * from normal") and EWMA (the recent trend — "drifting away from lately"). The
 * score is the larger of the two normalised scores, so either one can flag. It is
 * one engine because it is one fitted thing: a task that lists it gets back exactly
 * what it stores under `local:classical:anomaly-composite`.
 */
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { EwmaModel, fitEwma, type EwmaOptions } from "./ewma.js";
import { anomalyConfidence, fitRobustZ, RobustZModel, type RobustZOptions } from "./robust_z.js";

export interface AnomalyCompositeOptions {
  robustZ?: RobustZOptions;
  ewma?: EwmaOptions;
}

export class AnomalyCompositeModel implements MlPredictor {
  constructor(
    readonly population: RobustZModel,
    readonly trend: EwmaModel,
  ) {}

  /** Per feature, per detector: z / threshold (≥ 1 = out of range). */
  featureScores(x: ArrayLike<number>): { population: number; trend: number }[] {
    const pz = this.population.zScores(x);
    const tz = this.trend.zScores(x);
    return pz.map((z, i) => ({ population: z / this.population.threshold, trend: tz[i]! / this.trend.threshold }));
  }

  /** Fold a finished, non-anomalous observation into the trend. */
  observe(x: ArrayLike<number>): void {
    this.trend.update(x);
  }

  predict(x: Float32Array): MlPrediction {
    const score = Math.max(this.population.predict(x).scores[0]!, this.trend.predict(x).scores[0]!);
    return { scores: [score], confidence: anomalyConfidence(score) };
  }
}

/** Fit both detectors on a chronological history (feature vectors, oldest first). */
export function fitAnomalyComposite(samples: readonly MlSample[], opts: AnomalyCompositeOptions = {}): AnomalyCompositeModel {
  return new AnomalyCompositeModel(fitRobustZ(samples, opts.robustZ), fitEwma(samples, opts.ewma));
}
