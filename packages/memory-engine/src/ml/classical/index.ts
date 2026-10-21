/**
 * ml/classical — pure-TypeScript algorithms, one file each, and the ONE dispatch
 * from a {@link ClassicalAlgorithm} name to its fit.
 */
import type { ClassicalAlgorithm, MlPredictor, MlSample } from "../types.js";
import { fitKnn, type KnnOptions } from "./knn.js";
import { fitNaiveBayes, type NaiveBayesOptions } from "./naive_bayes.js";
import { fitLogistic, type LogisticOptions } from "./logistic.js";
import { fitLinear, type LinearOptions } from "./linear.js";
import { fitKMeans, type KMeansOptions } from "./kmeans.js";
import { fitRobustZ, type RobustZOptions } from "./robust_z.js";
import { fitEwma, type EwmaOptions } from "./ewma.js";
import { fitIsolationForest, type IsolationForestOptions } from "./isolation_forest.js";
import { fitGbm, type GbmOptions } from "./gbm.js";

export { KnnModel, fitKnn } from "./knn.js";
export type { KnnOptions, KnnMetric, Neighbour } from "./knn.js";
export { NaiveBayesModel, fitNaiveBayes } from "./naive_bayes.js";
export type { NaiveBayesOptions } from "./naive_bayes.js";
export { LogisticModel, fitLogistic } from "./logistic.js";
export type { LogisticOptions } from "./logistic.js";
export { LinearModel, fitLinear, solveLinearSystem } from "./linear.js";
export type { LinearOptions } from "./linear.js";
export { KMeansModel, fitKMeans } from "./kmeans.js";
export type { KMeansOptions } from "./kmeans.js";
export { RobustZModel, fitRobustZ, anomalyConfidence } from "./robust_z.js";
export type { RobustZOptions } from "./robust_z.js";
export { EwmaModel, fitEwma } from "./ewma.js";
export type { EwmaOptions } from "./ewma.js";
export { IsolationForestModel, fitIsolationForest } from "./isolation_forest.js";
export type { IsolationForestOptions } from "./isolation_forest.js";
export { GbmModel, fitGbm } from "./gbm.js";
export type { GbmOptions, GbmObjective } from "./gbm.js";
export { cosine, dot } from "./math.js";

/** Per-algorithm options, keyed by algorithm name. */
export interface ClassicalOptions {
  knn?: KnnOptions;
  "naive-bayes"?: NaiveBayesOptions;
  logistic?: LogisticOptions;
  linear?: LinearOptions;
  kmeans?: KMeansOptions;
  "robust-z"?: RobustZOptions;
  ewma?: EwmaOptions;
  "isolation-forest"?: IsolationForestOptions;
  "gbm-small"?: GbmOptions;
}

/** Fit `algorithm` on `samples`. Unsupervised algorithms ignore `y`. */
export function fitClassical(
  algorithm: ClassicalAlgorithm,
  samples: readonly MlSample[],
  opts: ClassicalOptions = {},
): MlPredictor {
  if (samples.length === 0) throw new Error(`fitClassical(${algorithm}): no samples`);
  switch (algorithm) {
    case "knn":
      return fitKnn(samples, opts.knn);
    case "naive-bayes":
      return fitNaiveBayes(samples, opts["naive-bayes"]);
    case "logistic":
      return fitLogistic(samples, opts.logistic);
    case "linear":
      return fitLinear(samples, opts.linear);
    case "kmeans":
      return fitKMeans(samples, opts.kmeans);
    case "robust-z":
      return fitRobustZ(samples, opts["robust-z"]);
    case "ewma":
      return fitEwma(samples, opts.ewma);
    case "isolation-forest":
      return fitIsolationForest(samples, opts["isolation-forest"]);
    case "gbm-small":
      return fitGbm(samples, opts["gbm-small"]);
  }
}
