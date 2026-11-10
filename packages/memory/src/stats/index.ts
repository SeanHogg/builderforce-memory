/**
 * The stats module (PRD 41 Part D / D13): the ONE home for the predicted-measure
 * maths. Pure and deterministic — seeded PRNG, fixed iteration caps, quantized
 * outputs. It holds only what the memory-engine `ml/classical` catalog lacks and
 * imports (never re-implements) those primitives; EWMA, Ledoit–Wolf and the
 * classical models stay in `ml/classical` (Ledoit–Wolf is re-exported here).
 *
 * One deliberate exception: the BINARY logistic head (`logistic.ts` — `sigmoid`,
 * `logisticProbability`, `onlineLogisticStep`) is float64 and engine-free, not the
 * engine's softmax `LogisticModel` (Float32 weights, batch fits). Device code scores a
 * stored head and folds in one outcome at a time; batch fitting (gradient descent or
 * IRLS) stays in the engine.
 */

export { createRng, normalDraw } from "./prng.js";
export type { Rng } from "./prng.js";

export {
    betaMoments,
    betaPosterior,
    gammaQuantile,
    garwoodPoissonInterval,
    inverseRegularizedIncompleteBeta,
    RATE_CREDIBLE_MASS,
    RATE_PRIOR,
    rateInterval,
    regularizedIncompleteBeta,
    regularizedIncompleteGamma,
    wilsonInterval,
} from "./distributions.js";
export type { BetaPosterior } from "./distributions.js";

export { logisticProbability, onlineLogisticStep, sigmoid } from "./logistic.js";

export { fitBayesianLogistic, fitIsotonic, fitPlattScaling } from "./fitting.js";
export type { BayesianLogisticOptions, IsotonicFit } from "./fitting.js";

export {
    pairedBootstrapCi,
    signTestSprt,
    wilcoxonSignedRank,
} from "./tests.js";
export type { PairedStatistic, SprtResult, WilcoxonResult } from "./tests.js";

export { bayesianOnlineChangePoint, cusum } from "./timeSeries.js";
export type { ChangePointResult, CusumResult } from "./timeSeries.js";

export { ledoitWolfCovariance, psi } from "./multivariate.js";
export type { CovarianceResult } from "./multivariate.js";

export { entropyOfCounts, shannonEntropy } from "./information.js";

export { meanAndSd } from "./moments.js";
export type { MeanAndSd } from "./moments.js";

export {
    chapmanEstimate,
    gammaPoissonPosterior,
    goelOkumotoMLE,
    kaplanMeier,
    lincolnPetersenEstimate,
} from "./estimators.js";
export type { InsufficientData, KaplanMeierResult } from "./estimators.js";

export {
    brierScore,
    effortAwareRecall,
    expectedCalibrationError,
    ndcgAtK,
    popt,
    prAuc,
} from "./scores.js";

export { thompsonRank, thompsonSample } from "./policies.js";
export type { BetaArm } from "./policies.js";
