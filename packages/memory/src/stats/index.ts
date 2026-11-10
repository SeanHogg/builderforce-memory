/**
 * The stats module (PRD 41 Part D / D13): the ONE home for the predicted-measure
 * maths. Pure and deterministic — seeded PRNG, fixed iteration caps, quantized
 * outputs. It holds only what the memory-engine `ml/classical` catalog lacks and
 * imports (never re-implements) those primitives; EWMA and the classical models
 * stay in `ml/classical`.
 */

export { createRng, normalDraw } from "./prng.js";
export type { Rng } from "./prng.js";

export {
    betaPosterior,
    gammaQuantile,
    garwoodPoissonInterval,
    inverseRegularizedIncompleteBeta,
    RATE_CREDIBLE_MASS,
    RATE_PRIOR,
    rateInterval,
    regularizedIncompleteBeta,
    regularizedIncompleteGamma,
} from "./distributions.js";
export type { BetaPosterior } from "./distributions.js";

export { fitBayesianLogistic, fitIsotonic, fitPlattScaling } from "./fitting.js";
export type { BayesianLogisticOptions, IsotonicFit } from "./fitting.js";

export {
    pairedBootstrapCi,
    signTestSprt,
    wilcoxonSignedRank,
} from "./tests.js";
export type { SprtResult, WilcoxonResult } from "./tests.js";

export { bayesianOnlineChangePoint, cusum } from "./timeSeries.js";
export type { ChangePointResult, CusumResult } from "./timeSeries.js";

export { ledoitWolfCovariance, psi } from "./multivariate.js";
export type { CovarianceResult } from "./multivariate.js";

export { entropyOfCounts, shannonEntropy } from "./information.js";

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
