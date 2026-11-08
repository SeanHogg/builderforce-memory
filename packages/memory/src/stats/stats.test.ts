/**
 * Tests for the stats module (PRD 41 Part D / D13). The done-checks these cover:
 *   - Chapman estimate against textbook values; m < 3 style degenerate inputs give
 *     `insufficient-data` (no recaptures).
 *   - Rule precision shapes (Beta posterior quantiles) and calibration (ECE, Brier).
 *   - The SPRT stops early on a fixture with a 20% regression and runs to the
 *     maximum on a no-change fixture.
 *   - Determinism: the same seed gives byte-identical draws and schedules.
 */

import { describe, expect, it } from "@jest/globals";
import {
    bayesianOnlineChangePoint,
    betaPosterior,
    brierScore,
    chapmanEstimate,
    createRng,
    cusum,
    effortAwareRecall,
    entropyOfCounts,
    expectedCalibrationError,
    fitBayesianLogistic,
    fitIsotonic,
    fitPlattScaling,
    gammaPoissonPosterior,
    gammaQuantile,
    garwoodPoissonInterval,
    goelOkumotoMLE,
    kaplanMeier,
    ledoitWolfCovariance,
    lincolnPetersenEstimate,
    ndcgAtK,
    pairedBootstrapCi,
    popt,
    prAuc,
    psi,
    shannonEntropy,
    signTestSprt,
    thompsonRank,
    thompsonSample,
    wilcoxonSignedRank,
} from "./index.js";

describe("prng", () => {
    it("is deterministic under its seed", () => {
        const a = createRng(42);
        const b = createRng(42);
        const seqA = Array.from({ length: 8 }, () => a());
        const seqB = Array.from({ length: 8 }, () => b());
        expect(seqA).toEqual(seqB);
        expect(seqA.every((v) => v >= 0 && v < 1)).toBe(true);
    });

    it("differs across seeds", () => {
        const a = createRng(1);
        const b = createRng(2);
        expect(a()).not.toEqual(b());
    });
});

describe("distributions", () => {
    it("beta posterior mean and quantiles bracket sensibly", () => {
        const post = betaPosterior(1, 1, 30, 100); // observed 30/100
        expect(post.mean).toBeCloseTo(31 / 102, 5);
        const lo = post.quantile(0.025);
        const hi = post.quantile(0.975);
        expect(lo).toBeGreaterThan(0);
        expect(hi).toBeLessThan(1);
        expect(lo).toBeLessThan(post.mean);
        expect(hi).toBeGreaterThan(post.mean);
    });

    it("gamma quantile inverts the CDF", () => {
        // median of Gamma(2, 1) ≈ 1.678
        expect(gammaQuantile(0.5, 2, 1)).toBeCloseTo(1.678, 2);
        expect(gammaQuantile(0, 2)).toBe(0);
    });

    it("garwood interval contains the point estimate and handles zero", () => {
        const interval = garwoodPoissonInterval(10, 1);
        expect(interval.low).toBeLessThan(10);
        expect(interval.high).toBeGreaterThan(10);
        const zero = garwoodPoissonInterval(0, 1);
        expect(zero.low).toBe(0);
        expect(zero.rateLow).toBe(0);
    });
});

describe("fitting", () => {
    it("bayesian logistic recovers a monotone relationship", () => {
        const xs = [-3, -2, -1.5, -1, 0, 1, 1.5, 2, 3, 3.5];
        const ys = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1];
        const fit = fitBayesianLogistic(xs, ys);
        expect(fit.predict(-3)).toBeLessThan(fit.predict(0));
        expect(fit.predict(0)).toBeLessThan(fit.predict(3));
        expect(fit.slope).toBeGreaterThan(0);
    });

    it("platt scaling maps low scores low and high scores high", () => {
        const scores = [-2, -1, -0.5, 0.5, 1, 2];
        const labels = [0, 0, 0, 1, 1, 1];
        const platt = fitPlattScaling(scores, labels);
        expect(platt.predict(-2)).toBeLessThan(0.2);
        expect(platt.predict(2)).toBeGreaterThan(0.8);
    });

    it("isotonic regression is monotone non-decreasing", () => {
        const xs = [1, 2, 3, 4, 5];
        const ys = [0.1, 0.4, 0.2, 0.9, 0.8]; // a dip and a peak to pool
        const iso = fitIsotonic(xs, ys);
        let prev = -Infinity;
        for (const x of xs) {
            const v = iso.predict(x);
            expect(v).toBeGreaterThanOrEqual(prev - 1e-12);
            prev = v;
        }
    });
});

describe("statistical tests", () => {
    it("wilcoxon signed-rank detects a consistent shift", () => {
        const before = Array.from({ length: 12 }, (_, i) => 100 + i * 7);
        const after = before.map((v) => v + 5);
        const result = wilcoxonSignedRank(before, after);
        expect(result.n).toBe(12);
        expect(result.pValue).toBeLessThan(0.01);
        expect(result.hodgesLehmann).toBeCloseTo(5, 5);
    });

    it("wilcoxon on a zero shift does not reject", () => {
        const values = Array.from({ length: 10 }, (_, i) => 50 + (i % 3));
        const result = wilcoxonSignedRank(values, [...values]);
        expect(result.n).toBe(0);
        expect(result.pValue).toBe(1);
    });

    it("paired bootstrap CI is deterministic under its seed and contains the estimate", () => {
        const before = [10, 12, 11, 13, 15, 9, 14, 10];
        const after = [11, 13, 12, 14, 16, 10, 15, 11];
        const a = pairedBootstrapCi(before, after, { seed: 7 });
        const b = pairedBootstrapCi(before, after, { seed: 7 });
        expect(a).toEqual(b);
        expect(a.low).toBeLessThanOrEqual(a.estimate);
        expect(a.high).toBeGreaterThanOrEqual(a.estimate);
    });

    it("SPRT stops early on a clear regression and continues on no change", () => {
        const regression = Array.from({ length: 40 }, () => [1, 1.25] as const); // 20%+ slower
        const stop = signTestSprt(regression, { p0: 0.5, p1: 0.7, alpha: 0.05, beta: 0.1 });
        expect(stop.decision).toBe("reject-null");
        expect(stop.observations).toBeLessThan(40);

        // No-change fixture: 3 improvements per 2 regressions keeps the cumulative
        // LLR wandering inside the boundaries, so the test runs to its maximum.
        const none: ReadonlyArray<readonly [number, number]> = Array.from(
            { length: 12 },
            () => [[1, 1.05], [1, 1.02], [1, 0.98], [1, 1.01], [1, 0.97]] as const,
        ).flat();
        const run = signTestSprt(none, { p0: 0.5, p1: 0.7, alpha: 0.05, beta: 0.1 });
        expect(run.decision).toBe("continue");
        expect(run.observations).toBe(60);
    });
});

describe("time series", () => {
    it("cusum alarms after a sustained shift", () => {
        const values = [...Array.from({ length: 30 }, () => 0), ...Array.from({ length: 30 }, () => 2.5)];
        const result = cusum(values, 0, { sigma: 1, threshold: 5 });
        expect(result.alarmIndex).toBeGreaterThanOrEqual(30);
        expect(result.alarmIndex).toBeLessThan(45);
    });

    it("bayesian change point flags the shift index", () => {
        const values = [...Array.from({ length: 40 }, () => 0), ...Array.from({ length: 40 }, () => 3)];
        const result = bayesianOnlineChangePoint(values, { hazard: 0.05 });
        expect(result.currentRunLength).toBeGreaterThanOrEqual(0);
        // The change-point probability near the shift should exceed the background.
        const nearShift = result.changePointProbabilities[41]!;
        const background = result.changePointProbabilities[10]!;
        expect(nearShift).toBeGreaterThan(background);
    });
});

describe("multivariate", () => {
    it("ledoit-wolf covariance recovers the diagonal of a near-identity", () => {
        const rng = createRng(11);
        const data = Array.from({ length: 200 }, () => [rng() < 0.5 ? 0 : 1, rng() < 0.5 ? 0 : 1]);
        const { covariance, shrinkage } = ledoitWolfCovariance(data);
        expect(shrinkage).toBeGreaterThanOrEqual(0);
        expect(shrinkage).toBeLessThanOrEqual(1);
        expect(covariance[0]![0]!).toBeGreaterThan(0);
    });

    it("psi is zero for identical distributions and positive for drift", () => {
        const a = Array.from({ length: 500 }, (_, i) => i / 500);
        expect(psi(a, a, 10)).toBeCloseTo(0, 6);
        const drifted = a.map((v) => v * 0.5 + 0.5);
        expect(psi(a, drifted, 10)).toBeGreaterThan(0.2);
    });
});

describe("information", () => {
    it("entropy is maximal for uniform and zero for a point mass", () => {
        expect(shannonEntropy([1])).toBeCloseTo(0, 10);
        expect(shannonEntropy([0.5, 0.5])).toBeCloseTo(Math.log(2), 10);
        expect(entropyOfCounts([2, 2, 2, 2])).toBeCloseTo(Math.log(4), 10);
    });
});

describe("estimators", () => {
    it("chapman estimate matches the textbook value; no recaptures is insufficient", () => {
        // Textbook: m = 20, c = 15, r = 5 → n̂ = (21·16)/6 − 1 = 55
        const estimate = chapmanEstimate(20, 15, 5);
        expect(estimate).toEqual({ estimate: 55 });
        expect(chapmanEstimate(20, 15, 0)).toEqual({ status: "insufficient-data" });
        expect(lincolnPetersenEstimate(20, 15, 5)).toEqual({ estimate: 60 });
    });

    it("kaplan-meier drops at event times and honors censoring", () => {
        const km = kaplanMeier([
            { time: 1, event: 1 },
            { time: 2, event: 0 },
            { time: 3, event: 1 },
            { time: 4, event: 0 },
        ]);
        expect(km.curve[0]!.survival).toBeCloseTo(0.75, 6);
        expect(km.survival).toBeCloseTo(0.75 * (1 - 1 / 2), 6);
        expect(km.variance).toBeGreaterThanOrEqual(0);
    });

    it("goel-okumoto MLE needs at least two times", () => {
        expect(goelOkumotoMLE([1])).toEqual({ status: "insufficient-data" });
        const fit = goelOkumotoMLE([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        if (fit.status === "insufficient-data") throw new Error("expected a fit");
        expect(fit.a).toBeGreaterThan(0);
        expect(fit.b).toBeGreaterThan(0);
    });

    it("gamma-poisson posterior mean sits between prior and data at small n", () => {
        const prior = gammaPoissonPosterior(2, 4, 0, 0); // Gamma(2, 4), mean 0.5
        const updated = gammaPoissonPosterior(2, 4, 2, 1); // Gamma(4, 5), mean 0.8
        expect(prior.mean).toBeCloseTo(0.5, 6);
        expect(updated.mean).toBeGreaterThan(prior.mean);
        expect(updated.variance).toBeGreaterThan(0);
    });
});

describe("scores", () => {
    it("brier and ECE penalize miscalibration", () => {
        expect(brierScore([1, 1, 1, 1], [1, 1, 1, 1])).toBeCloseTo(0, 10);
        expect(brierScore([1, 0], [0, 1])).toBeCloseTo(1, 10);
        expect(expectedCalibrationError([1, 1, 1, 1], [1, 1, 1, 1])).toBeCloseTo(0, 10);
        expect(expectedCalibrationError([0.9, 0.9, 0.1, 0.1], [0, 0, 1, 1])).toBeGreaterThan(0.7);
    });

    it("pr-auc is 1 for a perfect ranking and ndcg penalizes bad order", () => {
        expect(prAuc([0.9, 0.8, 0.1, 0.05], [1, 1, 0, 0])).toBeCloseTo(1, 6);
        expect(ndcgAtK([3, 2, 1], 3)).toBeCloseTo(1, 6);
        expect(ndcgAtK([1, 2, 3], 3)).toBeLessThan(1);
    });

    it("popt and effort-aware recall reward good ordering", () => {
        const efforts = [0.1, 0.2, 0.3, 0.4];
        const goodOrder = [0.9, 0.8, 0.2, 0.1];
        expect(popt(efforts, goodOrder)).toBeGreaterThan(0.5);
        expect(
            effortAwareRecall([true, true, false, false], efforts, 0.5),
        ).toBeCloseTo(1, 6);
    });
});

describe("policies", () => {
    it("thompson sampling favors the better arm and is deterministic under seed", () => {
        const arms = [
            { alpha: 30, beta: 10 }, // ~75%
            { alpha: 10, beta: 30 }, // ~25%
        ];
        const picksA = Array.from({ length: 200 }, () => thompsonSample(arms, createRng(99)));
        const picksB = Array.from({ length: 200 }, () => thompsonSample(arms, createRng(99)));
        expect(picksA).toEqual(picksB);
        const betterPicks = picksA.filter((i) => i === 0).length;
        expect(betterPicks).toBeGreaterThan(150);
    });

    it("thompson rank is a permutation of all arms", () => {
        const arms = [
            { alpha: 5, beta: 5 },
            { alpha: 20, beta: 5 },
            { alpha: 5, beta: 20 },
        ];
        const rank = thompsonRank(arms, createRng(3));
        expect([...rank].sort()).toEqual([0, 1, 2]);
    });
});
