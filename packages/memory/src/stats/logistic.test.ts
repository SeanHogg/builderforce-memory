/**
 * The PRD 41 D13 consolidation primitives: the ONE stable sigmoid and logistic head
 * (prefetch's online calibration, the difficulty head, the pipeline router), the Wilson
 * interval (replay impact and issue clusters, the canvas proportion card) and the Beta
 * moments (the learned-routing Thompson draw).
 */

import { describe, expect, it } from "@jest/globals";
import {
    betaMoments,
    betaPosterior,
    logisticProbability,
    onlineLogisticStep,
    sigmoid,
    wilsonInterval,
} from "./index.js";

describe("sigmoid", () => {
    it("is ½ at 0, symmetric, and finite in both tails", () => {
        expect(sigmoid(0)).toBe(0.5);
        for (const z of [0.3, 2, 9]) expect(sigmoid(-z)).toBeCloseTo(1 - sigmoid(z), 12);
        expect(sigmoid(-800)).toBe(0);
        expect(sigmoid(800)).toBe(1);
        expect(Number.isNaN(sigmoid(-1e308))).toBe(false);
    });
});

describe("logistic head", () => {
    it("scores σ(b + w·x)", () => {
        expect(logisticProbability([6, 2, 2], -5, [1, 0, 0])).toBeCloseTo(sigmoid(1), 15);
        expect(logisticProbability([1, -1], 0.5, [2, 3])).toBeCloseTo(sigmoid(-0.5), 15);
    });

    it("one SGD step moves toward the label and leaves its input untouched", () => {
        const weights = [0, 0];
        const x = [1, 0.5];
        const before = logisticProbability(weights, 0, x);
        const up = onlineLogisticStep(weights, 0, x, 1, 0.1);
        expect(logisticProbability(up.weights, up.bias, x)).toBeGreaterThan(before);
        expect(up.weights).toEqual([0.05, 0.025]);
        expect(up.bias).toBeCloseTo(0.05, 15);
        const down = onlineLogisticStep(weights, 0, x, 0, 0.1);
        expect(logisticProbability(down.weights, down.bias, x)).toBeLessThan(before);
        expect(weights).toEqual([0, 0]);
    });
});

describe("wilsonInterval", () => {
    it("matches the textbook 95% interval", () => {
        const { low, high } = wilsonInterval(50, 1_000);
        expect(low).toBeCloseTo(0.0381, 4);
        expect(high).toBeCloseTo(0.0653, 4);
    });

    it("keeps a width at the boundary and stays in [0, 1]", () => {
        const none = wilsonInterval(0, 40);
        expect(none.low).toBeCloseTo(0, 12);
        expect(none.high).toBeGreaterThan(0.05);
        expect(wilsonInterval(10, 10).high).toBeCloseTo(1, 12);
    });

    it("no evidence (n ≤ 0 or NaN, or non-finite successes) is [0, 1]", () => {
        expect(wilsonInterval(0, 0)).toEqual({ low: 0, high: 1 });
        expect(wilsonInterval(3, Number.NaN)).toEqual({ low: 0, high: 1 });
        expect(wilsonInterval(Number.NaN, 10)).toEqual({ low: 0, high: 1 });
        expect(wilsonInterval(Number.POSITIVE_INFINITY, 10)).toEqual({ low: 0, high: 1 });
    });

    it("a z that is not a finite positive number throws", () => {
        for (const z of [0, -1.96, Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(() => wilsonInterval(3, 10, z)).toThrow(RangeError);
        }
    });

    it("a wider z gives a wider interval", () => {
        const a = wilsonInterval(30, 100);
        const b = wilsonInterval(30, 100, 2.576);
        expect(b.high - b.low).toBeGreaterThan(a.high - a.low);
    });
});

describe("betaMoments", () => {
    it("gives the Beta mean and variance, and the posterior carries them", () => {
        expect(betaMoments(2, 3)).toEqual({ mean: 0.4, variance: 0.04 });
        const post = betaPosterior(1, 1, 1, 3);
        expect(post.mean).toBe(betaMoments(2, 3).mean);
        expect(post.variance).toBe(betaMoments(2, 3).variance);
    });
});
