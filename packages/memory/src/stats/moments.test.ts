/**
 * The PRD 41 D13 mean / sample sd primitive (`meanAndSd`): the one mean and standard
 * deviation for predicted measures (Builderforce run-delta's naturalness model).
 */

import { describe, expect, it } from "@jest/globals";
import { meanAndSd } from "./index.js";

describe("meanAndSd", () => {
    it("gives the textbook mean and sample sd", () => {
        // 2,4,4,4,5,5,7,9: mean 5, Σ(x−5)² = 32, sample variance 32/7.
        const r = meanAndSd([2, 4, 4, 4, 5, 5, 7, 9]);
        expect(r.n).toBe(8);
        expect(r.nonFinite).toBe(0);
        expect(r.mean).toBe(5);
        expect(r.sd).toBeCloseTo(Math.sqrt(32 / 7), 14);
        const s = meanAndSd([1, 2, 3, 4]);
        expect(s.mean).toBe(2.5);
        expect(s.sd).toBeCloseTo(Math.sqrt(5 / 3), 14);
    });

    it("survives catastrophic cancellation (large common offset)", () => {
        const r = meanAndSd([1e9 + 4, 1e9 + 7, 1e9 + 13, 1e9 + 16]);
        expect(r.n).toBe(4);
        expect(r.mean).toBe(1e9 + 10);
        expect(r.sd * r.sd).toBeCloseTo(30, 9);
    });

    it("n = 0: mean NaN, sd 0", () => {
        expect(meanAndSd([])).toEqual({ n: 0, mean: Number.NaN, sd: 0, nonFinite: 0 });
    });

    it("n = 1: mean is the value, sd 0", () => {
        expect(meanAndSd([3.25])).toEqual({ n: 1, mean: 3.25, sd: 0, nonFinite: 0 });
    });

    it("all-equal input has sd exactly 0", () => {
        const r = meanAndSd([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
        expect(r.n).toBe(7);
        expect(r.mean).toBeCloseTo(0.1, 15);
        expect(r.sd).toBe(0);
    });

    it("skips NaN and ±Infinity, counting them in nonFinite", () => {
        const r = meanAndSd([1, Number.NaN, 2, Number.POSITIVE_INFINITY, 3, Number.NEGATIVE_INFINITY, 4]);
        expect(r.n).toBe(4);
        expect(r.nonFinite).toBe(3);
        expect(r.mean).toBe(2.5);
        expect(r.sd).toBeCloseTo(Math.sqrt(5 / 3), 14);
        expect(meanAndSd([Number.NaN, Number.NaN])).toEqual({ n: 0, mean: Number.NaN, sd: 0, nonFinite: 2 });
    });

    it("is deterministic and does not mutate its input", () => {
        const input = Object.freeze([5, 1, 9, 3, 7]);
        expect(meanAndSd(input)).toEqual(meanAndSd(input));
        expect(input).toEqual([5, 1, 9, 3, 7]);
    });
});
