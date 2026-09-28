/**
 * tests/eval-baseline.test.ts
 * Gating a candidate relative to a baseline report on the same eval.
 */

import { baselineRatio, meetsBaselineRatio, sameEvalProblem, evaluateGate } from '../src/eval/index.js';
import type { EvalReport } from '../src/eval/index.js';

function report(meanScore: number, cases = 3, dataset = 'coding-v1'): EvalReport {
    return {
        dataset,
        cases: Array.from({ length: cases }, () => ({}) as EvalReport['cases'][number]),
        passRate: 1,
        meanScore,
        graderScores: {},
        errorRate: 0,
        meanCostUsd: 0,
        totalCostUsd: 0,
        p50LatencyMs: 0,
        p95LatencyMs: 0,
        durationMs: 0,
        gate: { passed: true, failures: [], vacuous: true },
    };
}

describe('baseline ratio rules', () => {
    it('compares only a usable pair', () => {
        expect(baselineRatio(0.81, 0.9)).toBeCloseTo(0.9, 12);
        expect(baselineRatio(0.5, 0)).toBeNull();
        expect(baselineRatio(-0.1, 0.9)).toBeNull();
        expect(baselineRatio(Number.NaN, 0.9)).toBeNull();
    });

    it('meets an exact bar despite float error, and never on a null ratio', () => {
        expect(meetsBaselineRatio(0.81 / 0.9, 0.9)).toBe(true);
        expect(meetsBaselineRatio(0.8, 0.9)).toBe(false);
        expect(meetsBaselineRatio(null, 0)).toBe(false);
    });

    it('names why two reports are not the same eval', () => {
        expect(sameEvalProblem({ dataset: 'a', meanScore: 1, cases: 2 }, { dataset: 'a', meanScore: 1, cases: [1, 2] })).toBeNull();
        expect(sameEvalProblem({ dataset: 'a', meanScore: 1, cases: 2 }, { dataset: 'b', meanScore: 1, cases: 2 })).toMatch(/different evals/);
        expect(sameEvalProblem({ dataset: 'a', meanScore: 1, cases: 2 }, { dataset: 'a', meanScore: 1, cases: 3 })).toMatch(/case counts/);
    });
});

describe('evaluateGate minBaselineRatio', () => {
    it('passes at the bar and fails below it', () => {
        expect(evaluateGate(report(0.81), { minBaselineRatio: 0.9 }, report(0.9)).passed).toBe(true);
        const low = evaluateGate(report(0.5), { minBaselineRatio: 0.9 }, report(0.9));
        expect(low.passed).toBe(false);
        expect(low.failures[0]).toMatch(/of the baseline/);
    });

    it('fails without a baseline, or with one from another eval', () => {
        expect(evaluateGate(report(0.9), { minBaselineRatio: 0.9 }).failures[0]).toMatch(/needs a baseline/);
        expect(evaluateGate(report(0.9), { minBaselineRatio: 0.9 }, report(0.9, 3, 'other')).failures[0]).toMatch(/different evals/);
    });
});
