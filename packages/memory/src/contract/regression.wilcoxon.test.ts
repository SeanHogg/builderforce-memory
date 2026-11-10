/**
 * src/contract/regression.wilcoxon.test.ts — the stats module's Wilcoxon signed-rank, paired Hodges–Lehmann and paired
 * bootstrap (shipped wrong in 2026.10.72):
 *   - the exact null CDF was built ONCE for n = 60 and indexed with W+ for every n, so p ≈ 0 for any n ≤ 60 (a null
 *     10-pair series gave 2.2e-15); the n > 60 branch used exp(−z²/2) instead of a normal tail;
 *   - the Hodges–Lehmann estimate dropped zero differences, biasing it away from 0;
 *   - the paired bootstrap could only resample the MEAN, not the Hodges–Lehmann estimate PRD 41 C6.2 reports.
 */

import { describe, expect, it } from '@jest/globals';
import { pairedBootstrapCi, wilcoxonSignedRank } from '../stats/index.js';
import { exactSignedRankPValue, normalSignedRankPValue } from '../stats/tests.js';
import { createRng } from '../stats/prng.js';

/** The textbook paired Hodges–Lehmann: materialise every Walsh average, sort, take the median. */
function bruteHodgesLehmann(d: readonly number[]): number {
    const w: number[] = [];
    for (let p = 0; p < d.length; p++) for (let q = p; q < d.length; q++) w.push((d[p]! + d[q]!) / 2);
    w.sort((a, b) => a - b);
    const m = w.length;
    return m % 2 === 1 ? w[(m - 1) / 2]! : (w[m / 2 - 1]! + w[m / 2]!) / 2;
}

/** Paired data with distinct |d| = 1..n and W+ = w exactly (greedy subset of 1..n summing to w). */
function pairsWithWPlus(n: number, w: number): { before: number[]; after: number[] } {
    let remaining = w;
    const positive = new Set<number>();
    for (let r = n; r >= 1; r--) if (r <= remaining) { positive.add(r); remaining -= r; }
    if (remaining !== 0) throw new Error(`no subset of 1..${n} sums to ${w}`);
    const before = Array.from({ length: n }, () => 100);
    const after = before.map((b, i) => b + (positive.has(i + 1) ? i + 1 : -(i + 1)));
    return { before, after };
}

/** Two-sided α = 0.05 critical values of the signed-rank statistic (min(W+, W−) ≤ T rejects), standard tables. */
const CRITICAL_05: Record<number, number> = {
    6: 0, 7: 2, 8: 3, 9: 5, 10: 8, 11: 10, 12: 13, 13: 17, 14: 21, 15: 25, 16: 29, 17: 34, 18: 40, 19: 46, 20: 52,
};

describe('regression: wilcoxon signed-rank p-values are right for every n', () => {
    it('regression: null data gives p ≈ 1, not ≈ 0', () => {
        const before = Array.from({ length: 10 }, () => 50);
        const after = [51, 49, 52, 48, 53, 47, 54, 46, 55, 45]; // ±1..±5, symmetric: W+ = W− = 27.5
        const r = wilcoxonSignedRank(before, after);
        expect(r.n).toBe(10);
        expect(r.wPlus).toBe(27.5);
        expect(r.pValue).toBe(1);
        const small = wilcoxonSignedRank([0, 0, 0], [1, -2, 3]);
        expect(small.pValue).toBeGreaterThan(0.5);
    });

    it('regression: all-positive n = 10 gives the exact two-sided p = 2/1024', () => {
        const before = Array.from({ length: 10 }, (_, i) => i);
        const after = before.map((b, i) => b + i + 1);
        const r = wilcoxonSignedRank(before, after);
        expect(r.wPlus).toBe(55);
        expect(r.pValue).toBeCloseTo(2 / 1024, 15);
        // n = 5 all positive cannot reach 0.05 two-sided: 2/32.
        expect(wilcoxonSignedRank([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]).pValue).toBeCloseTo(2 / 32, 15);
    });

    it('regression: matches the textbook α = 0.05 critical values for n = 5..20', () => {
        // n = 5: no critical value exists at α = 0.05 two-sided (the most extreme W gives 0.0625).
        expect(wilcoxonSignedRank(pairsWithWPlus(5, 0).before, pairsWithWPlus(5, 0).after).pValue).toBeGreaterThan(0.05);
        for (const [nKey, t] of Object.entries(CRITICAL_05)) {
            const n = Number(nKey);
            const at = pairsWithWPlus(n, t);
            const past = pairsWithWPlus(n, t + 1);
            expect({ n, p: wilcoxonSignedRank(at.before, at.after).pValue <= 0.05 }).toEqual({ n, p: true });
            expect({ n, p: wilcoxonSignedRank(past.before, past.after).pValue > 0.05 }).toEqual({ n, p: true });
        }
    });

    it('regression: at n = 50 the normal approximation matches the exact distribution (same n, same W+)', () => {
        const n = 50;
        const doubledRanks = Array.from({ length: n }, (_, i) => 2 * (i + 1));
        // Within 0.001 where decisions are made (p < 0.1), within 0.005 in the body of the distribution.
        for (const w of [300, 400, 450, 500, 560, 600, 637]) {
            const exact = exactSignedRankPValue(doubledRanks, 2 * w);
            const normal = normalSignedRankPValue(n, w, []);
            const tolerance = exact < 0.1 ? 0.001 : 0.005;
            expect({ w, close: Math.abs(exact - normal) < tolerance }).toEqual({ w, close: true });
        }
    });

    it('regression: the large-n branch is a real two-sided normal tail, centred and far-tail', () => {
        const n = 60;
        const sym = pairsWithWPlus(n, (n * (n + 1)) / 4);
        expect(wilcoxonSignedRank(sym.before, sym.after).pValue).toBeGreaterThan(0.95);
        const allUp = wilcoxonSignedRank(Array.from({ length: n }, () => 0), Array.from({ length: n }, (_, i) => i + 1));
        expect(allUp.pValue).toBeLessThan(1e-9);
    });

    it('regression: a far tail does not underflow to 0 (n = 100 all positive ≈ 4e-18)', () => {
        const n = 100;
        const r = wilcoxonSignedRank(Array.from({ length: n }, () => 0), Array.from({ length: n }, (_, i) => i + 1));
        expect(r.pValue).toBeGreaterThan(1e-19);
        expect(r.pValue).toBeLessThan(1e-16);
    });

    it('regression: ties share mid-ranks and the exact p-value conditions on them', () => {
        // |d| = 1,1,2,2 all positive: W+ = 1.5+1.5+3.5+3.5 = 10 = the maximum; P = 1/16 one-sided.
        const r = wilcoxonSignedRank([0, 0, 0, 0], [1, 1, 2, 2]);
        expect(r.wPlus).toBe(10);
        expect(r.pValue).toBeCloseTo(2 / 16, 15);
    });
});

describe('regression: the paired Hodges–Lehmann estimate counts zero differences', () => {
    it('regression: zeros are Walsh-averaged, not dropped', () => {
        // d = 0, 0, 0, 1: Walsh averages are six 0s, three 0.5s and one 1 → median 0 (dropping zeros gave 1).
        const r = wilcoxonSignedRank([5, 5, 5, 5], [5, 5, 5, 6]);
        expect(r.n).toBe(1);
        expect(r.hodgesLehmann).toBe(0);
    });

    it('regression: the O(n log n) selection returns exactly the brute-force median (ties, zeros, clusters)', () => {
        const rng = createRng(17);
        for (let trial = 0; trial < 200; trial++) {
            const n = 1 + Math.floor(rng() * 40);
            const kind = trial % 4;
            const d = Array.from({ length: n }, () => {
                const u = rng();
                if (kind === 0) return u * 10 - 5; // continuous
                if (kind === 1) return Math.round(u * 6) - 3; // heavy ties and zeros
                if (kind === 2) return u < 0.6 ? 0 : u * 1e-12; // a big cluster of zeros next to tiny values
                return u < 0.5 ? 1e9 : -1e-9 * u; // wildly mixed scales
            });
            const zeros = d.map(() => 0);
            expect({ trial, hl: wilcoxonSignedRank(zeros, d).hodgesLehmann }).toEqual({ trial, hl: bruteHodgesLehmann(d) });
        }
    });
});

describe('regression: non-finite differences are dropped and counted', () => {
    it('regression: NaN / Infinity pairs never reach the ranks or the estimate', () => {
        const r = wilcoxonSignedRank([0, 0, 0, 0, Number.NaN, 0], [1, 2, 3, 4, 5, Number.POSITIVE_INFINITY]);
        expect(r.nonFinite).toBe(2);
        expect(r.n).toBe(4);
        expect(r.pValue).toBeCloseTo(2 / 16, 15);
        expect(Number.isFinite(r.hodgesLehmann)).toBe(true);
        expect(wilcoxonSignedRank([1, 2], [2, 3]).nonFinite).toBe(0);
        const ci = pairedBootstrapCi([0, 0, Number.NaN], [1, 1, 1], { seed: 1, statistic: 'hodges-lehmann' });
        expect(ci).toEqual({ low: 1, high: 1, estimate: 1 });
    });
});

describe('regression: the paired bootstrap can resample the Hodges–Lehmann estimate', () => {
    it('regression: statistic "hodges-lehmann" is seeded, fixed-count and centred on the HL estimate', () => {
        const before = [10, 12, 11, 13, 15, 9, 14, 10, 12, 11];
        const after = [11, 13, 12, 14, 16, 10, 15, 11, 30, 11];
        const hl = pairedBootstrapCi(before, after, { seed: 3, statistic: 'hodges-lehmann' });
        expect(pairedBootstrapCi(before, after, { seed: 3, statistic: 'hodges-lehmann' })).toEqual(hl);
        expect(hl.estimate).toBe(wilcoxonSignedRank(before, after).hodgesLehmann);
        expect(hl.low).toBeLessThanOrEqual(hl.estimate);
        expect(hl.high).toBeGreaterThanOrEqual(hl.estimate);
        // The outlier drags the mean, not the HL estimate.
        const mean = pairedBootstrapCi(before, after, { seed: 3 });
        expect(mean.estimate).toBeGreaterThan(hl.estimate);
        expect(pairedBootstrapCi(before, after, { seed: 3, statistic: 'mean' })).toEqual(mean);
    });

    it('regression: the HL bootstrap is O(B·n log n), not O(B·n² log n) — within a small factor of the mean bootstrap', () => {
        // Relative, so it holds on any machine: at n = 200 the old n² Walsh list cost ~50× the mean bootstrap.
        const rng = createRng(5);
        const before = Array.from({ length: 200 }, () => 100 + rng() * 10);
        const after = before.map((b) => b + 1 + rng() * 4);
        const time = (statistic: 'mean' | 'hodges-lehmann') => {
            const started = performance.now();
            const ci = pairedBootstrapCi(before, after, { seed: 9, statistic });
            return { ms: performance.now() - started, ci };
        };
        time('hodges-lehmann'); // warm both paths
        const mean = time('mean');
        const hl = time('hodges-lehmann');
        expect(hl.ms).toBeLessThan(10 * Math.max(mean.ms, 5));
        expect(hl.ci.low).toBeLessThanOrEqual(hl.ci.estimate);
        expect(hl.ci.high).toBeGreaterThanOrEqual(hl.ci.estimate);
    });
});
