/**
 * Statistical tests for the stats module (PRD 41 Part D / D13): Wilcoxon signed-rank
 * (exact per n, normal approximation for large n) with the paired Hodges–Lehmann
 * estimator, a seeded paired bootstrap CI (mean or Hodges–Lehmann), and the sign-test
 * SPRT. All deterministic under their seeds.
 */

import { regularizedUpperIncompleteGamma } from "./distributions.js";
import { createRng, type Rng } from "./prng.js";

// Bound once at module scope for the hot loops: a bare `Math.x` is a global lookup on every call, which inside a vm
// context (the jest runner) costs ~1 µs each (the same reason as distributions.ts).
const { floor, max, min } = Math;

/** Up to this many nonzero differences the p-value is EXACT; above it, the normal approximation. */
export const WILCOXON_EXACT_MAX_N = 50;

/**
 * Upper standard-normal tail P(Z ≥ z) for z ≥ 0: ½·erfc(z/√2) = ½·Q(½, z²/2), with Q taken
 * directly (not 1 − P) so a far tail stays a tiny positive number instead of underflowing to 0.
 */
function normalUpperTail(z: number): number {
    if (z <= 0) return 0.5;
    return 0.5 * regularizedUpperIncompleteGamma(0.5, (z * z) / 2);
}

/** The paired differences `after − before` that are finite, and how many were not. */
function finiteDifferences(before: readonly number[], after: readonly number[]): { diffs: number[]; nonFinite: number } {
    const diffs: number[] = [];
    let nonFinite = 0;
    const n = Math.min(before.length, after.length);
    for (let i = 0; i < n; i++) {
        const d = after[i]! - before[i]!;
        if (Number.isFinite(d)) diffs.push(d);
        else nonFinite++;
    }
    return { diffs, nonFinite };
}

/**
 * Exact two-sided p-value: the permutation distribution of W+ given THESE ranks (mid-ranks
 * for ties), each sign ± with probability ½. Ranks are doubled so mid-ranks are integers,
 * and the distribution is built by DP over subset sums for this n — never a table built for
 * another n. Probabilities (halved at each step) rather than counts, so no overflow.
 */
export function exactSignedRankPValue(doubledRanks: readonly number[], doubledWPlus: number): number {
    const total = doubledRanks.reduce((a, b) => a + b, 0);
    let dp = new Float64Array(total + 1);
    dp[0] = 1;
    for (const r of doubledRanks) {
        const next = new Float64Array(total + 1);
        for (let s = 0; s <= total; s++) {
            const p = dp[s]!;
            if (p === 0) continue;
            next[s] = next[s]! + p / 2;
            next[s + r] = next[s + r]! + p / 2;
        }
        dp = next;
    }
    let lower = 0; // P(W+ ≤ observed)
    let upper = 0; // P(W+ ≥ observed)
    for (let s = 0; s <= total; s++) {
        if (s <= doubledWPlus) lower += dp[s]!;
        if (s >= doubledWPlus) upper += dp[s]!;
    }
    return Math.min(1, 2 * Math.min(lower, upper));
}

/**
 * Normal approximation for large n: mean n(n+1)/4, variance n(n+1)(2n+1)/24 reduced by
 * Σ(t³ − t)/48 over tie groups, continuity correction ½, two-sided normal tail.
 */
export function normalSignedRankPValue(n: number, wPlus: number, tieGroups: readonly number[]): number {
    const mean = (n * (n + 1)) / 4;
    const tieCorrection = tieGroups.reduce((acc, t) => acc + (t * t * t - t), 0) / 48;
    const variance = (n * (n + 1) * (2 * n + 1)) / 24 - tieCorrection;
    if (!(variance > 0)) return 1;
    const z = Math.max(0, Math.abs(wPlus - mean) - 0.5) / Math.sqrt(variance);
    return Math.min(1, 2 * normalUpperTail(z));
}

/** Walsh averages (sᵢ + sⱼ)/2, i ≤ j, of an ASCENDING array, that are ≤ x: one O(n) two-pointer sweep. */
function walshCountAtMost(s: readonly number[], x: number): number {
    const n = s.length;
    let count = 0;
    let j = n - 1;
    for (let i = 0; i < n; i++) {
        while (j >= i && (s[i]! + s[j]!) / 2 > x) j--;
        if (j < i) break;
        count += j - i + 1;
    }
    return count;
}

/** Bisection steps per round before a stuck range (a cluster of equal averages) is stepped past. */
const WALSH_BISECTION_MAX_STEPS = 128;

/** The smallest Walsh average of an ASCENDING array strictly above `x` (Infinity if none): O(n). */
function walshNextAbove(s: readonly number[], x: number): number {
    const n = s.length;
    let best = Infinity;
    let j = n - 1; // last j with avg ≤ x for the current i
    for (let i = 0; i < n; i++) {
        while (j >= i && (s[i]! + s[j]!) / 2 > x) j--;
        const first = max(i, j + 1);
        if (first < n) best = min(best, (s[i]! + s[first]!) / 2);
    }
    return best;
}

/** The largest Walsh average of an ASCENDING array at most `x` (−Infinity if none): O(n). */
function walshLargestAtMost(s: readonly number[], x: number): number {
    const n = s.length;
    let best = -Infinity;
    let j = n - 1;
    for (let i = 0; i < n; i++) {
        while (j >= i && (s[i]! + s[j]!) / 2 > x) j--;
        if (j < i) break;
        const v = (s[i]! + s[j]!) / 2;
        if (v > best) best = v;
    }
    return best;
}

/**
 * The kLow-th and kHigh-th smallest (0-based; kHigh ∈ {kLow, kLow + 1}) Walsh averages of
 * an ASCENDING array, EXACTLY, without materialising the n(n+1)/2 averages — an exact
 * value-space selection (same result as Monahan's algorithm), O(n log n) for the sort plus
 * O(n) per bisection step:
 *   - bisect on the VALUE with an O(n) two-pointer count until at most n averages lie in
 *     (lo, hi] around both ranks, then enumerate just those and select;
 *   - when a bisection point falls between the two ranks, the kLow-th is the largest
 *     average ≤ it and the kHigh-th the smallest above it;
 *   - a range that will not shrink (a cluster of EQUAL averages) is stepped past one
 *     distinct value at a time.
 * Each candidate is the same expression (sᵢ + sⱼ)/2 as a full sort, so the doubles match it.
 */
function kthWalshAverages(s: readonly number[], kLow: number, kHigh: number): [number, number] {
    const n = s.length;
    let lo = s[0]!; // the smallest average
    let loCount = walshCountAtMost(s, lo);
    if (loCount > kHigh) return [lo, lo];
    if (loCount > kLow) return [lo, walshNextAbove(s, lo)]; // kHigh = kLow + 1 = loCount
    let hi = s[n - 1]!; // the largest average (its count, m, exceeds every rank)
    let hiCount = walshCountAtMost(s, hi);
    // Invariant: loCount ≤ kLow and hiCount > kHigh.
    for (;;) {
        for (let step = 0; step < WALSH_BISECTION_MAX_STEPS && hiCount - loCount > n; step++) {
            const mid = lo + (hi - lo) / 2;
            if (mid <= lo || mid >= hi) break;
            const c = walshCountAtMost(s, mid);
            if (c > kHigh) { hi = mid; hiCount = c; }
            else if (c <= kLow) { lo = mid; loCount = c; }
            else return [walshLargestAtMost(s, mid), walshNextAbove(s, mid)]; // c = kHigh = kLow + 1
        }
        if (hiCount - loCount <= n) break;
        // Stuck on a cluster of equal averages: everything in (lo, next] equals `next`.
        const next = walshNextAbove(s, lo);
        const c = walshCountAtMost(s, next);
        if (c > kHigh) return [next, next];
        if (c > kLow) return [next, walshNextAbove(s, next)];
        lo = next;
        loCount = c;
    }
    // Both ranks lie in (lo, hi]: enumerate those averages (at most n) and select.
    const candidates: number[] = [];
    let jLo = n - 1; // last j with avg ≤ lo (for the current i)
    let jHi = n - 1; // last j with avg ≤ hi
    for (let i = 0; i < n; i++) {
        while (jHi >= i && (s[i]! + s[jHi]!) / 2 > hi) jHi--;
        while (jLo >= i && (s[i]! + s[jLo]!) / 2 > lo) jLo--;
        if (jHi < i) break;
        for (let j = max(i, jLo + 1); j <= jHi; j++) candidates.push((s[i]! + s[j]!) / 2);
    }
    candidates.sort((a, b) => a - b);
    return [candidates[kLow - loCount] ?? hi, candidates[kHigh - loCount] ?? hi];
}

/**
 * Median of the Walsh averages (dᵢ + dⱼ)/2, i ≤ j, over ALL the given differences (zeros
 * included), in O(n log n): one sort, then exact selection of the middle one or two
 * ({@link kthWalshAverages}).
 */
function hodgesLehmannOf(diffs: readonly number[]): number {
    const n = diffs.length;
    if (n === 0) return 0;
    const s = [...diffs].sort((a, b) => a - b);
    const m = (n * (n + 1)) / 2;
    if (m % 2 === 1) return kthWalshAverages(s, (m - 1) / 2, (m - 1) / 2)[0];
    const [a, b] = kthWalshAverages(s, m / 2 - 1, m / 2);
    return (a + b) / 2;
}

export interface WilcoxonResult {
    /** Nonzero differences — the ones the test ranks (zeros are dropped: Wilcoxon's rule). */
    n: number;
    /** Sum of the (mid-)ranks of the positive differences. */
    wPlus: number;
    /** Two-sided. Exact for n ≤ {@link WILCOXON_EXACT_MAX_N}, normal approximation (continuity + tie corrected) above. */
    pValue: number;
    /**
     * Paired Hodges–Lehmann estimator: median of the Walsh averages of ALL finite paired
     * differences, zeros included (Hollander & Wolfe). R's `wilcox.test(conf.int = TRUE)` drops
     * zeros for its estimate too; this does not, because dropping them biases the estimate away
     * from 0.
     */
    hodgesLehmann: number;
    /**
     * Pairs whose difference was not a finite number (NaN or ±Infinity, e.g. a missing or
     * failed measurement). They are DROPPED from the test and the estimate, never ranked;
     * a caller that cannot tolerate missing pairs should check this is 0.
     */
    nonFinite: number;
}

/**
 * Wilcoxon signed-rank test of `after − before`, plus the paired Hodges–Lehmann location
 * estimator.
 *
 * - Zeros: Wilcoxon's rule — zero differences are dropped before ranking (`n` counts the
 *   rest). They still count toward the Hodges–Lehmann estimate, which would otherwise be
 *   biased away from 0.
 * - Ties: tied |differences| share their mid-rank. The exact p-value conditions on those
 *   ranks; the normal approximation applies the tie variance correction.
 * - Non-finite differences (NaN, ±Infinity) are dropped and counted in `nonFinite`.
 */
export function wilcoxonSignedRank(before: readonly number[], after: readonly number[]): WilcoxonResult {
    const { diffs: all, nonFinite } = finiteDifferences(before, after);
    const hodgesLehmann = hodgesLehmannOf(all);
    const diffs = all.filter((d) => d !== 0);
    const n = diffs.length;
    if (n === 0) return { n: 0, wPlus: 0, pValue: 1, hodgesLehmann, nonFinite };
    const abs = diffs.map((d) => Math.abs(d));
    const order = abs.map((_, i) => i).sort((p, q) => abs[p]! - abs[q]!);
    const ranks = new Array<number>(n).fill(0);
    const tieGroups: number[] = [];
    let i = 0;
    while (i < n) {
        let j = i;
        while (j + 1 < n && abs[order[j + 1]!] === abs[order[i]!]) j++;
        const avg = (i + j) / 2 + 1;
        for (let k = i; k <= j; k++) ranks[order[k]!] = avg;
        if (j > i) tieGroups.push(j - i + 1);
        i = j + 1;
    }
    let wPlus = 0;
    for (let k = 0; k < n; k++) if (diffs[k]! > 0) wPlus += ranks[k]!;
    const pValue =
        n <= WILCOXON_EXACT_MAX_N
            ? exactSignedRankPValue(ranks.map((r) => Math.round(2 * r)), Math.round(2 * wPlus))
            : normalSignedRankPValue(n, wPlus, tieGroups);
    return { n, wPlus, pValue, hodgesLehmann, nonFinite };
}

/** The statistic a paired bootstrap resamples. */
export type PairedStatistic = "mean" | "hodges-lehmann";

/**
 * Paired bootstrap percentile confidence interval for a location statistic of
 * `after − before` — the mean difference (default) or the paired Hodges–Lehmann estimate
 * (PRD 41 C6.2) — resampling PAIRS so the pairing is preserved. Deterministic: seeded
 * (`seed`, default 1) with a fixed resample count (`iterations`, default 2000). The
 * Hodges–Lehmann statistic costs O(n log n) per resample (exact selection, no n² list).
 * Pairs with a non-finite difference are dropped, as in {@link wilcoxonSignedRank}.
 */
export function pairedBootstrapCi(
    before: readonly number[],
    after: readonly number[],
    options: { iterations?: number; confidence?: number; seed?: number; statistic?: PairedStatistic } = {},
): { low: number; high: number; estimate: number } {
    const iterations = options.iterations ?? 2000;
    const confidence = options.confidence ?? 0.95;
    const stat = options.statistic === "hodges-lehmann"
        ? hodgesLehmannOf
        : (d: readonly number[]) => d.reduce((a, b) => a + b, 0) / Math.max(d.length, 1);
    const rng: Rng = createRng(options.seed ?? 1);
    const { diffs } = finiteDifferences(before, after);
    const n = diffs.length;
    if (n === 0) return { low: 0, high: 0, estimate: 0 };
    const estimate = stat(diffs);
    const values: number[] = [];
    const sample = new Array<number>(n);
    for (let b = 0; b < iterations; b++) {
        for (let i = 0; i < n; i++) sample[i] = diffs[floor(rng() * n)]!;
        values.push(stat(sample));
    }
    values.sort((a, b) => a - b);
    const alpha = 1 - confidence;
    return {
        low: values[Math.floor((alpha / 2) * iterations)] ?? estimate,
        high: values[Math.floor((1 - alpha / 2) * iterations) - 1] ?? estimate,
        estimate,
    };
}

export interface SprtResult {
    decision: "reject-null" | "accept-null" | "continue";
    likelihoodRatio: number;
    observations: number;
}

/**
 * Sequential probability ratio test on paired signs. H0: P(after > before) = p0,
 * H1: = p1. Decides as soon as the likelihood ratio crosses a boundary, else reports
 * `continue` at `maxObservations`. Deterministic — the observations are supplied.
 */
export function signTestSprt(
    pairs: ReadonlyArray<readonly [number, number]>,
    options: {
        p0?: number;
        p1?: number;
        alpha?: number;
        beta?: number;
        alternative?: "greater" | "less";
    } = {},
): SprtResult {
    const p0 = options.p0 ?? 0.5;
    const p1 = options.p1 ?? 0.7;
    const alpha = options.alpha ?? 0.05;
    const beta = options.beta ?? 0.1;
    const aBound = Math.log((1 - beta) / alpha);
    const bBound = Math.log(beta / (1 - alpha));
    let llr = 0;
    let observations = 0;
    for (const [beforeV, afterV] of pairs) {
        const plus = afterV > beforeV;
        const inc = Math.log((plus ? p1 : 1 - p1) / (plus ? p0 : 1 - p0));
        const incAlt =
            options.alternative === "less"
                ? Math.log((plus ? 1 - p1 : p1) / (plus ? 1 - p0 : p0))
                : inc;
        llr += incAlt;
        observations++;
        if (llr >= aBound) return { decision: "reject-null", likelihoodRatio: llr, observations };
        if (llr <= bBound) return { decision: "accept-null", likelihoodRatio: llr, observations };
    }
    return { decision: "continue", likelihoodRatio: llr, observations };
}
