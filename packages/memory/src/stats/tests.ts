/**
 * Statistical tests for the stats module (PRD 41 Part D / D13): Wilcoxon signed-rank
 * with the paired Hodges–Lehmann estimator, a seeded paired bootstrap CI, and the
 * sign-test SPRT. All deterministic under their seeds.
 */

import { createRng, type Rng } from "./prng.js";

const WILCOXON_P: number[] = (() => {
    // Exact null distribution of W+ for n <= 60 via DP over subset sums.
    const n = 60;
    const total = (n * (n + 1)) / 2;
    const dp: number[] = new Array(total + 1).fill(0);
    dp[0] = 1;
    for (let rank = 1; rank <= n; rank++) {
        for (let s = total; s >= rank; s--) dp[s] = (dp[s] ?? 0) + (dp[s - rank] ?? 0);
    }
    const cum: number[] = new Array(total + 1).fill(0);
    let running = 0;
    for (let s = 0; s <= total; s++) {
        running += dp[s]!;
        cum[s] = running / 2 ** n;
    }
    return cum;
})();

function wilcoxonPValue(wPlus: number, n: number): number {
    if (n > 60) {
        // Normal approximation with continuity correction.
        const mean = (n * (n + 1)) / 4;
        const sd = Math.sqrt((n * (n + 1) * (2 * n + 1)) / 24);
        const z = (wPlus - mean + 0.5) / sd;
        return Math.min(1, Math.exp(-0.5 * z * z) * 0.5 * 2); // two-sided via |z|
    }
    const total = (n * (n + 1)) / 2;
    const lo = Math.max(0, Math.floor(wPlus));
    const hi = Math.min(total, Math.ceil(wPlus));
    const pLo = WILCOXON_P[lo] ?? 0;
    const pHi = 1 - (WILCOXON_P[hi - 1] ?? 0);
    return Math.min(1, 2 * Math.min(pLo, pHi));
}

export interface WilcoxonResult {
    n: number;
    wPlus: number;
    pValue: number;
    /** Paired Hodges–Lehmann estimator: median of pairwise averages (after > 0 filter). */
    hodgesLehmann: number;
}

/**
 * Wilcoxon signed-rank test on the nonzero paired differences of `before` and
 * `after`, plus the paired Hodges–Lehmann location estimator.
 */
export function wilcoxonSignedRank(before: readonly number[], after: readonly number[]): WilcoxonResult {
    const diffs: number[] = [];
    const n0 = Math.min(before.length, after.length);
    for (let i = 0; i < n0; i++) {
        const d = after[i]! - before[i]!;
        if (d !== 0) diffs.push(d);
    }
    const n = diffs.length;
    if (n === 0) return { n: 0, wPlus: 0, pValue: 1, hodgesLehmann: 0 };
    const abs = diffs.map((d) => Math.abs(d));
    const order = abs.map((_, i) => i).sort((p, q) => abs[p]! - abs[q]!);
    const ranks = new Array<number>(n).fill(0);
    let i = 0;
    while (i < n) {
        let j = i;
        while (j + 1 < n && abs[order[j + 1]!] === abs[order[i]!]) j++;
        const avg = (i + j) / 2 + 1;
        for (let k = i; k <= j; k++) ranks[order[k]!] = avg;
        i = j + 1;
    }
    let wPlus = 0;
    for (let k = 0; k < n; k++) if (diffs[k]! > 0) wPlus += ranks[k]!;
    // Hodges–Lehmann: median of Walsh averages over positive-magnitude pairs.
    const walsh: number[] = [];
    for (let p = 0; p < n; p++) {
        for (let q = p; q < n; q++) walsh.push((diffs[p]! + diffs[q]!) / 2);
    }
    walsh.sort((a, b) => a - b);
    const hl =
        walsh.length % 2 === 1
            ? walsh[(walsh.length - 1) / 2]!
            : (walsh[walsh.length / 2 - 1]! + walsh[walsh.length / 2]!) / 2;
    return { n, wPlus, pValue: wilcoxonPValue(wPlus, n), hodgesLehmann: hl };
}

/**
 * Paired bootstrap confidence interval for the mean difference `after − before`,
 * resampling PAIRS so the pairing is preserved. Deterministic under `seed`.
 */
export function pairedBootstrapCi(
    before: readonly number[],
    after: readonly number[],
    options: { iterations?: number; confidence?: number; seed?: number } = {},
): { low: number; high: number; estimate: number } {
    const iterations = options.iterations ?? 2000;
    const confidence = options.confidence ?? 0.95;
    const rng: Rng = createRng(options.seed ?? 1);
    const n = Math.min(before.length, after.length);
    const diffs: number[] = [];
    for (let i = 0; i < n; i++) diffs.push(after[i]! - before[i]!);
    const estimate = diffs.reduce((a, b) => a + b, 0) / Math.max(diffs.length, 1);
    if (n === 0) return { low: 0, high: 0, estimate: 0 };
    const means: number[] = [];
    for (let b = 0; b < iterations; b++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += diffs[Math.floor(rng() * n)]!;
        means.push(s / n);
    }
    means.sort((a, b) => a - b);
    const alpha = 1 - confidence;
    return {
        low: means[Math.floor((alpha / 2) * iterations)] ?? estimate,
        high: means[Math.floor((1 - alpha / 2) * iterations) - 1] ?? estimate,
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
