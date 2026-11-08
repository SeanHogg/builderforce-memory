/**
 * Estimators for the stats module (PRD 41 Part D / D13): capture–recapture
 * (Chapman / Lincoln–Petersen), Kaplan–Meier with Greenwood variance, the
 * Goel–Okumoto reliability-growth MLE and Gamma–Poisson empirical Bayes. Pure and
 * deterministic; `insufficient-data` is reported, never guessed around.
 */

export type InsufficientData = { status: "insufficient-data" };

/** Chapman estimator: n̂ = ((m+1)(c+1)/(r+1)) − 1. Needs r ≥ 1 (and m, c ≥ 3 by the D-done-check). */
export function chapmanEstimate(
    capturedM: number,
    capturedC: number,
    recapturedR: number,
): { estimate: number } | InsufficientData {
    if (recapturedR < 1) return { status: "insufficient-data" };
    return { estimate: ((capturedM + 1) * (capturedC + 1)) / (recapturedR + 1) - 1 };
}

/** Lincoln–Petersen: n̂ = m·c / r (the biased form; Chapman above is preferred). */
export function lincolnPetersenEstimate(
    capturedM: number,
    capturedC: number,
    recapturedR: number,
): { estimate: number } | InsufficientData {
    if (recapturedR < 1) return { status: "insufficient-data" };
    return { estimate: (capturedM * capturedC) / recapturedR };
}

export interface KaplanMeierResult {
    /** Survival probability after the last event time (the "current" survival). */
    survival: number;
    /** Greenwood variance of that survival. */
    variance: number;
    /** (time, survival) steps at each distinct event time. */
    curve: Array<{ time: number; survival: number }>;
}

/** Kaplan–Meier product-limit estimator on (time, event) rows; censored rows have event = 0. */
export function kaplanMeier(rows: readonly { time: number; event: 0 | 1 }[]): KaplanMeierResult {
    const times = [...new Set(rows.map((r) => r.time))].sort((a, b) => a - b);
    let survival = 1;
    let greenwood = 0;
    const curve: Array<{ time: number; survival: number }> = [];
    for (const t of times) {
        const atRisk = rows.filter((r) => r.time >= t).length;
        const events = rows.filter((r) => r.time === t && r.event === 1).length;
        if (atRisk > 0 && events > 0) {
            survival *= 1 - events / atRisk;
            if (atRisk - events > 0) greenwood += events / (atRisk * (atRisk - events));
        }
        curve.push({ time: t, survival });
    }
    return { survival, variance: survival * survival * greenwood, curve };
}

/**
 * Goel–Okumoto NHPP MLE: mean failures μ(t) = a(1 − e^{−bt}). a is solved in
 * closed form given b; b is refined by bisection on the score equation
 * Σ tᵢ = n/b − n·t̄·e^{−b·t̄}/(1 − e^{−b·t̄}). Needs n ≥ 2 distinct times.
 */
export function goelOkumotoMLE(
    times: readonly number[],
): { a: number; b: number } | InsufficientData {
    const sorted = [...times].sort((a, b) => a - b);
    const n = sorted.length;
    if (n < 2) return { status: "insufficient-data" };
    const tMax = sorted[n - 1]!;
    const tMean = sorted.reduce((a, b) => a + b, 0) / n;
    const score = (b: number) => {
        if (b <= 0) return Infinity;
        const denom = 1 - Math.exp(-b * tMax);
        return n / b - (n * tMean * Math.exp(-b * tMean)) / denom;
    };
    let lo = 1e-9;
    let hi = 1;
    while (score(hi) > 0 && hi < 1e6) hi *= 2;
    for (let i = 0; i < 100; i++) {
        const mid = (lo + hi) / 2;
        if (score(mid) > 0) lo = mid;
        else hi = mid;
    }
    const b = (lo + hi) / 2;
    const a = n / (1 - Math.exp(-b * tMax));
    return { a, b };
}

/**
 * Gamma–Poisson empirical Bayes rate posterior. Prior Gamma(α₀, β₀) over the rate;
 * observing `count` events over `exposure` gives Gamma(α₀ + count, β₀ + exposure).
 */
export function gammaPoissonPosterior(
    alpha0: number,
    beta0: number,
    count: number,
    exposure: number,
): { alpha: number; beta: number; mean: number; variance: number } {
    const alpha = alpha0 + count;
    const beta = beta0 + exposure;
    return { alpha, beta, mean: alpha / beta, variance: alpha / (beta * beta) };
}
