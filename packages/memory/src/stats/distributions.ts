/**
 * Distribution functions for the stats module (PRD 41 Part D / D13): Beta posterior
 * and quantile, Gamma quantile and the exact Poisson (Garwood) interval. Pure and
 * deterministic; the incomplete-beta and incomplete-gamma implementations follow the
 * continued-fraction / series methods (Numerical Recipes), which are stable for the
 * shape ranges Part D uses (posterior counts and sample sizes).
 */

const MAX_ITER = 200;
const EPS = 3e-14;
const FPMIN = 1e-300;

function logGamma(x: number): number {
    // Lanczos approximation, sufficient for the moderate arguments here.
    const p = [
        676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    if (x < 0.5) {
        return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    }
    const z = x - 1;
    let sum = 0.99999999999980993;
    for (let i = 0; i < p.length; i++) sum += p[i]! / (z + i + 1);
    const t = z + p.length - 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}

function betaCf(a: number, b: number, x: number): number {
    const qab = a + b;
    const qap = a + 1;
    const qam = a - 1;
    let c = 1;
    let d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAX_ITER; m++) {
        const m2 = 2 * m;
        let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
        d = 1 + aa * d;
        if (Math.abs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c;
        if (Math.abs(c) < FPMIN) c = FPMIN;
        d = 1 / d;
        h *= d * c;
        aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
        d = 1 + aa * d;
        if (Math.abs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c;
        if (Math.abs(c) < FPMIN) c = FPMIN;
        d = 1 / d;
        const del = d * c;
        h *= del;
        if (Math.abs(del - 1) < EPS) break;
    }
    return h;
}

/** Regularized incomplete beta I_x(a, b). */
export function regularizedIncompleteBeta(x: number, a: number, b: number): number {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(
        logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x),
    );
    if (x < (a + 1) / (a + b + 2)) return (bt * betaCf(a, b, x)) / a;
    return 1 - (bt * betaCf(b, a, 1 - x)) / b;
}

/** Inverse of the regularized incomplete beta, by bisection. Deterministic. */
export function inverseRegularizedIncompleteBeta(p: number, a: number, b: number): number {
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    let lo = 0;
    let hi = 1;
    for (let i = 0; i < 100; i++) {
        const mid = (lo + hi) / 2;
        if (regularizedIncompleteBeta(mid, a, b) < p) lo = mid;
        else hi = mid;
    }
    return (lo + hi) / 2;
}

/** Posterior Beta(α₀ + k, β₀ + n − k): mean and credible quantiles. */
export interface BetaPosterior {
    alpha: number;
    beta: number;
    mean: number;
    quantile(p: number): number;
}

export function betaPosterior(alpha0: number, beta0: number, k: number, n: number): BetaPosterior {
    const alpha = alpha0 + k;
    const beta = beta0 + (n - k);
    return {
        alpha,
        beta,
        mean: alpha / (alpha + beta),
        quantile: (p) => inverseRegularizedIncompleteBeta(p, alpha, beta),
    };
}

function gammaQ(a: number, x: number): number {
    // Regularized upper incomplete gamma Q(a, x) via continued fraction.
    let b = x + 1 - a;
    let c = 1 / FPMIN;
    let d = 1 / b;
    let h = d;
    for (let i = 1; i <= MAX_ITER; i++) {
        const an = -i * (i - a);
        b += 2;
        d = an * d + b;
        if (Math.abs(d) < FPMIN) d = FPMIN;
        c = b + an / c;
        if (Math.abs(c) < FPMIN) c = FPMIN;
        d = 1 / d;
        const del = d * c;
        h *= del;
        if (Math.abs(del - 1) < EPS) break;
    }
    return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
}

/** Regularized lower incomplete gamma P(a, x). */
export function regularizedIncompleteGamma(a: number, x: number): number {
    if (x <= 0) return 0;
    if (x < a + 1) {
        // Series representation for P.
        let sum = 1 / a;
        let term = sum;
        for (let n = 1; n <= MAX_ITER; n++) {
            term *= x / (a + n);
            sum += term;
            if (Math.abs(term) < Math.abs(sum) * EPS) break;
        }
        return Math.exp(-x + a * Math.log(x) - logGamma(a)) * sum;
    }
    return 1 - gammaQ(a, x);
}

/** Gamma(shape, scale) quantile by bisection on the CDF. Deterministic. */
export function gammaQuantile(p: number, shape: number, scale = 1): number {
    if (p <= 0) return 0;
    if (p >= 1) return Infinity;
    let lo = 0;
    let hi = Math.max(shape * scale * 2, scale, 1e-9);
    while (regularizedIncompleteGamma(shape, hi / scale) < p) {
        hi *= 2;
        if (hi > 1e12) break;
    }
    for (let i = 0; i < 100; i++) {
        const mid = (lo + hi) / 2;
        if (regularizedIncompleteGamma(shape, mid / scale) < p) lo = mid;
        else hi = mid;
    }
    return (lo + hi) / 2;
}

/**
 * Exact Poisson confidence interval (Garwood, 1936): the equal-tailed interval
 * built from chi-square / gamma quantiles. `count` events over `exposure`.
 */
export function garwoodPoissonInterval(
    count: number,
    exposure = 1,
    confidence = 0.95,
): { low: number; high: number; rateLow: number; rateHigh: number } {
    const alpha = 1 - confidence;
    // Gamma quantiles with chi-square scaling: χ²_{2k, α} = 2·GammaQuantile(α, k).
    const low =
        count === 0 ? 0 : gammaQuantile(alpha / 2, count) * 2;
    const high = gammaQuantile(1 - alpha / 2, count + 1) * 2;
    return {
        low,
        high,
        rateLow: low / (2 * exposure),
        rateHigh: high / (2 * exposure),
    };
}
