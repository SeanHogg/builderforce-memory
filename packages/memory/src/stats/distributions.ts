/**
 * Distribution functions for the stats module (PRD 41 Part D / D13): Beta posterior
 * and quantile, Gamma quantile and the exact Poisson (Garwood) interval. Pure and
 * deterministic; the incomplete-beta and incomplete-gamma implementations follow the
 * continued-fraction / series methods (Numerical Recipes), which are stable for the
 * shape ranges Part D uses (posterior counts and sample sizes).
 */

// Bound once at module scope: a bare `Math.x` is a global lookup on every call, which inside a vm context (the jest
// runner) costs ~1 µs each and made the inverse beta ~100× slower there; a module binding is free everywhere.
const { abs, exp, log, max, min, pow, sin, sqrt, PI } = Math;
const isFiniteNumber = Number.isFinite;

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
        return log(PI / sin(PI * x)) - logGamma(1 - x);
    }
    const z = x - 1;
    let sum = 0.99999999999980993;
    for (let i = 0; i < p.length; i++) sum += p[i]! / (z + i + 1);
    const t = z + p.length - 0.5;
    return 0.5 * log(2 * PI) + (z + 0.5) * log(t) - t + log(sum);
}

function betaCf(a: number, b: number, x: number): number {
    const qab = a + b;
    const qap = a + 1;
    const qam = a - 1;
    let c = 1;
    let d = 1 - (qab * x) / qap;
    if (abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAX_ITER; m++) {
        const m2 = 2 * m;
        let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
        d = 1 + aa * d;
        if (abs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c;
        if (abs(c) < FPMIN) c = FPMIN;
        d = 1 / d;
        h *= d * c;
        aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
        d = 1 + aa * d;
        if (abs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c;
        if (abs(c) < FPMIN) c = FPMIN;
        d = 1 / d;
        const del = d * c;
        h *= del;
        if (abs(del - 1) < EPS) break;
    }
    return h;
}

/** Regularized incomplete beta I_x(a, b). */
export function regularizedIncompleteBeta(x: number, a: number, b: number): number {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = exp(
        logGamma(a + b) - logGamma(a) - logGamma(b) + a * log(x) + b * log(1 - x),
    );
    if (x < (a + 1) / (a + b + 2)) return (bt * betaCf(a, b, x)) / a;
    return 1 - (bt * betaCf(b, a, 1 - x)) / b;
}

/** The inverse stops when its Newton step is below this RELATIVE to the nearer end's distance, min(x, 1 − x). */
const INVERSE_BETA_TOL = 1e-10;
/** Hard cap on the inverse's steps (each costs one I_x evaluation). */
const INVERSE_BETA_MAX_STEPS = 100;

/** Starting point of the inverse (Numerical Recipes 3e §6.14 `invbetai`): a normal approximation, or the tail power law below a, b = 1. */
function inverseBetaGuess(p: number, a: number, b: number): number {
    if (a >= 1 && b >= 1) {
        const pp = p < 0.5 ? p : 1 - p;
        const t = sqrt(-2 * log(pp));
        let x = (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t;
        if (p < 0.5) x = -x;
        const al = (x * x - 3) / 6;
        const h = 2 / (1 / (2 * a - 1) + 1 / (2 * b - 1));
        const w = (x * sqrt(al + h)) / h - (1 / (2 * b - 1) - 1 / (2 * a - 1)) * (al + 5 / 6 - 2 / (3 * h));
        return a / (a + b * exp(2 * w));
    }
    const lna = log(a / (a + b));
    const lnb = log(b / (a + b));
    const t = exp(a * lna) / a;
    const u = exp(b * lnb) / b;
    const w = t + u;
    return p < t / w ? pow(a * w * p, 1 / a) : 1 - pow(b * w * (1 - p), 1 / b);
}

/** The scale a tolerance is relative to: the distance to the nearer end of (0, 1), so both tails keep their digits. */
const nearEnd = (x: number): number => max(min(x, 1 - x), 1e-300);

/**
 * Inverse of the regularized incomplete beta: Halley steps on I_x(a, b) − p, kept inside a shrinking bracket [lo, hi].
 *   - Converged when the Newton step |f / density| is below INVERSE_BETA_TOL · min(x, 1 − x) (relative, so a quantile near 0
 *     or 1 keeps its accuracy); the step is then applied and clamped to the bracket. The test runs on the RAW step, before any
 *     bracket fallback, and a step landing exactly on a bracket end is converged too (a step under 1 ulp).
 *   - A step that leaves the bracket, or a degenerate density, falls back to the bracket's midpoint.
 *   - If INVERSE_BETA_MAX_STEPS pass without converging, plain bisection finishes the bracket to tolerance (or to the last
 *     representable midpoint): an unconverged x is never returned.
 * Deterministic: a fixed start and a fixed sequence of steps.
 */
export function inverseRegularizedIncompleteBeta(p: number, a: number, b: number): number {
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    const lnBeta = logGamma(a) + logGamma(b) - logGamma(a + b);
    let lo = 0;
    let hi = 1;
    const guess = inverseBetaGuess(p, a, b);
    let x = guess > 0 && guess < 1 ? guess : 0.5;
    for (let i = 0; i < INVERSE_BETA_MAX_STEPS; i++) {
        const f = regularizedIncompleteBeta(x, a, b) - p;
        if (f === 0) return x;
        if (f < 0) lo = x;
        else hi = x;
        const density = exp((a - 1) * log(x) + (b - 1) * log(1 - x) - lnBeta);
        let next = -1; // outside every bracket: falls back to bisection
        if (density > 0 && isFiniteNumber(density)) {
            const newton = f / density;
            next = x - newton / (1 - 0.5 * min(1, newton * ((a - 1) / x - (b - 1) / (1 - x))));
            // Converged: judged on the undamped Newton distance (a Halley damping can shrink a far step), before any fallback.
            if (abs(newton) < INVERSE_BETA_TOL * nearEnd(x) || next === lo || next === hi) {
                return min(max(next, lo), hi);
            }
        }
        if (!(next > lo && next < hi)) next = (lo + hi) / 2;
        x = next;
    }
    // Step cap hit: bisect the bracket to tolerance.
    for (;;) {
        const mid = (lo + hi) / 2;
        if (mid <= lo || mid >= hi || hi - lo < INVERSE_BETA_TOL * nearEnd(mid)) return mid;
        if (regularizedIncompleteBeta(mid, a, b) < p) lo = mid;
        else hi = mid;
    }
}

/** Posterior Beta(α₀ + k, β₀ + n − k): mean, variance and credible quantiles. */
export interface BetaPosterior {
    alpha: number;
    beta: number;
    mean: number;
    variance: number;
    quantile(p: number): number;
}

/** Mean α/(α+β) and variance αβ/((α+β)²(α+β+1)) of Beta(α, β). */
export function betaMoments(alpha: number, beta: number): { mean: number; variance: number } {
    const total = alpha + beta;
    return { mean: alpha / total, variance: (alpha * beta) / (total * total * (total + 1)) };
}

export function betaPosterior(alpha0: number, beta0: number, k: number, n: number): BetaPosterior {
    const alpha = alpha0 + k;
    const beta = beta0 + (n - k);
    return {
        alpha,
        beta,
        ...betaMoments(alpha, beta),
        quantile: (p) => inverseRegularizedIncompleteBeta(p, alpha, beta),
    };
}

/** Two-sided 95% standard-normal critical value. */
const Z_95 = 1.96;

/**
 * Wilson score interval of `successes / n` at critical value `z` (default 95%): stays
 * inside [0, 1] and keeps a non-zero width at 0 or n successes, where the Wald interval
 * collapses. `n` not > 0 (including NaN) or a non-finite `successes` is no evidence:
 * [0, 1]. A `z` that is not a finite positive number is a caller error and throws.
 */
export function wilsonInterval(successes: number, n: number, z: number = Z_95): { low: number; high: number } {
    if (!(z > 0) || !isFiniteNumber(z)) throw new RangeError(`wilsonInterval: z must be a finite positive number, got ${z}`);
    if (!(n > 0) || !isFiniteNumber(successes)) return { low: 0, high: 1 };
    const p = min(1, max(0, successes / n));
    const z2 = z * z;
    const denom = 1 + z2 / n;
    const centre = (p + z2 / (2 * n)) / denom;
    const half = (z * sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
    return { low: max(0, centre - half), high: min(1, centre + half) };
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
        if (abs(d) < FPMIN) d = FPMIN;
        c = b + an / c;
        if (abs(c) < FPMIN) c = FPMIN;
        d = 1 / d;
        const del = d * c;
        h *= del;
        if (abs(del - 1) < EPS) break;
    }
    return exp(-x + a * log(x) - logGamma(a)) * h;
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
            if (abs(term) < abs(sum) * EPS) break;
        }
        return exp(-x + a * log(x) - logGamma(a)) * sum;
    }
    return 1 - gammaQ(a, x);
}

/**
 * Regularized UPPER incomplete gamma Q(a, x) = 1 − P(a, x), computed directly in the
 * tail (continued fraction) so a tiny Q does not underflow to 0 through 1 − P. Module-
 * internal to `./stats` (the normal tail of the Wilcoxon large-n branch); not a public export.
 */
export function regularizedUpperIncompleteGamma(a: number, x: number): number {
    if (x <= 0) return 1;
    return x < a + 1 ? 1 - regularizedIncompleteGamma(a, x) : gammaQ(a, x);
}

/** Gamma(shape, scale) quantile by bisection on the CDF. Deterministic. */
export function gammaQuantile(p: number, shape: number, scale = 1): number {
    if (p <= 0) return 0;
    if (p >= 1) return Infinity;
    let lo = 0;
    let hi = max(shape * scale * 2, scale, 1e-9);
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

/** The Jeffreys prior Beta(½, ½) of {@link rateInterval}. */
export const RATE_PRIOR = 0.5;
/** Mass of {@link rateInterval}: an equal-tailed 90% interval. */
export const RATE_CREDIBLE_MASS = 0.9;

/**
 * The credible interval of an observed rate k of n: the equal-tailed RATE_CREDIBLE_MASS interval of the Jeffreys
 * posterior Beta(½ + k, ½ + n − k), with Brown–Cai–DasGupta's boundary rule (low = 0 at k = 0, high = 1 at k = n) so
 * the interval always holds k/n. THE interval of every rate a graph badge or a hop shows (PRD 38, PRD 41 D3).
 */
export function rateInterval(k: number, n: number): { low: number; high: number } {
    const post = betaPosterior(RATE_PRIOR, RATE_PRIOR, k, n);
    const tail = (1 - RATE_CREDIBLE_MASS) / 2;
    return {
        low: k <= 0 ? 0 : post.quantile(tail),
        high: k >= n ? 1 : post.quantile(1 - tail),
    };
}
