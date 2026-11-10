/**
 * Sample moments for the stats module (PRD 41 Part D / D13): the ONE mean and
 * standard deviation for predicted measures, so callers stop hand-rolling them.
 */

/** Result of {@link meanAndSd}. */
export interface MeanAndSd {
    /** Number of finite values actually used. */
    n: number;
    /** Arithmetic mean of the finite values; NaN when `n` is 0. */
    mean: number;
    /** Sample standard deviation (n − 1 denominator); 0 when `n` < 2. */
    sd: number;
    /**
     * Values that were not finite (NaN or ±Infinity) and were skipped, the same
     * convention as `WilcoxonResult.nonFinite`.
     */
    nonFinite: number;
}

/**
 * Mean and SAMPLE standard deviation by Welford's online algorithm, which stays
 * accurate where the naive Σx² − (Σx)²/n form cancels catastrophically (values
 * with a large common offset).
 *
 * - Non-finite values (NaN, ±Infinity) are skipped and counted in `nonFinite`;
 *   `n` is the count of values used.
 * - `n` = 0: `mean` is NaN and `sd` is 0. One result shape is kept rather than the
 *   estimators' `InsufficientData` union, so callers branch on `n`.
 * - `n` = 1: `mean` is the value and `sd` is 0 (a sample sd needs two values).
 * - A z-score consumer must guard `n >= 2 && sd > 0`: sd is 0 (not NaN) for one
 *   value or all-equal data, so an unguarded division yields ±Infinity.
 *
 * Pure and deterministic: one pass in input order, no allocation beyond the result.
 */
export function meanAndSd(values: readonly number[]): MeanAndSd {
    let n = 0;
    let nonFinite = 0;
    let mean = 0;
    let m2 = 0;
    for (const x of values) {
        if (!Number.isFinite(x)) {
            nonFinite++;
            continue;
        }
        n++;
        const delta = x - mean;
        mean += delta / n;
        m2 += delta * (x - mean);
    }
    if (n === 0) return { n: 0, mean: Number.NaN, sd: 0, nonFinite };
    const sd = n < 2 ? 0 : Math.sqrt(Math.max(0, m2) / (n - 1));
    return { n, mean, sd, nonFinite };
}
