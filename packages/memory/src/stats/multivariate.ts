/**
 * Multivariate summaries for the stats module (PRD 41 Part D / D13): Ledoit–Wolf
 * shrinkage covariance and the population stability index (PSI). Pure and
 * deterministic.
 *
 * Ledoit–Wolf lives in memory-engine `ml/classical/covariance.ts` (the anomaly
 * composite's Mahalanobis term fits on it) and is re-exported here, never copied. It
 * comes through the engine's light `./ml/classical` subpath, NEVER the engine root:
 * browser bundles import `./stats`, and the root carries WebGPU kernels and training
 * (`src/contract/regression.statsEngineFree.test.ts` holds this).
 */

export { ledoitWolfCovariance } from "@seanhogg/builderforce-memory-engine/ml/classical";
export type { CovarianceResult } from "@seanhogg/builderforce-memory-engine/ml/classical";

/**
 * Population stability index, Σ (aᵢ − eᵢ) ln(aᵢ / eᵢ), over `bins` equal-width bins
 * spanning [min, max] of the pooled data. Zero-count bins contribute 0 (the
 * convention PS ≈ 0 uses). PSI above 0.2 signals drift (PRD 41 D12).
 */
export function psi(actual: readonly number[], expected: readonly number[], bins = 10): number {
    if (!actual.length || !expected.length) return 0;
    const lo = Math.min(...actual, ...expected);
    const hi = Math.max(...actual, ...expected);
    if (hi === lo) return 0;
    const width = (hi - lo) / bins;
    const count = (values: readonly number[]) => {
        const counts = new Array<number>(bins).fill(0);
        for (const v of values) {
            const idx = Math.min(bins - 1, Math.floor((v - lo) / width));
            counts[idx]!++;
        }
        return counts.map((c) => c / values.length);
    };
    const a = count(actual);
    const e = count(expected);
    let sum = 0;
    for (let i = 0; i < bins; i++) {
        if (a[i]! > 0 && e[i]! > 0) sum += (a[i]! - e[i]!) * Math.log(a[i]! / e[i]!);
    }
    return sum;
}
