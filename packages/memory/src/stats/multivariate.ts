/**
 * Multivariate summaries for the stats module (PRD 41 Part D / D13): Ledoit–Wolf
 * shrinkage covariance and the population stability index (PSI). Pure and
 * deterministic.
 */

export interface CovarianceResult {
    /** Column means of the input matrix (rows = observations). */
    means: number[];
    covariance: number[][];
    /** Shrinkage intensity in [0, 1]. */
    shrinkage: number;
}

/**
 * Ledoit–Wolf shrunk covariance estimator. `data` is an n×d matrix (rows are
 * observations). The optimal shrinkage intensity follows the Ledoit–Wolf formula
 * toward the scaled-identity target (tr(S)/d)·I.
 */
export function ledoitWolfCovariance(data: readonly (readonly number[])[]): CovarianceResult {
    const n = data.length;
    const d = data[0]?.length ?? 0;
    if (n === 0 || d === 0) return { means: new Array(d).fill(0), covariance: [], shrinkage: 0 };
    const means = new Array<number>(d).fill(0);
    for (const row of data) for (let j = 0; j < d; j++) means[j]! += row[j]! / n;
    const center = data.map((row) => row.map((v, j) => v - means[j]!));
    const scale = 1 / n; // MLE denominator, matching the LW derivation
    const s: number[][] = [];
    for (let i = 0; i < d; i++) {
        s.push(new Array<number>(d).fill(0));
        for (let j = 0; j < d; j++) {
            let acc = 0;
            for (let r = 0; r < n; r++) acc += center[r]![i]! * center[r]![j]!;
            s[i]![j] = acc * scale;
        }
    }
    const trace = s.reduce((acc, row, i) => acc + row[i]!, 0);
    const mu = trace / d;
    // Sum of asymptotic variances of the off-diagonal (and diagonal) entries.
    let deltaSum = 0;
    for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
            let acc = 0;
            for (let r = 0; r < n; r++) {
                acc += (center[r]![i]! * center[r]![j]! - (i === j ? s[i]![i]! : s[i]![j]!)) ** 2;
            }
            deltaSum += acc / n;
        }
    }
    deltaSum *= scale * scale;
    // Squared distance from S to the target.
    let targetDist = 0;
    for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
            const target = i === j ? mu : 0;
            targetDist += (s[i]![j]! - target) ** 2;
        }
    }
    const beta = Math.min(deltaSum, targetDist);
    const shrinkage = targetDist > 0 ? beta / targetDist : 0;
    const covariance = s.map((row, i) =>
        row.map((v, j) => (1 - shrinkage) * v + (i === j ? shrinkage * mu : 0)),
    );
    return { means, covariance, shrinkage };
}

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
