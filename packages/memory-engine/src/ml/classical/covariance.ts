/**
 * ml/classical/covariance.ts — covariance estimation and inversion for the classical
 * catalog: Ledoit–Wolf shrinkage covariance (the ONE implementation; memory's `./stats`
 * re-exports it) and a symmetric positive-definite inverse by one Cholesky factorization.
 * Pure and deterministic.
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
 * Inverse of a symmetric positive-definite d×d matrix (row-major) by ONE Cholesky
 * factorization A = L·Lᵀ: A⁻¹ = L⁻ᵀ·L⁻¹. Null when A is not positive definite (a
 * non-positive pivot), so the caller decides what an uninvertible covariance means.
 */
export function choleskyInverse(a: ArrayLike<number>, d: number): Float64Array | null {
  const L = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = a[i * d + j]!;
      for (let k = 0; k < j; k++) sum -= L[i * d + k]! * L[j * d + k]!;
      if (i === j) {
        if (!(sum > 0)) return null;
        L[i * d + i] = Math.sqrt(sum);
      } else {
        L[i * d + j] = sum / L[j * d + j]!;
      }
    }
  }
  // Y = L⁻¹ (lower triangular), by forward substitution column by column.
  const Y = new Float64Array(d * d);
  for (let col = 0; col < d; col++) {
    for (let i = col; i < d; i++) {
      let sum = i === col ? 1 : 0;
      for (let k = col; k < i; k++) sum -= L[i * d + k]! * Y[k * d + col]!;
      Y[i * d + col] = sum / L[i * d + i]!;
    }
  }
  // A⁻¹ = Yᵀ·Y.
  const inv = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      let sum = 0;
      for (let k = j; k < d; k++) sum += Y[k * d + i]! * Y[k * d + j]!;
      inv[i * d + j] = sum;
      inv[j * d + i] = sum;
    }
  }
  return inv;
}
