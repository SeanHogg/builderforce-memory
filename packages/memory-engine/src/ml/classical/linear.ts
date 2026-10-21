/**
 * ml/classical/linear.ts — ridge linear regression, closed form.
 *
 * Solves (XᵀWX + λI)w = XᵀWy with an unpenalised intercept by Gaussian
 * elimination with partial pivoting. Small dims only (the catalog's numeric
 * tasks), which is exactly where a closed form beats iterating.
 */
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { featureDim } from "./math.js";

export interface LinearOptions {
  /** L2 penalty on weights (not the intercept). Default 1e-6. */
  l2?: number;
}

export class LinearModel implements MlPredictor {
  constructor(
    readonly weights: Float32Array,
    public intercept: number,
    /** Coefficient of determination on the fitted data, clamped to [0, 1]. */
    readonly r2: number,
  ) {}

  value(x: ArrayLike<number>): number {
    let s = this.intercept;
    for (let i = 0; i < this.weights.length; i++) s += this.weights[i]! * (x[i] ?? 0);
    return s;
  }

  predict(x: Float32Array): MlPrediction {
    return { scores: [this.value(x)], confidence: this.r2 };
  }

  exportParams(): Float32Array {
    const out = new Float32Array(this.weights.length + 1);
    out.set(this.weights, 0);
    out[this.weights.length] = this.intercept;
    return out;
  }

  importParams(params: ArrayLike<number>): void {
    if (params.length !== this.weights.length + 1) {
      throw new Error(`LinearModel.importParams: expected ${this.weights.length + 1} values, got ${params.length}`);
    }
    for (let i = 0; i < this.weights.length; i++) this.weights[i] = params[i]!;
    this.intercept = params[this.weights.length]!;
  }
}

/** Solve A·x = b in place (A is n×n row-major). Throws when singular. */
export function solveLinearSystem(A: Float64Array, b: Float64Array, n: number): Float64Array {
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r * n + col]!) > Math.abs(A[piv * n + col]!)) piv = r;
    if (Math.abs(A[piv * n + col]!) < 1e-12) throw new Error("solveLinearSystem: singular matrix");
    if (piv !== col) {
      for (let c = 0; c < n; c++) {
        const t = A[col * n + c]!;
        A[col * n + c] = A[piv * n + c]!;
        A[piv * n + c] = t;
      }
      const t = b[col]!;
      b[col] = b[piv]!;
      b[piv] = t;
    }
    for (let r = col + 1; r < n; r++) {
      const f = A[r * n + col]! / A[col * n + col]!;
      if (f === 0) continue;
      for (let c = col; c < n; c++) A[r * n + c] = A[r * n + c]! - f * A[col * n + c]!;
      b[r] = b[r]! - f * b[col]!;
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r]!;
    for (let c = r + 1; c < n; c++) s -= A[r * n + c]! * x[c]!;
    x[r] = s / A[r * n + r]!;
  }
  return x;
}

export function fitLinear(samples: readonly MlSample[], opts: LinearOptions = {}): LinearModel {
  const d = featureDim(samples.map((s) => s.x));
  const n = d + 1; // last column = intercept
  const A = new Float64Array(n * n);
  const b = new Float64Array(n);
  const row = new Float64Array(n);
  for (const s of samples) {
    const w = s.weight ?? 1;
    for (let i = 0; i < d; i++) row[i] = s.x[i]!;
    row[d] = 1;
    for (let i = 0; i < n; i++) {
      b[i] = b[i]! + w * row[i]! * s.y;
      for (let j = 0; j < n; j++) A[i * n + j] = A[i * n + j]! + w * row[i]! * row[j]!;
    }
  }
  const l2 = opts.l2 ?? 1e-6;
  for (let i = 0; i < d; i++) A[i * n + i] = A[i * n + i]! + l2;
  const sol = solveLinearSystem(A, b, n);
  const model = new LinearModel(Float32Array.from(sol.subarray(0, d)), sol[d]!, 0);

  // R² on the fitted data — the confidence the runner reports.
  let wSum = 0;
  let mean = 0;
  for (const s of samples) {
    wSum += s.weight ?? 1;
    mean += (s.weight ?? 1) * s.y;
  }
  mean /= wSum;
  let ssRes = 0;
  let ssTot = 0;
  for (const s of samples) {
    const w = s.weight ?? 1;
    ssRes += w * (s.y - model.value(s.x)) ** 2;
    ssTot += w * (s.y - mean) ** 2;
  }
  const r2 = ssTot > 0 ? Math.min(1, Math.max(0, 1 - ssRes / ssTot)) : ssRes < 1e-12 ? 1 : 0;
  return new LinearModel(model.weights, model.intercept, r2);
}
