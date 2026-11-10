/**
 * ml/classical/logistic.ts — (multinomial) logistic regression.
 *
 * Softmax regression with L2, sample weights and class weights, fitted by full-batch
 * gradient descent; two classes is ordinary logistic regression, and two classes may
 * instead be fitted by IRLS (Newton–Raphson on the same penalized objective, with
 * step-halving and a fixed iteration cap). No internal feature scaling: a task
 * supplies features on a stable scale, so a weight means the same thing on every
 * machine — which is what lets trained weights FEDERATE (`exportParams` /
 * `importParams`).
 */
import { mlEvalLoss } from "../eval_loss.js";
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { solveLinearSystem } from "./linear.js";
import { classCount, featureDim, sigmoid, softmax, topProbability } from "./math.js";

export interface LogisticOptions {
  classes?: number;
  learningRate?: number;
  epochs?: number;
  /** L2 penalty on weights (not biases). Default 1e-3. */
  l2?: number;
  /**
   * Per-class weight, index-aligned with the class label; multiplies each sample's own
   * `weight`. A class without an entry weighs 1. Absent = every class weighs 1.
   */
  classWeights?: readonly number[];
  /**
   * `gradient-descent` (default) runs `epochs` full-batch steps. `irls` (two classes only)
   * runs Newton–Raphson to the optimum of the same objective — mean weighted cross-entropy
   * plus the L2 penalty — in at most `maxIterations` steps.
   */
  solver?: "gradient-descent" | "irls";
  /** IRLS iteration cap. Default 50. */
  maxIterations?: number;
}

/** A sample's effective weight: its own weight times its class's weight. */
function effectiveWeight(s: MlSample, classWeights: readonly number[] | undefined): number {
  return (s.weight ?? 1) * (classWeights?.[s.y] ?? 1);
}

export class LogisticModel implements MlPredictor {
  /** Row-major weights [classes × dim]. */
  readonly weights: Float32Array;
  readonly bias: Float32Array;

  constructor(
    readonly dim: number,
    readonly classes: number,
  ) {
    this.weights = new Float32Array(classes * dim);
    this.bias = new Float32Array(classes);
  }

  logits(x: ArrayLike<number>): number[] {
    const z: number[] = [];
    for (let c = 0; c < this.classes; c++) {
      let s = this.bias[c]!;
      const off = c * this.dim;
      for (let i = 0; i < this.dim; i++) s += this.weights[off + i]! * (x[i] ?? 0);
      z.push(s);
    }
    return z;
  }

  proba(x: ArrayLike<number>): number[] {
    return Array.from(softmax(this.logits(x)));
  }

  predict(x: Float32Array): MlPrediction {
    const scores = this.proba(x);
    return { scores, confidence: topProbability(scores) };
  }

  /** Weighted mean cross-entropy over `samples` — exactly the eval-gate loss ({@link mlEvalLoss}); 0 for an empty set. */
  loss(samples: readonly MlSample[]): number {
    return mlEvalLoss("classify", this, samples) ?? 0;
  }

  /** One full-batch gradient step; returns the pre-step mean loss. */
  step(samples: readonly MlSample[], learningRate: number, l2: number, classWeights?: readonly number[]): number {
    const gW = new Float64Array(this.weights.length);
    const gB = new Float64Array(this.classes);
    let loss = 0;
    let wSum = 0;
    for (const s of samples) {
      const w = effectiveWeight(s, classWeights);
      const p = this.proba(s.x);
      loss += -w * Math.log(Math.max(p[s.y]!, 1e-12));
      wSum += w;
      for (let c = 0; c < this.classes; c++) {
        const g = w * (p[c]! - (c === s.y ? 1 : 0));
        gB[c] = gB[c]! + g;
        const off = c * this.dim;
        for (let i = 0; i < this.dim; i++) gW[off + i] = gW[off + i]! + g * s.x[i]!;
      }
    }
    const inv = wSum > 0 ? 1 / wSum : 0;
    for (let j = 0; j < this.weights.length; j++) {
      this.weights[j] = this.weights[j]! - learningRate * (gW[j]! * inv + l2 * this.weights[j]!);
    }
    for (let c = 0; c < this.classes; c++) this.bias[c] = this.bias[c]! - learningRate * gB[c]! * inv;
    return loss * inv;
  }

  exportParams(): Float32Array {
    const out = new Float32Array(this.weights.length + this.bias.length);
    out.set(this.weights, 0);
    out.set(this.bias, this.weights.length);
    return out;
  }

  importParams(params: ArrayLike<number>): void {
    if (params.length !== this.weights.length + this.bias.length) {
      throw new Error(`LogisticModel.importParams: expected ${this.weights.length + this.bias.length} values, got ${params.length}`);
    }
    for (let j = 0; j < this.weights.length; j++) this.weights[j] = params[j]!;
    for (let c = 0; c < this.bias.length; c++) this.bias[c] = params[this.weights.length + c]!;
  }
}

/**
 * Fit logistic regression. Pass `init` to continue from existing weights (a local
 * round on top of a merged federated head).
 */
export function fitLogistic(
  samples: readonly MlSample[],
  opts: LogisticOptions = {},
  init?: LogisticModel,
): LogisticModel {
  const d = featureDim(samples.map((s) => s.x));
  const k = opts.classes ?? init?.classes ?? classCount(samples.map((s) => s.y));
  // A label at or above `classes` has no row of weights: its probability would be
  // read as undefined and every loss and gradient would turn NaN, silently.
  const top = classCount(samples.map((s) => s.y));
  if (top > k && samples.some((s) => s.y >= k)) {
    throw new Error(`fitLogistic: labels go up to ${top - 1} but the model has ${k} classes`);
  }
  const model = init ?? new LogisticModel(d, k);
  if (model.dim !== d || model.classes !== k) throw new Error(`fitLogistic: init is ${model.dim}×${model.classes}, data is ${d}×${k}`);
  const l2 = opts.l2 ?? 1e-3;
  const badWeight = opts.classWeights?.findIndex((w) => !Number.isFinite(w) || w < 0) ?? -1;
  if (badWeight >= 0) {
    throw new Error(`fitLogistic: classWeights[${badWeight}] must be a finite number ≥ 0, got ${opts.classWeights![badWeight]}`);
  }
  if (opts.solver === "irls") {
    if (k !== 2) throw new Error(`fitLogistic: the irls solver fits two classes, not ${k}`);
    fitBinaryIrls(model, samples, l2, opts.classWeights, opts.maxIterations ?? 50);
    return model;
  }
  const lr = opts.learningRate ?? 0.5;
  const epochs = opts.epochs ?? 300;
  for (let e = 0; e < epochs; e++) model.step(samples, lr, l2, opts.classWeights);
  return model;
}

/** Diagonal jitter that keeps the Newton system solvable on separable data with l2 = 0. */
const IRLS_JITTER = 1e-9;
/** Converged when every coordinate of the full (unhalved) Newton step is below this. */
const IRLS_TOLERANCE = 1e-10;
/** Step-halvings tried before an iteration gives up and the fit stops. */
const IRLS_MAX_HALVINGS = 40;

/**
 * Binary IRLS: Newton–Raphson on `mean weighted cross-entropy + λ/2·‖w‖²` over
 * θ = [w, b] (the bias is not penalized), each step accepted only while the objective
 * does not rise (halved otherwise). Writes the result into `model` as a two-class
 * softmax whose class-0 row is zero, so `P(class 1) = σ(w·x + b)` exactly. Starts from
 * the model's current params, reduced to binary form.
 *
 * λ = l2 / 2: gradient descent penalizes BOTH softmax rows, and at its optimum they are
 * w₀ = −w/2, w₁ = w/2, so `l2/2·(‖w₀‖² + ‖w₁‖²) = (l2/2)/2·‖w‖²`. With this λ both
 * solvers minimise the same function and reach the same model.
 */
function fitBinaryIrls(
  model: LogisticModel,
  samples: readonly MlSample[],
  softmaxL2: number,
  classWeights: readonly number[] | undefined,
  maxIterations: number,
): void {
  const l2 = softmaxL2 / 2;
  const d = model.dim;
  const n = d + 1;
  const theta = new Float64Array(n);
  for (let i = 0; i < d; i++) theta[i] = model.weights[d + i]! - model.weights[i]!;
  theta[d] = model.bias[1]! - model.bias[0]!;
  const omega = samples.map((s) => effectiveWeight(s, classWeights));
  const wSum = omega.reduce((a, b) => a + b, 0);
  const inv = wSum > 0 ? 1 / wSum : 0;
  const eta = (t: Float64Array, x: ArrayLike<number>): number => {
    let z = t[d]!;
    for (let i = 0; i < d; i++) z += t[i]! * x[i]!;
    return z;
  };
  const objective = (t: Float64Array): number => {
    let loss = 0;
    samples.forEach((s, r) => {
      const z = eta(t, s.x);
      // log(1 + e^z) − y·z, numerically stable.
      const softplus = z >= 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
      loss += omega[r]! * (softplus - (s.y === 1 ? z : 0));
    });
    let penalty = 0;
    for (let i = 0; i < d; i++) penalty += t[i]! * t[i]!;
    return loss * inv + (l2 / 2) * penalty;
  };
  let current = objective(theta);
  for (let iter = 0; iter < maxIterations; iter++) {
    const g = new Float64Array(n);
    const H = new Float64Array(n * n);
    samples.forEach((s, r) => {
      const p = sigmoid(eta(theta, s.x));
      const gr = omega[r]! * (p - (s.y === 1 ? 1 : 0)) * inv;
      const hr = omega[r]! * p * (1 - p) * inv;
      for (let i = 0; i < n; i++) {
        const xi = i < d ? s.x[i]! : 1;
        g[i] = g[i]! + gr * xi;
        for (let j = 0; j < n; j++) {
          const xj = j < d ? s.x[j]! : 1;
          H[i * n + j] = H[i * n + j]! + hr * xi * xj;
        }
      }
    });
    for (let i = 0; i < n; i++) {
      if (i < d) {
        g[i] = g[i]! + l2 * theta[i]!;
        H[i * n + i] = H[i * n + i]! + l2;
      }
      H[i * n + i] = H[i * n + i]! + IRLS_JITTER;
    }
    const delta = solveLinearSystem(H, g, n);
    // Converged when the FULL Newton step is negligible (a halved step can be small far from the optimum).
    let largest = 0;
    for (let i = 0; i < n; i++) largest = Math.max(largest, Math.abs(delta[i]!));
    const converged = largest < IRLS_TOLERANCE;
    let scale = 1;
    let accepted = false;
    for (let h = 0; h < IRLS_MAX_HALVINGS; h++) {
      const candidate = new Float64Array(n);
      for (let i = 0; i < n; i++) candidate[i] = theta[i]! - scale * delta[i]!;
      const value = objective(candidate);
      if (value <= current + 1e-15) {
        theta.set(candidate);
        current = value;
        accepted = true;
        break;
      }
      scale /= 2;
    }
    if (!accepted || converged) break;
  }
  model.weights.fill(0);
  model.bias.fill(0);
  for (let i = 0; i < d; i++) model.weights[d + i] = theta[i]!;
  model.bias[1] = theta[d]!;
}
