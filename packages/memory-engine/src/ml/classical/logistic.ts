/**
 * ml/classical/logistic.ts — (multinomial) logistic regression.
 *
 * Softmax regression with L2, fitted by full-batch gradient descent; two classes
 * is ordinary logistic regression. No internal feature scaling: a task supplies
 * features on a stable scale, so a weight means the same thing on every machine —
 * which is what lets trained weights FEDERATE (`exportParams` / `importParams`).
 */
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { classCount, featureDim, softmax, topProbability } from "./math.js";

export interface LogisticOptions {
  classes?: number;
  learningRate?: number;
  epochs?: number;
  /** L2 penalty on weights (not biases). Default 1e-3. */
  l2?: number;
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

  /** Weighted mean cross-entropy over `samples` (the eval-gate loss). */
  loss(samples: readonly MlSample[]): number {
    let sum = 0;
    let wSum = 0;
    for (const s of samples) {
      const w = s.weight ?? 1;
      sum += -w * Math.log(Math.max(this.proba(s.x)[s.y] ?? 0, 1e-12));
      wSum += w;
    }
    return wSum > 0 ? sum / wSum : 0;
  }

  /** One full-batch gradient step; returns the pre-step mean loss. */
  step(samples: readonly MlSample[], learningRate: number, l2: number): number {
    const gW = new Float64Array(this.weights.length);
    const gB = new Float64Array(this.classes);
    let loss = 0;
    let wSum = 0;
    for (const s of samples) {
      const w = s.weight ?? 1;
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
  const model = init ?? new LogisticModel(d, k);
  if (model.dim !== d || model.classes !== k) throw new Error(`fitLogistic: init is ${model.dim}×${model.classes}, data is ${d}×${k}`);
  const lr = opts.learningRate ?? 0.5;
  const l2 = opts.l2 ?? 1e-3;
  const epochs = opts.epochs ?? 300;
  for (let e = 0; e < epochs; e++) model.step(samples, lr, l2);
  return model;
}
