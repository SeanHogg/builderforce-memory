/**
 * ml/dense_head.ts — the catalog's trainable head: linear, or one tanh hidden layer,
 * over a feature vector (usually an experience embedding).
 *
 * It is a {@link TrainableHead} (so the ONE {@link HeadTrainer} fits it, on the
 * GPU when there is one) and an {@link MlPredictor} (so the runner answers with
 * it). Its parameters flatten to one vector — the unit that federates.
 */
import { gaussianArray, SeededRng } from "../utils/rng.js";
import { softmax } from "../training/cpu_ops.js";
import { assignParams, flattenParams, type HeadParam, type TrainableHead } from "./head_trainer.js";
import type { MlPrediction, MlPredictor, MlSample } from "./types.js";

export interface DenseHeadConfig {
  inputDim: number;
  /** Hidden width; 0 = a linear head. */
  hiddenDim: number;
  /** Classes (softmax) or 1 (mse regression). */
  outputDim: number;
  /** `softmax` = classification over `outputDim` classes; `mse` = scalar regression. */
  objective: "softmax" | "mse";
  seed?: number;
}

export class DenseHead implements TrainableHead<MlSample>, MlPredictor {
  readonly config: Required<DenseHeadConfig>;
  private readonly _params: HeadParam[];
  private readonly _grads: HeadParam[];

  constructor(config: DenseHeadConfig) {
    this.config = { seed: 0x4e4d, ...config };
    const { inputDim, hiddenDim, outputDim, objective } = this.config;
    if (objective === "mse" && outputDim !== 1) throw new Error("DenseHead: mse objective needs outputDim 1");
    if (objective === "softmax" && outputDim < 2) throw new Error("DenseHead: softmax objective needs outputDim ≥ 2");
    const rng = new SeededRng(this.config.seed);
    const next = () => rng.next();
    const inner = hiddenDim > 0 ? hiddenDim : inputDim;
    const shapes: [string, number, number][] = [
      ...(hiddenDim > 0
        ? ([
            ["w1", hiddenDim * inputDim, 1 / Math.sqrt(inputDim)],
            ["b1", hiddenDim, 0],
          ] as [string, number, number][])
        : []),
      ["w2", outputDim * inner, 1 / Math.sqrt(inner)],
      ["b2", outputDim, 0],
    ];
    this._params = shapes.map(([name, n, std]) => ({ name, numel: n, data: std > 0 ? gaussianArray(n, std, next) : new Float32Array(n) }));
    this._grads = shapes.map(([name, n]) => ({ name, numel: n, data: new Float32Array(n) }));
  }

  private _p(name: string): Float32Array {
    return this._params.find((p) => p.name === name)!.data;
  }

  private _g(name: string): Float32Array {
    return this._grads.find((p) => p.name === name)!.data;
  }

  parameters(): HeadParam[] {
    return this._params;
  }

  gradients(): HeadParam[] {
    return this._grads;
  }

  zeroGrad(): void {
    for (const g of this._grads) g.data.fill(0);
  }

  /** Hidden activation (or the input itself for a linear head) and the output logits/value. */
  private _forward(x: ArrayLike<number>): { h: Float32Array; z: Float32Array } {
    const { inputDim, hiddenDim, outputDim } = this.config;
    let h: Float32Array;
    if (hiddenDim > 0) {
      const w1 = this._p("w1");
      const b1 = this._p("b1");
      h = new Float32Array(hiddenDim);
      for (let j = 0; j < hiddenDim; j++) {
        let s = b1[j]!;
        for (let i = 0; i < inputDim; i++) s += w1[j * inputDim + i]! * (x[i] ?? 0);
        h[j] = Math.tanh(s);
      }
    } else {
      h = Float32Array.from({ length: inputDim }, (_, i) => x[i] ?? 0);
    }
    const w2 = this._p("w2");
    const b2 = this._p("b2");
    const z = new Float32Array(outputDim);
    for (let k = 0; k < outputDim; k++) {
      let s = b2[k]!;
      for (let j = 0; j < h.length; j++) s += w2[k * h.length + j]! * h[j]!;
      z[k] = s;
    }
    return { h, z };
  }

  /** Loss and dLoss/dz of one sample (unweighted). */
  private _lossGrad(z: Float32Array, y: number): { loss: number; dz: Float32Array } {
    if (this.config.objective === "mse") {
      const d = z[0]! - y;
      return { loss: 0.5 * d * d, dz: Float32Array.of(d) };
    }
    const p = softmax(z);
    const dz = Float32Array.from(p, (v, k) => v - (k === y ? 1 : 0));
    return { loss: -Math.log(Math.max(p[y] ?? 0, 1e-12)), dz };
  }

  loss(s: MlSample): number {
    return (s.weight ?? 1) * this._lossGrad(this._forward(s.x).z, s.y).loss;
  }

  accumulate(s: MlSample): number {
    const w = s.weight ?? 1;
    const { inputDim, hiddenDim, outputDim } = this.config;
    const { h, z } = this._forward(s.x);
    const { loss, dz } = this._lossGrad(z, s.y);
    const w2 = this._p("w2");
    const gW2 = this._g("w2");
    const gB2 = this._g("b2");
    const dh = new Float32Array(h.length);
    for (let k = 0; k < outputDim; k++) {
      const g = w * dz[k]!;
      gB2[k] = gB2[k]! + g;
      for (let j = 0; j < h.length; j++) {
        gW2[k * h.length + j] = gW2[k * h.length + j]! + g * h[j]!;
        dh[j] = dh[j]! + g * w2[k * h.length + j]!;
      }
    }
    if (hiddenDim > 0) {
      const gW1 = this._g("w1");
      const gB1 = this._g("b1");
      for (let j = 0; j < hiddenDim; j++) {
        const dpre = dh[j]! * (1 - h[j]! * h[j]!);
        gB1[j] = gB1[j]! + dpre;
        for (let i = 0; i < inputDim; i++) gW1[j * inputDim + i] = gW1[j * inputDim + i]! + dpre * (s.x[i] ?? 0);
      }
    }
    return w * loss;
  }

  predict(x: Float32Array): MlPrediction {
    const { z } = this._forward(x);
    if (this.config.objective === "mse") return { scores: [z[0]!], confidence: 1 };
    const scores = Array.from(softmax(z));
    return { scores, confidence: Math.max(...scores) };
  }

  exportParams(): Float32Array {
    return flattenParams(this._params);
  }

  importParams(params: ArrayLike<number>): void {
    assignParams(this._params, params);
  }
}
