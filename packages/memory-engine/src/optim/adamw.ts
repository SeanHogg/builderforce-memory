/**
 * adamw.ts — AdamW optimiser over a model's flat parameter list.
 *
 * Shared by every CPU-reference trainer in the engine (MoE FFN, the full
 * EvermindLM) so the optimiser maths lives in exactly one place. Operates on any
 * object exposing index-aligned `parameters()` / `gradients()` Float32Arrays.
 */

export interface OptimParam {
  data: Float32Array;
}

export interface OptimTarget {
  parameters(): OptimParam[];
  gradients(): OptimParam[];
}

export interface AdamWOptions {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
}

export class AdamW {
  private readonly m: Float32Array[] = [];
  private readonly v: Float32Array[] = [];
  private t = 0;
  private readonly opt: Required<AdamWOptions>;

  constructor(
    private readonly target: OptimTarget,
    options: AdamWOptions = {},
  ) {
    this.opt = {
      lr: options.lr ?? 0.01,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
    };
    for (const p of target.parameters()) {
      this.m.push(new Float32Array(p.data.length));
      this.v.push(new Float32Array(p.data.length));
    }
  }

  /** One optimiser step from the currently-accumulated gradients. */
  step(): void {
    this.t++;
    const { lr, beta1, beta2, eps, weightDecay } = this.opt;
    const params = this.target.parameters();
    const grads = this.target.gradients();
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);
    for (let p = 0; p < params.length; p++) {
      const w = params[p]!.data;
      const g = grads[p]!.data;
      const m = this.m[p]!;
      const v = this.v[p]!;
      for (let i = 0; i < w.length; i++) {
        const gi = g[i]!;
        m[i] = beta1 * m[i]! + (1 - beta1) * gi;
        v[i] = beta2 * v[i]! + (1 - beta2) * gi * gi;
        const mh = m[i]! / bc1;
        const vh = v[i]! / bc2;
        w[i] = w[i]! - lr * (mh / (Math.sqrt(vh) + eps) + weightDecay * w[i]!);
      }
    }
  }
}
