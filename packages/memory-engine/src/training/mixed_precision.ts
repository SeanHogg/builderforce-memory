/**
 * mixed_precision.ts — mixed-precision training discipline (AMP) for the CPU
 * reference, faithful to the cookbook's recipe:
 *
 *   • fp16 compute, fp32 MASTER weights. The model's Float32Array params ARE the
 *     master copy; the forward runs on an fp16-rounded view, so tiny weight
 *     updates still accumulate in fp32 instead of vanishing.
 *   • Dynamic LOSS SCALING. fp16 gradients underflow to zero below ~6e-5; scaling
 *     the loss by a large factor S lifts them into fp16 range, then we unscale by
 *     S before the optimiser step. On overflow (inf/nan) we skip the step and back
 *     the scale off; after a clean streak we grow it. This is the mechanism that
 *     makes fp16 training numerically stable.
 *
 * The payoff is memory/throughput: activations and gradients at half precision.
 * On CPU we can't reclaim the bytes, so this module models the NUMERICS exactly
 * (fp16 rounding + loss scaling) — the property the WGSL path will inherit — and
 * its tests prove a small gradient survives WITH scaling and underflows WITHOUT.
 */

import { floatToFp16, fp16ToFloat } from "../utils/quantization.js";

/** Round a value to what fp16 storage would preserve (half-precision simulation). */
export function roundFp16(x: number): number {
  return fp16ToFloat(floatToFp16(x));
}

/** Round every element of a copy to fp16 precision (leaves the input untouched). */
export function fp16View(a: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = roundFp16(a[i]!);
  return out;
}

export interface LossScalerOptions {
  /** Initial scale. Default 2^14 = 16384. */
  initScale?: number;
  /** Multiply the scale by this after `growthInterval` clean steps. Default 2. */
  growthFactor?: number;
  /** Multiply the scale by this on overflow. Default 0.5. */
  backoffFactor?: number;
  /** Clean steps between growth attempts. Default 200. */
  growthInterval?: number;
  /** Floor for the scale so it can't collapse to zero. Default 1. */
  minScale?: number;
}

/**
 * Dynamic loss scaler — the standard AMP controller. Scale the loss/grads up
 * before backward, {@link check} them for overflow, {@link unscale} the good ones,
 * and {@link update} the scale based on whether this step overflowed.
 */
export class DynamicLossScaler {
  private _scale: number;
  private readonly growthFactor: number;
  private readonly backoffFactor: number;
  private readonly growthInterval: number;
  private readonly minScale: number;
  private cleanStreak = 0;
  private _overflows = 0;
  private _skipped = 0;

  constructor(opts: LossScalerOptions = {}) {
    this._scale = opts.initScale ?? 16384;
    this.growthFactor = opts.growthFactor ?? 2;
    this.backoffFactor = opts.backoffFactor ?? 0.5;
    this.growthInterval = opts.growthInterval ?? 200;
    this.minScale = opts.minScale ?? 1;
  }

  get scale(): number {
    return this._scale;
  }
  get overflows(): number {
    return this._overflows;
  }
  get skipped(): number {
    return this._skipped;
  }

  /** True if any gradient is inf/nan (an fp16 overflow) — such a step must be skipped. */
  check(grads: { data: Float32Array }[]): boolean {
    for (const g of grads) {
      const a = g.data;
      for (let i = 0; i < a.length; i++) {
        const x = a[i]!;
        if (!Number.isFinite(x)) return true;
      }
    }
    return false;
  }

  /** Divide gradients back down by the current scale, in place. */
  unscale(grads: { data: Float32Array }[]): void {
    const inv = 1 / this._scale;
    for (const g of grads) {
      const a = g.data;
      for (let i = 0; i < a.length; i++) a[i] = a[i]! * inv;
    }
  }

  /**
   * Advance the scale controller. Pass whether this step overflowed. Returns
   * `true` if the optimiser step should PROCEED, `false` if it must be skipped.
   */
  update(overflow: boolean): boolean {
    if (overflow) {
      this._overflows++;
      this._skipped++;
      this._scale = Math.max(this.minScale, this._scale * this.backoffFactor);
      this.cleanStreak = 0;
      return false;
    }
    if (++this.cleanStreak >= this.growthInterval) {
      this._scale *= this.growthFactor;
      this.cleanStreak = 0;
    }
    return true;
  }
}
