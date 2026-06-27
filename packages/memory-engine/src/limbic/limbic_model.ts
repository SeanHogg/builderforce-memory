/**
 * limbic_model.ts – LimbicModel: a small, trainable recurrent affect head.
 *
 * The limbic model learns the *dynamics* of an agent's affective/motivational
 * state. Given an experience embedding (produced upstream by the hippocampus
 * SSM) and the current affective state, it predicts:
 *   • a bounded affect *delta* (how valence/arousal/drives/attention/exploration
 *     should move in response to this experience), and
 *   • a scalar *reward* prediction (how good/bad this experience was).
 *
 * Architecture (intentionally tiny — the heavy representation work happens in
 * the hippocampus; this head just maps representation → affect):
 *
 *   pre[j]  = Σ_i Win[j,i]·x[i] + Σ_k Ws[j,k]·s[k]        (hidden pre-activation)
 *   a[j]    = sigmoid(A[j])                               (per-channel SSM gate)
 *   h'[j]   = a[j]·h[j] + (1-a[j])·tanh(pre[j])           (recurrent leak/input)
 *   Δ[k]    = tanh( Σ_j Wout[k,j]·h'[j] + b[k] )          (bounded affect delta)
 *   r       = Σ_j Wr[j]·h'[j] + br                        (reward prediction)
 *
 * It runs on WebGPU (per-turn `step()` via {@link LIMBIC_AFFECT_WGSL}, and the
 * AdamW optimiser via the shared WEIGHT_UPDATE_WGSL kernel during training) with
 * a numerically-identical pure-CPU reference path used when no GPUDevice is
 * available — the same WebGPU-or-fallback contract the rest of the engine uses.
 *
 * Training uses truncated BPTT(1): the recurrent state is carried forward across
 * a sequence, but each step's gradient treats the incoming hidden state as a
 * constant. For an affect head this is stable and sufficient — each
 * (experience, state) → affect pair is close to an independent regression.
 */

import { SeededRng } from "../utils/rng.js";
import { quantizeFp16, dequantizeFp16 } from "../utils/quantization.js";
import { LIMBIC_STATE_DIM } from "./regions.js";

export interface LimbicModelConfig {
  /** Experience-embedding dimension (input). Default 32. */
  inputDim: number;
  /** Hidden recurrent width. Must be ≤ 64 (the GPU kernel's workgroup size). Default 16. */
  hiddenDim: number;
  /** Affective state dimension. Default {@link LIMBIC_STATE_DIM} (8). */
  stateDim: number;
  /** Deterministic init seed for reproducible cold-start weights. */
  seed?: number;
  /** Weight of the reward-prediction term in the training loss. Default 0.5. */
  rewardWeight?: number;
}

export const DEFAULT_LIMBIC_CONFIG: Required<Omit<LimbicModelConfig, "seed">> = {
  inputDim: 32,
  hiddenDim: 16,
  stateDim: LIMBIC_STATE_DIM,
  rewardWeight: 0.5,
};

/** Result of a single forward step. */
export interface LimbicForward {
  /** Next hidden recurrent state (length hiddenDim). */
  hidden: Float32Array;
  /** Bounded affect delta in (-1, 1) per state dim (length stateDim). */
  delta: Float32Array;
  /** Reward prediction (scalar). */
  reward: number;
}

/** Per-step forward intermediates retained for the backward pass. */
interface ForwardCache {
  x: Float32Array;
  sPrev: Float32Array;
  hPrev: Float32Array;
  a: Float32Array; // sigmoid(A)
  t: Float32Array; // tanh(pre)
  hn: Float32Array; // hidden
  delta: Float32Array; // tanh(preDelta)
  reward: number;
}

/** A named trainable parameter tensor (flat row-major Float32Array). */
export interface LimbicParam {
  name: string;
  data: Float32Array;
  numel: number;
}

const MAGIC = 0x4c4d4243; // "LMBC"

/** Fixed default init seed — reproducible byte-identical cold start across machines. */
export const DEFAULT_LIMBIC_SEED = 0x11b1c5ee;

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export class LimbicModel {
  readonly config: Required<Omit<LimbicModelConfig, "seed">>;

  // Parameters (flat, row-major).
  win: Float32Array; // hidden × input
  ws: Float32Array; // hidden × state
  aLogit: Float32Array; // hidden
  woutState: Float32Array; // state × hidden
  boutState: Float32Array; // state
  woutReward: Float32Array; // hidden
  boutReward: Float32Array; // 1

  // Gradient accumulators (same shapes), allocated lazily.
  private gWin: Float32Array;
  private gWs: Float32Array;
  private gALogit: Float32Array;
  private gWoutState: Float32Array;
  private gBoutState: Float32Array;
  private gWoutReward: Float32Array;
  private gBoutReward: Float32Array;

  constructor(config: Partial<LimbicModelConfig> = {}) {
    const cfg = { ...DEFAULT_LIMBIC_CONFIG, ...config };
    if (cfg.hiddenDim > 64) {
      throw new Error(`LimbicModel hiddenDim must be ≤ 64 (got ${cfg.hiddenDim})`);
    }
    this.config = cfg;
    const { inputDim, hiddenDim, stateDim } = cfg;

    const rng = new SeededRng(((config.seed ?? DEFAULT_LIMBIC_SEED) >>> 0) || 1);
    const randn = (std: number): number => {
      const u1 = Math.max(rng.next(), 1e-12);
      const u2 = rng.next();
      return std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
    const gauss = (n: number, std: number): Float32Array => {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = randn(std);
      return a;
    };

    // Small init — affect deltas should start near zero so an untrained model is
    // inert (no spurious mood swings) until it has learned from experience.
    this.win = gauss(hiddenDim * inputDim, 0.1);
    this.ws = gauss(hiddenDim * stateDim, 0.1);
    this.aLogit = gauss(hiddenDim, 0.05); // sigmoid(~0) ≈ 0.5 leak
    this.woutState = gauss(stateDim * hiddenDim, 0.05);
    this.boutState = new Float32Array(stateDim);
    this.woutReward = gauss(hiddenDim, 0.05);
    this.boutReward = new Float32Array(1);

    this.gWin = new Float32Array(this.win.length);
    this.gWs = new Float32Array(this.ws.length);
    this.gALogit = new Float32Array(this.aLogit.length);
    this.gWoutState = new Float32Array(this.woutState.length);
    this.gBoutState = new Float32Array(this.boutState.length);
    this.gWoutReward = new Float32Array(this.woutReward.length);
    this.gBoutReward = new Float32Array(1);
  }

  /** Trainable parameters, in the canonical checkpoint order. */
  parameters(): LimbicParam[] {
    return [
      { name: "win", data: this.win, numel: this.win.length },
      { name: "ws", data: this.ws, numel: this.ws.length },
      { name: "aLogit", data: this.aLogit, numel: this.aLogit.length },
      { name: "woutState", data: this.woutState, numel: this.woutState.length },
      { name: "boutState", data: this.boutState, numel: this.boutState.length },
      { name: "woutReward", data: this.woutReward, numel: this.woutReward.length },
      { name: "boutReward", data: this.boutReward, numel: this.boutReward.length },
    ];
  }

  /** Gradient buffers, index-aligned with {@link parameters}. */
  gradients(): LimbicParam[] {
    return [
      { name: "win", data: this.gWin, numel: this.gWin.length },
      { name: "ws", data: this.gWs, numel: this.gWs.length },
      { name: "aLogit", data: this.gALogit, numel: this.gALogit.length },
      { name: "woutState", data: this.gWoutState, numel: this.gWoutState.length },
      { name: "boutState", data: this.gBoutState, numel: this.gBoutState.length },
      { name: "woutReward", data: this.gWoutReward, numel: this.gWoutReward.length },
      { name: "boutReward", data: this.gBoutReward, numel: this.gBoutReward.length },
    ];
  }

  zeroGrad(): void {
    this.gWin.fill(0);
    this.gWs.fill(0);
    this.gALogit.fill(0);
    this.gWoutState.fill(0);
    this.gBoutState.fill(0);
    this.gWoutReward.fill(0);
    this.gBoutReward.fill(0);
  }

  /** A fresh zeroed hidden state. */
  initHidden(): Float32Array {
    return new Float32Array(this.config.hiddenDim);
  }

  /**
   * One forward step (CPU reference). Pure — does not mutate the model or the
   * inputs. The GPU kernel path produces numerically-identical results.
   */
  forward(x: ArrayLike<number>, hPrev: ArrayLike<number>, sPrev: ArrayLike<number>): LimbicForward {
    const cache = this._forwardCached(x, hPrev, sPrev);
    return { hidden: cache.hn, delta: cache.delta, reward: cache.reward };
  }

  private _forwardCached(
    x: ArrayLike<number>,
    hPrev: ArrayLike<number>,
    sPrev: ArrayLike<number>,
  ): ForwardCache {
    const { inputDim, hiddenDim, stateDim } = this.config;
    const xa = Float32Array.from({ length: inputDim }, (_, i) => x[i] ?? 0);
    const ha = Float32Array.from({ length: hiddenDim }, (_, j) => hPrev[j] ?? 0);
    const sa = Float32Array.from({ length: stateDim }, (_, k) => sPrev[k] ?? 0);

    const a = new Float32Array(hiddenDim);
    const t = new Float32Array(hiddenDim);
    const hn = new Float32Array(hiddenDim);
    for (let j = 0; j < hiddenDim; j++) {
      let pre = 0;
      const wiOff = j * inputDim;
      for (let i = 0; i < inputDim; i++) pre += this.win[wiOff + i]! * xa[i]!;
      const wsOff = j * stateDim;
      for (let k = 0; k < stateDim; k++) pre += this.ws[wsOff + k]! * sa[k]!;
      a[j] = sigmoid(this.aLogit[j]!);
      t[j] = Math.tanh(pre);
      hn[j] = a[j]! * ha[j]! + (1 - a[j]!) * t[j]!;
    }

    const delta = new Float32Array(stateDim);
    for (let k = 0; k < stateDim; k++) {
      let acc = this.boutState[k]!;
      const off = k * hiddenDim;
      for (let j = 0; j < hiddenDim; j++) acc += this.woutState[off + j]! * hn[j]!;
      delta[k] = Math.tanh(acc);
    }

    let reward = this.boutReward[0]!;
    for (let j = 0; j < hiddenDim; j++) reward += this.woutReward[j]! * hn[j]!;

    return { x: xa, sPrev: sa, hPrev: ha, a, t, hn, delta, reward };
  }

  /**
   * Accumulate gradients for one (input, state) → (deltaTarget, rewardTarget)
   * sample using truncated BPTT(1). Returns the scalar loss for this step and
   * the next hidden state to carry forward. Call {@link zeroGrad} before a batch
   * and apply the optimiser after.
   */
  backwardStep(
    x: ArrayLike<number>,
    hPrev: ArrayLike<number>,
    sPrev: ArrayLike<number>,
    deltaTarget: ArrayLike<number>,
    rewardTarget: number,
  ): { loss: number; hidden: Float32Array } {
    const { inputDim, hiddenDim, stateDim, rewardWeight } = this.config;
    const c = this._forwardCached(x, hPrev, sPrev);

    // Loss
    let loss = 0;
    const dDelta = new Float32Array(stateDim);
    for (let k = 0; k < stateDim; k++) {
      const diff = c.delta[k]! - (deltaTarget[k] ?? 0);
      loss += 0.5 * diff * diff;
      dDelta[k] = diff;
    }
    const dRewardDiff = c.reward - rewardTarget;
    loss += 0.5 * rewardWeight * dRewardDiff * dRewardDiff;
    const dReward = rewardWeight * dRewardDiff;

    // Backprop to hidden through both heads.
    const dHn = new Float32Array(hiddenDim);
    for (let k = 0; k < stateDim; k++) {
      const dPreDelta = dDelta[k]! * (1 - c.delta[k]! * c.delta[k]!); // tanh'
      this.gBoutState[k] = this.gBoutState[k]! + dPreDelta;
      const off = k * hiddenDim;
      for (let j = 0; j < hiddenDim; j++) {
        this.gWoutState[off + j] = this.gWoutState[off + j]! + dPreDelta * c.hn[j]!;
        dHn[j] = dHn[j]! + dPreDelta * this.woutState[off + j]!;
      }
    }
    this.gBoutReward[0] = this.gBoutReward[0]! + dReward;
    for (let j = 0; j < hiddenDim; j++) {
      this.gWoutReward[j] = this.gWoutReward[j]! + dReward * c.hn[j]!;
      dHn[j] = dHn[j]! + dReward * this.woutReward[j]!;
    }

    // Backprop through the recurrent update hn = a·hPrev + (1-a)·t.
    for (let j = 0; j < hiddenDim; j++) {
      const aj = c.a[j]!;
      const tj = c.t[j]!;
      // ∂hn/∂A = (hPrev - t)·a·(1-a)
      this.gALogit[j] = this.gALogit[j]! + dHn[j]! * (c.hPrev[j]! - tj) * aj * (1 - aj);
      // ∂hn/∂t = (1-a);  ∂t/∂pre = 1 - t²
      const dPre = dHn[j]! * (1 - aj) * (1 - tj * tj);
      const wiOff = j * inputDim;
      for (let i = 0; i < inputDim; i++) this.gWin[wiOff + i] = this.gWin[wiOff + i]! + dPre * c.x[i]!;
      const wsOff = j * stateDim;
      for (let k = 0; k < stateDim; k++) this.gWs[wsOff + k] = this.gWs[wsOff + k]! + dPre * c.sPrev[k]!;
    }

    return { loss, hidden: c.hn };
  }

  // ── Checkpoint serialisation ──────────────────────────────────────────────

  /**
   * Serialise weights to a compact "LMBC" binary. fp16 (v2) halves the size at
   * ~0.5% precision cost; f32 (v1) is exact. Layout: magic, version, [inputDim,
   * hiddenDim, stateDim], then params in {@link parameters} order.
   */
  exportWeights(opts: { fp16?: boolean } = {}): ArrayBuffer {
    const fp16 = opts.fp16 ?? false;
    const params = this.parameters();
    const total = params.reduce((n, p) => n + p.numel, 0);
    const headerEls = 5; // magic, version, inputDim, hiddenDim, stateDim
    const headerBytes = headerEls * 4;
    const dataBytes = fp16 ? total * 2 : total * 4;
    const buf = new ArrayBuffer(headerBytes + dataBytes);
    const head = new Uint32Array(buf, 0, headerEls);
    head[0] = MAGIC;
    head[1] = fp16 ? 2 : 1;
    head[2] = this.config.inputDim;
    head[3] = this.config.hiddenDim;
    head[4] = this.config.stateDim;

    if (fp16) {
      const flat = new Float32Array(total);
      let o = 0;
      for (const p of params) {
        flat.set(p.data, o);
        o += p.numel;
      }
      const q = quantizeFp16(flat); // Uint16Array
      new Uint16Array(buf, headerBytes, total).set(q);
    } else {
      const out = new Float32Array(buf, headerBytes, total);
      let o = 0;
      for (const p of params) {
        out.set(p.data, o);
        o += p.numel;
      }
    }
    return buf;
  }

  /** Load weights from an "LMBC" binary. Validates magic + dims. */
  loadWeights(buffer: ArrayBuffer): void {
    const head = new Uint32Array(buffer, 0, 5);
    if (head[0] !== MAGIC) throw new Error("LimbicModel.loadWeights: bad magic (not an LMBC checkpoint)");
    const version = head[1]!;
    const inputDim = head[2]!;
    const hiddenDim = head[3]!;
    const stateDim = head[4]!;
    if (inputDim !== this.config.inputDim || hiddenDim !== this.config.hiddenDim || stateDim !== this.config.stateDim) {
      throw new Error(
        `LimbicModel.loadWeights: dim mismatch — checkpoint ${inputDim}/${hiddenDim}/${stateDim} vs model ${this.config.inputDim}/${this.config.hiddenDim}/${this.config.stateDim}`,
      );
    }
    const params = this.parameters();
    const total = params.reduce((n, p) => n + p.numel, 0);
    const headerBytes = 20;
    let flat: Float32Array;
    if (version === 2) {
      flat = dequantizeFp16(new Uint16Array(buffer, headerBytes, total));
    } else {
      flat = new Float32Array(buffer.slice(headerBytes, headerBytes + total * 4));
    }
    let o = 0;
    for (const p of params) {
      p.data.set(flat.subarray(o, o + p.numel));
      o += p.numel;
    }
  }
}
