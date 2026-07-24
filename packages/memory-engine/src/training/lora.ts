/**
 * lora.ts — Low-Rank Adaptation (LoRA) + QLoRA for the EvermindLM CPU reference.
 *
 * The Meta llama-cookbook's central PEFT idea: freeze the base weights and train
 * a tiny low-rank delta ΔW = (α/r)·B·A on top. Three payoffs, all realised here:
 *
 *   • Cheap — you train r·(rows+cols) params instead of rows·cols. For the tied
 *     embedding (vocab×dModel) at rank 8 that is orders of magnitude smaller.
 *   • Composable — the adapter serialises to a few KB (see {@link LoRAAdapter.serialize}),
 *     so a persona / tenant / project is an MB-scale artifact you swap at load
 *     time, not a full checkpoint.
 *   • Forgetting-safe — the base never moves, so an adapter cannot catastrophically
 *     overwrite the pretrained model (the exact property WSLA approximates with a
 *     trust region; LoRA gets it structurally).
 *
 * QLoRA adds one more: the frozen base is held QUANTIZED (fp16 or int8) and
 * dequantized on the fly for the merged forward, while the small adapter trains
 * in fp32. On a single constrained device (our WebGPU target) the frozen base is
 * where most bytes live, so quantizing it is the biggest memory unlock.
 *
 * Pure CPU, exact gradients (finite-difference checked in tests). The WGSL path
 * is a future acceleration with the same shapes — same contract as EvermindLM.
 */

import { AdamW, type AdamWOptions } from "../optim/adamw.js";
import { SeededRng } from "../utils/rng.js";
import {
  quantizeFp16,
  dequantizeFp16,
  quantizeInt8,
  dequantizeInt8,
} from "../utils/quantization.js";

/** How the frozen base matrix is stored. `none` = fp32 (plain LoRA); the others = QLoRA. */
export type BaseQuant = "none" | "fp16" | "int8";

export interface LoRAConfig {
  /** Low-rank bottleneck. Higher = more capacity, larger adapter. Default 8. */
  rank?: number;
  /** LoRA scaling; the delta is scaled by alpha/rank. Default = rank (unit scale). */
  alpha?: number;
  /** Deterministic init seed for the A matrix. */
  seed?: number;
}

const ADAPTER_MAGIC = 0x4c4f5241; // "LORA"

/**
 * A low-rank adapter over one base matrix of shape [rows × cols] (row-major).
 *
 *   ΔW[i,j] = (α/r) · Σ_k B[i,k]·A[k,j]          B: [rows×r]  A: [r×cols]
 *
 * Standard LoRA init: A ~ N(0, σ²), B = 0, so ΔW starts at exactly zero and the
 * adapted model equals the base until training moves it. Only A and B are
 * trainable ({@link parameters}/{@link gradients} expose them to {@link AdamW});
 * the base is supplied by the caller and never mutated here.
 */
export class LoRAAdapter {
  readonly rows: number;
  readonly cols: number;
  readonly rank: number;
  readonly alpha: number;
  readonly scale: number;
  /** [rows × rank], zero-initialised. */
  readonly B: Float32Array;
  /** [rank × cols], gaussian-initialised. */
  readonly A: Float32Array;
  private readonly gB: Float32Array;
  private readonly gA: Float32Array;

  constructor(rows: number, cols: number, config: LoRAConfig = {}) {
    if (rows <= 0 || cols <= 0) throw new Error("LoRAAdapter: rows and cols must be > 0");
    const rank = config.rank ?? 8;
    if (rank <= 0 || rank > Math.min(rows, cols)) {
      throw new Error(`LoRAAdapter: rank must be in [1, min(rows,cols)=${Math.min(rows, cols)}]`);
    }
    this.rows = rows;
    this.cols = cols;
    this.rank = rank;
    this.alpha = config.alpha ?? rank;
    this.scale = this.alpha / rank;

    this.B = new Float32Array(rows * rank); // zero-init ⇒ ΔW starts at 0
    this.A = new Float32Array(rank * cols);
    const rng = new SeededRng((config.seed ?? 0x4c6f5241) >>> 0 || 1);
    // Kaiming-ish small init for A (std ~ 1/sqrt(cols)).
    const std = 1 / Math.sqrt(cols);
    for (let i = 0; i < this.A.length; i++) {
      const u1 = Math.max(rng.next(), 1e-12);
      const u2 = rng.next();
      this.A[i] = std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    this.gB = new Float32Array(this.B.length);
    this.gA = new Float32Array(this.A.length);
  }

  /** Trainable count — the LoRA saving vs a full rows·cols matrix. */
  numParams(): number {
    return this.B.length + this.A.length;
  }

  /** Materialise ΔW = scale·B·A as a flat [rows·cols] row-major array. */
  delta(): Float32Array {
    const { rows, cols, rank, scale } = this;
    const out = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      const bOff = i * rank;
      const oOff = i * cols;
      for (let k = 0; k < rank; k++) {
        const b = this.B[bOff + k]! * scale;
        if (b === 0) continue;
        const aOff = k * cols;
        for (let j = 0; j < cols; j++) out[oOff + j] = out[oOff + j]! + b * this.A[aOff + j]!;
      }
    }
    return out;
  }

  /** Base + ΔW (new array; `base` is not mutated). */
  applyTo(base: Float32Array): Float32Array {
    if (base.length !== this.rows * this.cols) {
      throw new Error("LoRAAdapter.applyTo: base length mismatch");
    }
    const d = this.delta();
    const out = new Float32Array(base.length);
    for (let i = 0; i < out.length; i++) out[i] = base[i]! + d[i]!;
    return out;
  }

  /**
   * Project dL/dW (flat [rows·cols], the gradient the base matrix WOULD receive)
   * onto the adapter, accumulating dL/dA and dL/dB. Exact chain rule for
   * ΔW = scale·B·A:  gA = scale·Bᵀ·G,  gB = scale·G·Aᵀ.
   */
  accumulateGradient(gW: Float32Array): void {
    const { rows, cols, rank, scale } = this;
    if (gW.length !== rows * cols) throw new Error("LoRAAdapter.accumulateGradient: gW length mismatch");
    for (let i = 0; i < rows; i++) {
      const gOff = i * cols;
      const bOff = i * rank;
      for (let k = 0; k < rank; k++) {
        const aOff = k * cols;
        let gb = 0;
        const bScaled = this.B[bOff + k]! * scale;
        for (let j = 0; j < cols; j++) {
          const g = gW[gOff + j]!;
          gb += g * this.A[aOff + j]!; // gB[i,k] = scale·Σ_j G[i,j]·A[k,j]
          this.gA[aOff + j] = this.gA[aOff + j]! + bScaled * g; // gA[k,j] += scale·B[i,k]·G[i,j]
        }
        this.gB[bOff + k] = this.gB[bOff + k]! + scale * gb;
      }
    }
  }

  // AdamW OptimTarget surface — trains ONLY the adapter.
  parameters(): { data: Float32Array }[] {
    return [{ data: this.B }, { data: this.A }];
  }
  gradients(): { data: Float32Array }[] {
    return [{ data: this.gB }, { data: this.gA }];
  }
  zeroGrad(): void {
    this.gB.fill(0);
    this.gA.fill(0);
  }

  /** Compact self-describing adapter blob (magic, rows, cols, rank, alpha, B, A). */
  serialize(): ArrayBuffer {
    const headerEls = 5;
    const headerBytes = headerEls * 4;
    const buf = new ArrayBuffer(headerBytes + (this.B.length + this.A.length) * 4);
    const head = new Uint32Array(buf, 0, headerEls);
    head[0] = ADAPTER_MAGIC;
    head[1] = this.rows;
    head[2] = this.cols;
    head[3] = this.rank;
    new Float32Array(buf, 16, 1)[0] = this.alpha;
    const body = new Float32Array(buf, headerBytes);
    body.set(this.B, 0);
    body.set(this.A, this.B.length);
    return buf;
  }

  static deserialize(buffer: ArrayBuffer): LoRAAdapter {
    const head = new Uint32Array(buffer, 0, 5);
    if (head[0] !== ADAPTER_MAGIC) throw new Error("LoRAAdapter.deserialize: bad magic");
    const rows = head[1]!;
    const cols = head[2]!;
    const rank = head[3]!;
    const alpha = new Float32Array(buffer, 16, 1)[0]!;
    const adapter = new LoRAAdapter(rows, cols, { rank, alpha });
    const body = new Float32Array(buffer, 20);
    adapter.B.set(body.subarray(0, adapter.B.length));
    adapter.A.set(body.subarray(adapter.B.length, adapter.B.length + adapter.A.length));
    return adapter;
  }
}

/** Quantize a base matrix for QLoRA storage; returns a dequantized view + byte cost. */
export function quantizeBase(base: Float32Array, mode: BaseQuant): { view: Float32Array; bytes: number } {
  switch (mode) {
    case "none":
      return { view: base, bytes: base.length * 4 };
    case "fp16": {
      const q = quantizeFp16(base);
      return { view: dequantizeFp16(q), bytes: q.length * 2 };
    }
    case "int8": {
      const q = quantizeInt8(base);
      return { view: dequantizeInt8(q.data, q.scale), bytes: q.data.length + 4 };
    }
  }
}
