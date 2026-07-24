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

import { EvermindLM } from "../lm/evermind_lm.js";

export interface LoRAFitOptions extends AdamWOptions {
  epochs?: number;
  /**
   * Gradient accumulation: average the adapter gradient over this many sequences
   * (micro-batches) before each optimiser step. Lets a memory-constrained device
   * train at a larger *effective* batch. Default 1 (step per sequence).
   */
  accumSteps?: number;
}

/**
 * LoRA / QLoRA fine-tuning of an {@link EvermindLM} through its tied token
 * embedding — the dominant parameter (vocab×dModel) and the natural adapter
 * target (input lookup and output head share it). The base model is frozen; only
 * the {@link LoRAAdapter} trains.
 *
 * QLoRA: pass `baseQuant: "fp16" | "int8"` and the frozen base is stored
 * quantized and dequantized on the fly for the merged forward, so the resident
 * base costs half (fp16) or a quarter (int8) of the bytes while the adapter
 * trains full-precision.
 *
 * The adapter is the shippable artifact: {@link serializeAdapter} emits a few KB
 * you swap per persona/tenant/project, versus rewriting the whole checkpoint.
 */
export class EvermindLMLoRA {
  readonly adapter: LoRAAdapter;
  readonly baseQuant: BaseQuant;
  /** Frozen base embedding as used in the merged forward (dequantized under QLoRA). */
  private readonly frozenBase: Float32Array;
  private readonly baseBytesStored: number;
  private readonly rows: number;
  private readonly cols: number;

  constructor(
    private readonly model: EvermindLM,
    config: LoRAConfig & { baseQuant?: BaseQuant } = {},
  ) {
    this.rows = model.config.vocabSize;
    this.cols = model.config.dModel;
    this.baseQuant = config.baseQuant ?? "none";
    const { view, bytes } = quantizeBase(Float32Array.from(model.emb), this.baseQuant);
    this.frozenBase = view;
    this.baseBytesStored = bytes;
    this.adapter = new LoRAAdapter(this.rows, this.cols, config);
  }

  /** The frozen base model this adapter rides on. */
  get baseModel(): EvermindLM {
    return this.model;
  }

  /** Effective embedding used for training/generation: frozenBase + adapter delta. */
  mergedEmb(): Float32Array {
    return this.adapter.applyTo(this.frozenBase);
  }

  /**
   * One forward+backward on the merged weights, projecting the base-embedding
   * gradient onto the adapter. The underlying model's own weights are left
   * exactly as they were (base frozen); adapter gradients ACCUMULATE (caller
   * zeroes between optimiser windows).
   */
  private _accumulate(seq: number[]): number {
    const merged = this.mergedEmb();
    const saved = this.model.emb;
    this.model.emb = merged; // swap in effective weights (tied lookup + head)
    this.model.zeroGrad();
    const loss = this.model.lossAndBackward(seq);
    const gW = this.model.gradients()[0]!.data; // dL/dW_eff == dL/dΔW, shape [rows×cols]
    this.adapter.accumulateGradient(gW);
    this.model.emb = saved; // restore the frozen base
    return loss;
  }

  /** Train the adapter (only) with AdamW + optional gradient accumulation. */
  fit(sequences: number[][], opts: LoRAFitOptions = {}): number[] {
    const epochs = opts.epochs ?? 1;
    const accum = Math.max(1, opts.accumSteps ?? 1);
    const adam = new AdamW(this.adapter, opts);
    const grads = this.adapter.gradients();
    const history: number[] = [];
    for (let e = 0; e < epochs; e++) {
      let total = 0;
      let n = 0;
      let pending = 0;
      this.adapter.zeroGrad();
      const flush = () => {
        if (pending === 0) return;
        if (pending !== 1) for (const g of grads) for (let i = 0; i < g.data.length; i++) g.data[i] = g.data[i]! / pending;
        adam.step();
        this.adapter.zeroGrad();
        pending = 0;
      };
      for (const seq of sequences) {
        if (seq.length < 2) continue;
        total += this._accumulate(seq);
        n++;
        if (++pending >= accum) flush();
      }
      flush();
      history.push(n > 0 ? total / n : 0);
    }
    return history;
  }

  generate(prompt: number[], opts: import("../lm/evermind_lm.js").LMGenerateOptions): number[] {
    const saved = this.model.emb;
    this.model.emb = this.mergedEmb();
    try {
      return this.model.generate(prompt, opts);
    } finally {
      this.model.emb = saved;
    }
  }

  generateText(prompt: string, codec: import("../lm/evermind_lm.js").TextCodec, opts: import("../lm/evermind_lm.js").LMGenerateOptions): string {
    return codec.decode(this.generate(codec.encode(prompt), opts));
  }

  /** The shippable adapter artifact (a few KB). */
  serializeAdapter(): ArrayBuffer {
    return this.adapter.serialize();
  }

  /** Reconstruct a fine-tuned model from a base model + a serialized adapter. */
  static loadAdapter(model: EvermindLM, adapterBuffer: ArrayBuffer, baseQuant: BaseQuant = "none"): EvermindLMLoRA {
    const loaded = LoRAAdapter.deserialize(adapterBuffer);
    const wrap = new EvermindLMLoRA(model, { rank: loaded.rank, alpha: loaded.alpha, baseQuant });
    wrap.adapter.B.set(loaded.B);
    wrap.adapter.A.set(loaded.A);
    return wrap;
  }

  /** Bake the adapter into the base and return a standalone EVL0 checkpoint. */
  merge(opts: { fp16?: boolean } = {}): ArrayBuffer {
    this.model.emb.set(this.mergedEmb());
    return this.model.exportWeights(opts);
  }

  /** Byte cost of the trainable adapter vs the frozen base — the LoRA/QLoRA saving. */
  footprint(): { adapterBytes: number; baseBytes: number; trainableParams: number; baseParams: number } {
    return {
      adapterBytes: this.adapter.serialize().byteLength,
      baseBytes: this.baseBytesStored,
      trainableParams: this.adapter.numParams(),
      baseParams: this.rows * this.cols,
    };
  }
}
