/**
 * evermind_lm.ts — EvermindLM: a small but complete generative language model.
 *
 * This is what turns a trained checkpoint into an *AI that generates text* (the
 * thing a marketplace buyer actually runs). Architecture (Mamba-flavoured, the
 * minimal exact-gradient CPU reference):
 *
 *   x_t = Embed[token_t]
 *   per layer:
 *     x_t += DepthwiseCausalConv(x)_t            // temporal mixing (short conv)
 *     x_t += SharedExpertMoE(x_t)                // per-position channel mixing (sparse)
 *   logits_t = x_t · Embedᵀ                       // tied output head
 *
 * The token mixer is a depthwise causal convolution (each channel sees a short
 * window of its own past — Mamba's pre-conv) and the channel mixer is the
 * shared-expert MoE, so the model is genuinely sparse. Embeddings are tied
 * (input lookup == output head), which the gradient code accounts for.
 *
 * Pure CPU, exact forward + backward (finite-difference checked), reusing the
 * engine's MoE, cross-entropy, and AdamW. The WGSL/WebGPU path is a future
 * acceleration with the same shapes.
 */

import { SharedExpertMoE } from "../moe/moe_model.js";
import { crossEntropyLoss, crossEntropyGrad } from "../training/autograd.js";
import { AdamW, type AdamWOptions } from "../optim/adamw.js";
import { DynamicLossScaler, roundFp16, type LossScalerOptions } from "../training/mixed_precision.js";
import { SeededRng } from "../utils/rng.js";
import { quantizeFp16, dequantizeFp16 } from "../utils/quantization.js";
import { appendCrcTrailer, verifyCrcTrailer } from "../utils/crc32.js";
import { computeRowDelta, applyRowDelta, serializeRowDelta, deserializeRowDelta } from "../utils/delta.js";

export interface EvermindLMConfig {
  /** Vocabulary size (the only required field; everything else has a default). */
  vocabSize: number;
  /** Model (channel) dimension. Default 64. */
  dModel?: number;
  /** Number of (conv + MoE) blocks. Default 2. */
  numLayers?: number;
  /** Causal conv kernel width. Default 3. */
  convKernel?: number;
  /** Hidden width of each MoE expert FFN. Default 2·dModel. */
  hiddenDim?: number;
  /** Routed experts per MoE layer. Default 4. */
  numExperts?: number;
  /** Experts activated per token. Default 2. */
  topK?: number;
  /** Deterministic init seed. */
  seed?: number;
}

export const DEFAULT_LM_CONFIG: Required<Omit<EvermindLMConfig, "seed" | "vocabSize">> = {
  dModel: 64,
  numLayers: 2,
  convKernel: 3,
  hiddenDim: 128,
  numExperts: 4,
  topK: 2,
};

export const DEFAULT_LM_SEED = 0x45564c4d; // "EVLM"
const MAGIC = 0x45564c30; // "EVL0"

interface MoECacheLike {
  x: Float32Array;
  route: { experts: number[]; gates: number[]; probs: Float32Array };
  sharedPre: Float32Array;
  sharedH: Float32Array;
  expertOut: Float32Array[];
  expertPre: Float32Array[];
  expertH: Float32Array[];
}

interface LayerCache {
  layerIn: Float32Array[]; // residual base for the conv sub-block (the layer input)
  normedConv: Float32Array[]; // RMSNorm(layerIn) — the conv input
  rmsConv: number[]; // per-position RMS denom for the conv norm
  afterConv: Float32Array[]; // residual base for the MoE sub-block
  rmsMoe: number[]; // per-position RMS denom for the MoE norm
  moeCache: MoECacheLike[]; // per position
}

interface ForwardCache {
  tokens: number[];
  layers: LayerCache[];
  finalX: Float32Array[]; // per position, fed to the tied head
}

/** A tokenizer the LM can read/write text through (the engine's `BPETokenizer` fits). */
export interface TextCodec {
  encode(text: string): number[];
  decode(ids: number[]): string;
}

export interface LMGenerateOptions {
  maxNewTokens: number;
  /** Sampling temperature; ≤0 ⇒ greedy argmax. Default 0 (greedy). */
  temperature?: number;
  /** Deterministic sampler seed (only used when temperature > 0). */
  seed?: number;
  /** Stop generating when this token id is produced. */
  stopToken?: number;
}

export class EvermindLM {
  readonly config: Required<Omit<EvermindLMConfig, "seed">>;

  /** Tied token embedding / output head: vocabSize × dModel (row-major). */
  emb: Float32Array;
  private gEmb: Float32Array;
  /** Per-layer depthwise causal conv kernels: dModel × convKernel. */
  private readonly conv: Float32Array[];
  private readonly gConv: Float32Array[];
  /** Per-layer pre-conv / pre-MoE RMSNorm gains (dModel each). */
  private readonly nConv: Float32Array[];
  private readonly gNConv: Float32Array[];
  private readonly nMoe: Float32Array[];
  private readonly gNMoe: Float32Array[];
  /** Per-layer channel mixer. */
  private readonly moe: SharedExpertMoE[];

  constructor(config: EvermindLMConfig) {
    const dModel = config.dModel ?? DEFAULT_LM_CONFIG.dModel;
    const cfg: Required<Omit<EvermindLMConfig, "seed">> = {
      vocabSize: config.vocabSize,
      dModel,
      numLayers: config.numLayers ?? DEFAULT_LM_CONFIG.numLayers,
      convKernel: config.convKernel ?? DEFAULT_LM_CONFIG.convKernel,
      hiddenDim: config.hiddenDim ?? dModel * 2,
      numExperts: config.numExperts ?? DEFAULT_LM_CONFIG.numExperts,
      topK: config.topK ?? DEFAULT_LM_CONFIG.topK,
    };
    if (cfg.vocabSize <= 0) throw new Error("EvermindLM: vocabSize must be > 0");
    if (cfg.topK > cfg.numExperts) throw new Error("EvermindLM: topK must be ≤ numExperts");
    this.config = cfg;

    const seed = (config.seed ?? DEFAULT_LM_SEED) >>> 0 || 1;
    const rng = new SeededRng(seed);
    const gauss = (n: number, std: number): Float32Array => {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const u1 = Math.max(rng.next(), 1e-12);
        const u2 = rng.next();
        a[i] = std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      return a;
    };

    this.emb = gauss(cfg.vocabSize * cfg.dModel, 0.02);
    this.gEmb = new Float32Array(this.emb.length);
    this.conv = [];
    this.gConv = [];
    this.nConv = [];
    this.gNConv = [];
    this.nMoe = [];
    this.gNMoe = [];
    this.moe = [];
    for (let l = 0; l < cfg.numLayers; l++) {
      // Conv init near an identity passthrough (current tap ≈ 1, history ≈ 0) so
      // an untrained block is close to a residual no-op.
      const k = new Float32Array(cfg.dModel * cfg.convKernel);
      for (let c = 0; c < cfg.dModel; c++) k[c * cfg.convKernel] = 1;
      this.conv.push(k);
      this.gConv.push(new Float32Array(k.length));
      // RMSNorm gains start at 1 (identity scale).
      this.nConv.push(new Float32Array(cfg.dModel).fill(1));
      this.gNConv.push(new Float32Array(cfg.dModel));
      this.nMoe.push(new Float32Array(cfg.dModel).fill(1));
      this.gNMoe.push(new Float32Array(cfg.dModel));
      // Each MoE layer gets a distinct seed for varied expert init.
      this.moe.push(
        new SharedExpertMoE({
          modelDim: cfg.dModel,
          hiddenDim: cfg.hiddenDim,
          numExperts: cfg.numExperts,
          topK: cfg.topK,
          seed: seed + 1 + l,
        }),
      );
    }
  }

  // ── Forward ────────────────────────────────────────────────────────────────

  /** Embed a token sequence into per-position channel vectors. */
  private _embed(tokens: number[]): Float32Array[] {
    const { dModel } = this.config;
    return tokens.map((tok) => {
      const row = new Float32Array(dModel);
      const off = tok * dModel;
      for (let c = 0; c < dModel; c++) row[c] = this.emb[off + c]!;
      return row;
    });
  }

  /**
   * One (conv + MoE) block: pre-norm → depthwise causal conv → residual, then
   * pre-norm → MoE channel mixer → residual. Returns the block output and the
   * activation cache its backward needs. Isolating this is what lets
   * {@link lossAndBackwardCheckpointed} recompute a layer's activations on demand
   * instead of retaining every layer's cache at once.
   */
  private _forwardLayer(l: number, layerIn: Float32Array[]): { afterMoe: Float32Array[]; cache: LayerCache } {
    const { dModel, convKernel } = this.config;
    const T = layerIn.length;
    const ker = this.conv[l]!;
    const nConv = this.nConv[l]!;
    const nMoe = this.nMoe[l]!;

    const normedConv: Float32Array[] = [];
    const rmsConv: number[] = [];
    for (let t = 0; t < T; t++) {
      const { y, r } = rmsNorm(layerIn[t]!, nConv);
      normedConv.push(y);
      rmsConv.push(r);
    }
    const afterConv: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const out = Float32Array.from(layerIn[t]!); // residual base
      for (let c = 0; c < dModel; c++) {
        let acc = 0;
        for (let j = 0; j < convKernel; j++) {
          const ti = t - j;
          if (ti >= 0) acc += ker[c * convKernel + j]! * normedConv[ti]![c]!;
        }
        out[c] = out[c]! + acc;
      }
      afterConv.push(out);
    }

    const rmsMoe: number[] = [];
    const moeCache: MoECacheLike[] = [];
    const afterMoe: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const { y, r } = rmsNorm(afterConv[t]!, nMoe);
      rmsMoe.push(r);
      const out = Float32Array.from(afterConv[t]!); // residual base
      const mr = this.moe[l]!.forward(y);
      for (let c = 0; c < dModel; c++) out[c] = out[c]! + mr.output[c]!;
      afterMoe.push(out);
      moeCache.push(mr.cache as unknown as MoECacheLike);
    }
    return { afterMoe, cache: { layerIn, normedConv, rmsConv, afterConv, rmsMoe, moeCache } };
  }

  /** Tied output head: logits_t[v] = x_t · emb[v]. */
  private _head(x: Float32Array[]): Float32Array[] {
    const { dModel, vocabSize } = this.config;
    return x.map((xt) => {
      const lg = new Float32Array(vocabSize);
      for (let v = 0; v < vocabSize; v++) {
        let acc = 0;
        const off = v * dModel;
        for (let c = 0; c < dModel; c++) acc += xt[c]! * this.emb[off + c]!;
        lg[v] = acc;
      }
      return lg;
    });
  }

  /** Run the model over a token sequence; returns per-position logits + a cache. */
  forward(tokens: number[]): { logits: Float32Array[]; cache: ForwardCache } {
    let x = this._embed(tokens);
    const layers: LayerCache[] = [];
    for (let l = 0; l < this.config.numLayers; l++) {
      const { afterMoe, cache } = this._forwardLayer(l, x);
      layers.push(cache);
      x = afterMoe;
    }
    return { logits: this._head(x), cache: { tokens, layers, finalX: x } };
  }

  // ── Loss + backward ──────────────────────────────────────────────────────────

  /**
   * Head + tied-embedding gradient. Accumulates dL/d(head→emb) into gEmb and
   * returns the mean next-token loss plus dL/d(finalX). Shared by the full and
   * checkpointed backward paths so the head maths lives in one place.
   */
  private _headBackward(tokens: number[], logits: Float32Array[], finalX: Float32Array[]): { loss: number; dX: Float32Array[] } {
    const { dModel, vocabSize } = this.config;
    const T = tokens.length;
    const predPositions = T - 1; // positions 0..T-2 predict the next token
    const inv = 1 / predPositions;
    const dX: Float32Array[] = Array.from({ length: T }, () => new Float32Array(dModel));
    let loss = 0;
    for (let t = 0; t < predPositions; t++) {
      const target = tokens[t + 1]!;
      loss += crossEntropyLoss(logits[t]!, target) * inv;
      const dLogit = crossEntropyGrad(logits[t]!, target); // probs - onehot
      const xt = finalX[t]!;
      for (let v = 0; v < vocabSize; v++) {
        const g = dLogit[v]! * inv;
        if (g === 0) continue;
        const off = v * dModel;
        for (let c = 0; c < dModel; c++) {
          this.gEmb[off + c] = this.gEmb[off + c]! + g * xt[c]!; // head → emb
          dX[t]![c] = dX[t]![c]! + g * this.emb[off + c]!; // head → x_t
        }
      }
    }
    return { loss, dX };
  }

  /**
   * Backward through one (conv + MoE) block given dL/d(block output) and the
   * block's activation cache. Accumulates conv/norm/MoE gradients and returns
   * dL/d(block input). The inverse of {@link _forwardLayer}.
   */
  private _backwardLayer(l: number, dOut: Float32Array[], lc: LayerCache): Float32Array[] {
    const { dModel, convKernel } = this.config;
    const T = dOut.length;
    const ker = this.conv[l]!;
    const gker = this.gConv[l]!;
    const nConv = this.nConv[l]!;
    const gNConv = this.gNConv[l]!;
    const nMoe = this.nMoe[l]!;
    const gNMoe = this.gNMoe[l]!;

    // MoE sub-block: afterMoe = afterConv + MoE(RMSNorm(afterConv, nMoe)).
    const dAfterConv: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const dMoeNormed = this.moe[l]!.backward(dOut[t]!, lc.moeCache[t] as never);
      const { dx, dgain } = rmsNormBackward(dMoeNormed, lc.afterConv[t]!, lc.rmsMoe[t]!, nMoe);
      for (let c = 0; c < dModel; c++) gNMoe[c] = gNMoe[c]! + dgain[c]!;
      const d = Float32Array.from(dOut[t]!); // residual passthrough
      for (let c = 0; c < dModel; c++) d[c] = d[c]! + dx[c]!;
      dAfterConv.push(d);
    }

    // Conv sub-block: afterConv = layerIn + conv(RMSNorm(layerIn, nConv)).
    const dNormedConv: Float32Array[] = Array.from({ length: T }, () => new Float32Array(dModel));
    const dLayerIn: Float32Array[] = dAfterConv.map((v) => Float32Array.from(v)); // residual passthrough
    for (let t = 0; t < T; t++) {
      for (let c = 0; c < dModel; c++) {
        const dmix = dAfterConv[t]![c]!;
        if (dmix === 0) continue;
        for (let j = 0; j < convKernel; j++) {
          const ti = t - j;
          if (ti < 0) continue;
          gker[c * convKernel + j] = gker[c * convKernel + j]! + dmix * lc.normedConv[ti]![c]!;
          dNormedConv[ti]![c] = dNormedConv[ti]![c]! + dmix * ker[c * convKernel + j]!;
        }
      }
    }
    for (let t = 0; t < T; t++) {
      const { dx, dgain } = rmsNormBackward(dNormedConv[t]!, lc.layerIn[t]!, lc.rmsConv[t]!, nConv);
      for (let c = 0; c < dModel; c++) {
        gNConv[c] = gNConv[c]! + dgain[c]!;
        dLayerIn[t]![c] = dLayerIn[t]![c]! + dx[c]!;
      }
    }
    return dLayerIn;
  }

  /** Embedding lookup: dL/d(layer-0 input) flows into the row for each token. */
  private _embedBackward(tokens: number[], dIn: Float32Array[]): void {
    const { dModel } = this.config;
    for (let t = 0; t < tokens.length; t++) {
      const off = tokens[t]! * dModel;
      for (let c = 0; c < dModel; c++) this.gEmb[off + c] = this.gEmb[off + c]! + dIn[t]![c]!;
    }
  }

  /**
   * Next-token cross-entropy over the sequence (predict tokens[t+1] from
   * position t), accumulating exact gradients. Returns the mean loss. Call
   * {@link zeroGrad} before and an optimiser step after.
   */
  lossAndBackward(tokens: number[]): number {
    if (tokens.length < 2) return 0;
    const { logits, cache } = this.forward(tokens);
    const { loss, dX } = this._headBackward(tokens, logits, cache.finalX);
    let d = dX;
    for (let l = this.config.numLayers - 1; l >= 0; l--) d = this._backwardLayer(l, d, cache.layers[l]!);
    this._embedBackward(tokens, d);
    return loss;
  }

  /**
   * Activation-checkpointed backward — numerically identical gradients to
   * {@link lossAndBackward}, but retains only the per-LAYER inputs during the
   * forward instead of every layer's full activation cache. Each layer's
   * activations are RECOMPUTED (a cheap extra forward) when its backward runs,
   * so peak activation memory is one layer's cache, not all of them. This is the
   * memory-for-compute trade the cookbook pairs with FSDP to fit longer
   * sequences / bigger models on a constrained device.
   */
  lossAndBackwardCheckpointed(tokens: number[]): number {
    if (tokens.length < 2) return 0;
    let x = this._embed(tokens);
    const layerInputs: Float32Array[][] = [];
    for (let l = 0; l < this.config.numLayers; l++) {
      layerInputs.push(x); // keep only the input; drop the cache
      x = this._forwardLayer(l, x).afterMoe;
    }
    const logits = this._head(x);
    const { loss, dX } = this._headBackward(tokens, logits, x);
    let d = dX;
    for (let l = this.config.numLayers - 1; l >= 0; l--) {
      const { cache } = this._forwardLayer(l, layerInputs[l]!); // recompute this layer's activations
      d = this._backwardLayer(l, d, cache);
    }
    this._embedBackward(tokens, d);
    return loss;
  }

  // ── Generation ───────────────────────────────────────────────────────────────

  /**
   * Text-level generation: encode the prompt, generate, decode. `codec` is any
   * tokenizer exposing encode/decode (the engine's `BPETokenizer` satisfies it),
   * so the LM consumes and emits real text rather than raw token ids. The model's
   * `vocabSize` must match the codec's vocabulary.
   */
  generateText(prompt: string, codec: TextCodec, opts: LMGenerateOptions): string {
    return codec.decode(this.generate(codec.encode(prompt), opts));
  }

  /** Greedy / temperature-sampled autoregressive generation. Returns NEW token ids. */
  generate(prompt: number[], opts: LMGenerateOptions): number[] {
    const temperature = opts.temperature ?? 0;
    const rng = temperature > 0 ? new SeededRng((opts.seed ?? 1) >>> 0 || 1) : null;
    const tokens = [...prompt];
    const produced: number[] = [];
    for (let n = 0; n < opts.maxNewTokens; n++) {
      const { logits } = this.forward(tokens.length > 0 ? tokens : [0]);
      const last = logits[logits.length - 1]!;
      const next = rng ? sampleTemperature(last, temperature, rng) : argmax(last);
      produced.push(next);
      tokens.push(next);
      if (opts.stopToken !== undefined && next === opts.stopToken) break;
    }
    return produced;
  }

  // ── Parameters / checkpoint ──────────────────────────────────────────────────

  /** All trainable parameters as {data} (AdamW-compatible), canonical order. */
  parameters(): { data: Float32Array }[] {
    const out: { data: Float32Array }[] = [{ data: this.emb }];
    for (let l = 0; l < this.config.numLayers; l++) {
      out.push({ data: this.conv[l]! }, { data: this.nConv[l]! }, { data: this.nMoe[l]! });
      for (const p of this.moe[l]!.parameters()) out.push({ data: p.data });
    }
    return out;
  }

  /** Gradient buffers, index-aligned with {@link parameters}. */
  gradients(): { data: Float32Array }[] {
    const out: { data: Float32Array }[] = [{ data: this.gEmb }];
    for (let l = 0; l < this.config.numLayers; l++) {
      out.push({ data: this.gConv[l]! }, { data: this.gNConv[l]! }, { data: this.gNMoe[l]! });
      for (const g of this.moe[l]!.gradients()) out.push({ data: g.data });
    }
    return out;
  }

  zeroGrad(): void {
    this.gEmb.fill(0);
    for (let l = 0; l < this.config.numLayers; l++) {
      this.gConv[l]!.fill(0);
      this.gNConv[l]!.fill(0);
      this.gNMoe[l]!.fill(0);
      this.moe[l]!.zeroGrad();
    }
  }

  /** Serialise to an "EVL0" binary (fp16 or f32), params in {@link parameters} order. */
  exportWeights(opts: { fp16?: boolean } = {}): ArrayBuffer {
    const fp16 = opts.fp16 ?? false;
    const params = this.parameters();
    const total = params.reduce((n, p) => n + p.data.length, 0);
    // magic, version, vocab, dModel, numLayers, convKernel, hiddenDim, numExperts, topK.
    // numExperts and topK get distinct slots (an earlier *16 packing collided once
    // numExperts ≥ 16 — e.g. (20,20) and (21,4) both packed to 340).
    const headerEls = 9;
    const headerBytes = headerEls * 4;
    const buf = new ArrayBuffer(headerBytes + (fp16 ? total * 2 : total * 4));
    const head = new Uint32Array(buf, 0, headerEls);
    head[0] = MAGIC;
    head[1] = fp16 ? 2 : 1;
    head[2] = this.config.vocabSize;
    head[3] = this.config.dModel;
    head[4] = this.config.numLayers;
    head[5] = this.config.convKernel;
    head[6] = this.config.hiddenDim;
    head[7] = this.config.numExperts;
    head[8] = this.config.topK;
    const flat = new Float32Array(total);
    let o = 0;
    for (const p of params) {
      flat.set(p.data, o);
      o += p.data.length;
    }
    if (fp16) new Uint16Array(buf, headerBytes, total).set(quantizeFp16(flat));
    else new Float32Array(buf, headerBytes, total).set(flat);
    // Append a CRC-32 trailer so a corrupt/truncated checkpoint is caught on load
    // (backward-compatible: pre-CRC readers ignore the trailing bytes). (EVM-7)
    return appendCrcTrailer(buf);
  }

  /** Load weights from an "EVL0" binary. Validates CRC (when present), magic + dims. */
  loadWeights(buffer: ArrayBuffer): void {
    this._setFlat(this._readFlat(buffer));
  }

  /** Parse + validate an EVL0 checkpoint to a flat f32 param vector (no mutation). */
  private _readFlat(buffer: ArrayBuffer): Float32Array {
    const crc = verifyCrcTrailer(buffer);
    if (crc.hasTrailer && !crc.ok) {
      throw new Error("EvermindLM.loadWeights: checkpoint failed CRC integrity check (corrupt or truncated)");
    }
    const head = new Uint32Array(buffer, 0, 9);
    if (head[0] !== MAGIC) throw new Error("EvermindLM.loadWeights: bad magic (not an EVL0 checkpoint)");
    const version = head[1]!;
    if (
      head[2] !== this.config.vocabSize ||
      head[3] !== this.config.dModel ||
      head[4] !== this.config.numLayers ||
      head[5] !== this.config.convKernel ||
      head[6] !== this.config.hiddenDim ||
      head[7] !== this.config.numExperts ||
      head[8] !== this.config.topK
    ) {
      throw new Error("EvermindLM.loadWeights: config mismatch with checkpoint");
    }
    const total = this.parameters().reduce((n, p) => n + p.data.length, 0);
    const headerBytes = 36;
    return version === 2
      ? dequantizeFp16(new Uint16Array(buffer, headerBytes, total))
      : new Float32Array(buffer.slice(headerBytes, headerBytes + total * 4));
  }

  /** Flat concatenation of all params, canonical order (matches checkpoint layout). */
  private _flat(): Float32Array {
    const params = this.parameters();
    const total = params.reduce((n, p) => n + p.data.length, 0);
    const flat = new Float32Array(total);
    let o = 0;
    for (const p of params) { flat.set(p.data, o); o += p.data.length; }
    return flat;
  }

  /** Distribute a flat param vector back into the model's parameters. */
  private _setFlat(flat: Float32Array): void {
    let o = 0;
    for (const p of this.parameters()) {
      p.data.set(flat.subarray(o, o + p.data.length));
      o += p.data.length;
    }
  }

  /**
   * Export a SPARSE DELTA of the current weights against a base EVL0 checkpoint
   * (EVM-6). Online WSLA updates only a few rows, so a delta persists kilobytes
   * instead of rewriting the whole model. Reconstruct with {@link loadDelta}.
   */
  exportDelta(baseCheckpoint: ArrayBuffer, opts: { eps?: number } = {}): ArrayBuffer {
    const base = this._readFlat(baseCheckpoint);
    const current = this._flat();
    // Row-granular when the param vector tiles evenly by the model width (the
    // embedding dominates and is dModel-wide); else fall back to per-element.
    const rowSize = current.length % this.config.dModel === 0 ? this.config.dModel : 1;
    return serializeRowDelta(computeRowDelta(base, current, rowSize, opts.eps ?? 0));
  }

  /** Reconstruct weights from a base EVL0 checkpoint + a delta (EVM-6). */
  loadDelta(baseCheckpoint: ArrayBuffer, delta: ArrayBuffer): void {
    const base = this._readFlat(baseCheckpoint);
    this._setFlat(applyRowDelta(base, deserializeRowDelta(delta)));
  }
}

export interface EvermindLMTrainOptions extends AdamWOptions {
  epochs?: number;
  /**
   * Gradient accumulation: average gradients over this many sequences (micro-
   * batches) before each optimiser step, for a larger effective batch on a
   * memory-constrained device. Default 1.
   */
  accumSteps?: number;
  /**
   * Use activation checkpointing (recompute layer activations in backward) to
   * cap peak activation memory. Identical gradients, a little extra compute.
   * Default false.
   */
  checkpoint?: boolean;
  /**
   * Mixed-precision training: fp16-rounded gradients with dynamic loss scaling
   * over fp32 master weights (the model params). Overflowing steps are skipped
   * and the scale backs off. Default false.
   */
  mixedPrecision?: boolean | LossScalerOptions;
}

/** Minimal sequence trainer: AdamW over next-token cross-entropy. */
export class EvermindLMTrainer {
  private readonly adam: AdamW;
  private readonly scaler: DynamicLossScaler | null;
  private readonly grads: { data: Float32Array }[];
  constructor(
    private readonly model: EvermindLM,
    private readonly opts: EvermindLMTrainOptions = {},
  ) {
    this.adam = new AdamW(model, opts);
    this.grads = model.gradients();
    this.scaler = opts.mixedPrecision
      ? new DynamicLossScaler(typeof opts.mixedPrecision === "object" ? opts.mixedPrecision : {})
      : null;
  }

  /** The dynamic loss scaler (mixed-precision mode only) — exposes scale/overflow stats. */
  get lossScaler(): DynamicLossScaler | null {
    return this.scaler;
  }

  private _backward(seq: number[]): number {
    return this.opts.checkpoint ? this.model.lossAndBackwardCheckpointed(seq) : this.model.lossAndBackward(seq);
  }

  /** Divide accumulated gradients by `n` (accumulation averaging), in place. */
  private _scaleGrads(n: number): void {
    if (n === 1) return;
    const inv = 1 / n;
    for (const g of this.grads) for (let i = 0; i < g.data.length; i++) g.data[i] = g.data[i]! * inv;
  }

  /** Round accumulated gradients to fp16 precision (mixed-precision simulation), in place. */
  private _fp16Grads(): void {
    for (const g of this.grads) for (let i = 0; i < g.data.length; i++) g.data[i] = roundFp16(g.data[i]!);
  }

  /** Multiply accumulated gradients by `s`, in place. */
  private _mulGrads(s: number): void {
    if (s === 1) return;
    for (const g of this.grads) for (let i = 0; i < g.data.length; i++) g.data[i] = g.data[i]! * s;
  }

  /** Train on a set of token sequences; returns per-epoch mean loss. */
  fit(sequences: number[][]): number[] {
    const epochs = this.opts.epochs ?? 1;
    const accum = Math.max(1, this.opts.accumSteps ?? 1);
    const history: number[] = [];
    for (let e = 0; e < epochs; e++) {
      let total = 0;
      let n = 0;
      let pending = 0;
      // The scale is fixed for the duration of one accumulation window, so scale-up
      // (per sequence) and unscale (at flush) always use the SAME factor even as the
      // controller grows/backs it off between windows.
      let windowScale = this.scaler ? this.scaler.scale : 1;
      this.model.zeroGrad();
      const flush = () => {
        if (pending === 0) return;
        this._scaleGrads(pending); // accumulation averaging
        if (this.scaler) {
          this._fp16Grads(); // half-precision gradients
          const overflow = this.scaler.check(this.grads);
          if (this.scaler.update(overflow)) {
            this._mulGrads(1 / windowScale); // unscale by the SAME factor we scaled by
            this.adam.step();
          } // else: skip the step, scale has backed off
          windowScale = this.scaler.scale; // next window uses the updated scale
        } else {
          this.adam.step();
        }
        this.model.zeroGrad();
        pending = 0;
      };
      for (const seq of sequences) {
        if (seq.length < 2) continue;
        if (pending === 0 && this.scaler) windowScale = this.scaler.scale;
        const loss = this._backward(seq);
        // Emulate scaling the loss before backward: the gradient is linear in the
        // loss, so scaling the just-produced gradients by the loss scale is equivalent.
        this._mulGrads(windowScale);
        total += loss;
        n++;
        if (++pending >= accum) flush();
      }
      flush();
      history.push(n > 0 ? total / n : 0);
    }
    return history;
  }
}

const RMS_EPS = 1e-5;

/** RMSNorm: y[c] = gain[c]·x[c]/rms, rms = sqrt(mean(x²)+eps). Returns y and the denom. */
function rmsNorm(x: Float32Array, gain: Float32Array): { y: Float32Array; r: number } {
  const D = x.length;
  let ss = 0;
  for (let c = 0; c < D; c++) ss += x[c]! * x[c]!;
  const r = Math.sqrt(ss / D + RMS_EPS);
  const y = new Float32Array(D);
  for (let c = 0; c < D; c++) y[c] = (gain[c]! * x[c]!) / r;
  return { y, r };
}

/**
 * RMSNorm backward. Given dL/dy and the cached input/denom/gain, returns dL/dx and
 * dL/dgain. dx_j = gain_j·dy_j/r − x_j·A/(D·r³) with A = Σ_c dy_c·gain_c·x_c;
 * dgain_c = dy_c·x_c/r.
 */
function rmsNormBackward(
  dy: Float32Array,
  x: Float32Array,
  r: number,
  gain: Float32Array,
): { dx: Float32Array; dgain: Float32Array } {
  const D = x.length;
  let A = 0;
  for (let c = 0; c < D; c++) A += dy[c]! * gain[c]! * x[c]!;
  const dx = new Float32Array(D);
  const dgain = new Float32Array(D);
  const r3 = r * r * r;
  for (let c = 0; c < D; c++) {
    dx[c] = (gain[c]! * dy[c]!) / r - (x[c]! * A) / (D * r3);
    dgain[c] = (dy[c]! * x[c]!) / r;
  }
  return { dx, dgain };
}

function argmax(v: Float32Array): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i]! > v[best]!) best = i;
  return best;
}

function sampleTemperature(logits: Float32Array, temperature: number, rng: SeededRng): number {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i]! / temperature > max) max = logits[i]! / temperature;
  let sum = 0;
  const probs = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    probs[i] = Math.exp(logits[i]! / temperature - max);
    sum += probs[i]!;
  }
  let r = rng.next() * sum;
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i]!;
    if (r <= 0) return i;
  }
  return probs.length - 1;
}
