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
import { SeededRng } from "../utils/rng.js";
import { quantizeFp16, dequantizeFp16 } from "../utils/quantization.js";

export interface EvermindLMConfig {
  /** Vocabulary size. */
  vocabSize: number;
  /** Model (channel) dimension. Default 64. */
  dModel: number;
  /** Number of (conv + MoE) blocks. Default 2. */
  numLayers: number;
  /** Causal conv kernel width. Default 3. */
  convKernel: number;
  /** Hidden width of each MoE expert FFN. Default 2·dModel. */
  hiddenDim: number;
  /** Routed experts per MoE layer. Default 4. */
  numExperts: number;
  /** Experts activated per token. Default 2. */
  topK: number;
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
  convIn: Float32Array[]; // input to the conv (== layer input), per position
  moeIn: Float32Array[]; // input to the MoE (post-conv residual), per position
  moeCache: MoECacheLike[]; // per position
}

interface ForwardCache {
  tokens: number[];
  layers: LayerCache[];
  finalX: Float32Array[]; // per position, fed to the tied head
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
    this.moe = [];
    for (let l = 0; l < cfg.numLayers; l++) {
      // Conv init near an identity passthrough (current tap ≈ 1, history ≈ 0) so
      // an untrained block is close to a residual no-op.
      const k = new Float32Array(cfg.dModel * cfg.convKernel);
      for (let c = 0; c < cfg.dModel; c++) k[c * cfg.convKernel] = 1;
      this.conv.push(k);
      this.gConv.push(new Float32Array(k.length));
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

  /** Run the model over a token sequence; returns per-position logits + a cache. */
  forward(tokens: number[]): { logits: Float32Array[]; cache: ForwardCache } {
    const { dModel, convKernel, numLayers, vocabSize } = this.config;
    const T = tokens.length;

    // Embed.
    let x: Float32Array[] = tokens.map((tok) => {
      const row = new Float32Array(dModel);
      const off = tok * dModel;
      for (let c = 0; c < dModel; c++) row[c] = this.emb[off + c]!;
      return row;
    });

    const layers: LayerCache[] = [];
    for (let l = 0; l < numLayers; l++) {
      const convIn = x.map((v) => Float32Array.from(v));
      // Depthwise causal conv → residual.
      const ker = this.conv[l]!;
      const afterConv: Float32Array[] = [];
      for (let t = 0; t < T; t++) {
        const out = Float32Array.from(x[t]!);
        for (let c = 0; c < dModel; c++) {
          let acc = 0;
          for (let j = 0; j < convKernel; j++) {
            const ti = t - j;
            if (ti >= 0) acc += ker[c * convKernel + j]! * convIn[ti]![c]!;
          }
          out[c] = out[c]! + acc; // residual
        }
        afterConv.push(out);
      }
      // MoE channel mixer → residual.
      const moeIn = afterConv.map((v) => Float32Array.from(v));
      const moeCache: MoECacheLike[] = [];
      const afterMoe: Float32Array[] = [];
      for (let t = 0; t < T; t++) {
        const r = this.moe[l]!.forward(moeIn[t]!);
        const out = Float32Array.from(moeIn[t]!);
        for (let c = 0; c < dModel; c++) out[c] = out[c]! + r.output[c]!;
        afterMoe.push(out);
        moeCache.push(r.cache as unknown as MoECacheLike);
      }
      layers.push({ convIn, moeIn, moeCache });
      x = afterMoe;
    }

    // Tied head: logits_t[v] = x_t · emb[v].
    const logits: Float32Array[] = x.map((xt) => {
      const lg = new Float32Array(vocabSize);
      for (let v = 0; v < vocabSize; v++) {
        let acc = 0;
        const off = v * dModel;
        for (let c = 0; c < dModel; c++) acc += xt[c]! * this.emb[off + c]!;
        lg[v] = acc;
      }
      return lg;
    });

    return { logits, cache: { tokens, layers, finalX: x } };
  }

  // ── Loss + backward ──────────────────────────────────────────────────────────

  /**
   * Next-token cross-entropy over the sequence (predict tokens[t+1] from
   * position t), accumulating exact gradients. Returns the mean loss. Call
   * {@link zeroGrad} before and an optimiser step after.
   */
  lossAndBackward(tokens: number[]): number {
    const { dModel, convKernel, numLayers, vocabSize } = this.config;
    const T = tokens.length;
    if (T < 2) return 0;
    const { logits, cache } = this.forward(tokens);

    const predPositions = T - 1; // positions 0..T-2 predict the next token
    const inv = 1 / predPositions;

    // dL/d(finalX_t) and head gradient into the tied embedding.
    const dX: Float32Array[] = Array.from({ length: T }, () => new Float32Array(dModel));
    let loss = 0;
    for (let t = 0; t < predPositions; t++) {
      const target = tokens[t + 1]!;
      loss += crossEntropyLoss(logits[t]!, target) * inv;
      const dLogit = crossEntropyGrad(logits[t]!, target); // probs - onehot
      const xt = cache.finalX[t]!;
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

    // Backprop through layers in reverse.
    for (let l = numLayers - 1; l >= 0; l--) {
      const lc = cache.layers[l]!;
      // MoE residual: x = moeIn + MoE(moeIn). dMoeIn = dX + MoE.backward(dX).
      const dMoeIn: Float32Array[] = [];
      for (let t = 0; t < T; t++) {
        const dInner = this.moe[l]!.backward(dX[t]!, lc.moeCache[t] as never);
        const d = Float32Array.from(dX[t]!);
        for (let c = 0; c < dModel; c++) d[c] = d[c]! + dInner[c]!;
        dMoeIn.push(d);
      }
      // Conv residual: afterConv = convIn + conv(convIn). dConvIn = dMoeIn + convBwd(dMoeIn).
      const ker = this.conv[l]!;
      const gker = this.gConv[l]!;
      const dConvIn: Float32Array[] = dMoeIn.map((v) => Float32Array.from(v)); // residual passthrough
      for (let t = 0; t < T; t++) {
        for (let c = 0; c < dModel; c++) {
          const dmix = dMoeIn[t]![c]!;
          if (dmix === 0) continue;
          for (let j = 0; j < convKernel; j++) {
            const ti = t - j;
            if (ti < 0) continue;
            gker[c * convKernel + j] = gker[c * convKernel + j]! + dmix * lc.convIn[ti]![c]!;
            dConvIn[ti]![c] = dConvIn[ti]![c]! + dmix * ker[c * convKernel + j]!;
          }
        }
      }
      for (let t = 0; t < T; t++) dX[t] = dConvIn[t]!;
    }

    // Embedding lookup: dX at layer-0 input flows into the row for token_t.
    for (let t = 0; t < T; t++) {
      const off = tokens[t]! * dModel;
      for (let c = 0; c < dModel; c++) this.gEmb[off + c] = this.gEmb[off + c]! + dX[t]![c]!;
    }

    return loss;
  }

  // ── Generation ───────────────────────────────────────────────────────────────

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
      out.push({ data: this.conv[l]! });
      for (const p of this.moe[l]!.parameters()) out.push({ data: p.data });
    }
    return out;
  }

  /** Gradient buffers, index-aligned with {@link parameters}. */
  gradients(): { data: Float32Array }[] {
    const out: { data: Float32Array }[] = [{ data: this.gEmb }];
    for (let l = 0; l < this.config.numLayers; l++) {
      out.push({ data: this.gConv[l]! });
      for (const g of this.moe[l]!.gradients()) out.push({ data: g.data });
    }
    return out;
  }

  zeroGrad(): void {
    this.gEmb.fill(0);
    for (let l = 0; l < this.config.numLayers; l++) {
      this.gConv[l]!.fill(0);
      this.moe[l]!.zeroGrad();
    }
  }

  /** Serialise to an "EVL0" binary (fp16 or f32), params in {@link parameters} order. */
  exportWeights(opts: { fp16?: boolean } = {}): ArrayBuffer {
    const fp16 = opts.fp16 ?? false;
    const params = this.parameters();
    const total = params.reduce((n, p) => n + p.data.length, 0);
    const headerEls = 8; // magic, version, vocab, dModel, numLayers, convKernel, hiddenDim, numExperts(+topK packed)
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
    head[7] = this.config.numExperts * 16 + this.config.topK; // pack numExperts,topK
    const flat = new Float32Array(total);
    let o = 0;
    for (const p of params) {
      flat.set(p.data, o);
      o += p.data.length;
    }
    if (fp16) new Uint16Array(buf, headerBytes, total).set(quantizeFp16(flat));
    else new Float32Array(buf, headerBytes, total).set(flat);
    return buf;
  }

  /** Load weights from an "EVL0" binary. Validates magic + dims. */
  loadWeights(buffer: ArrayBuffer): void {
    const head = new Uint32Array(buffer, 0, 8);
    if (head[0] !== MAGIC) throw new Error("EvermindLM.loadWeights: bad magic (not an EVL0 checkpoint)");
    const version = head[1]!;
    if (
      head[2] !== this.config.vocabSize ||
      head[3] !== this.config.dModel ||
      head[4] !== this.config.numLayers ||
      head[5] !== this.config.convKernel ||
      head[6] !== this.config.hiddenDim ||
      head[7] !== this.config.numExperts * 16 + this.config.topK
    ) {
      throw new Error("EvermindLM.loadWeights: config mismatch with checkpoint");
    }
    const params = this.parameters();
    const total = params.reduce((n, p) => n + p.data.length, 0);
    const headerBytes = 32;
    const flat =
      version === 2
        ? dequantizeFp16(new Uint16Array(buffer, headerBytes, total))
        : new Float32Array(buffer.slice(headerBytes, headerBytes + total * 4));
    let o = 0;
    for (const p of params) {
      p.data.set(flat.subarray(o, o + p.data.length));
      o += p.data.length;
    }
  }
}

/** Minimal sequence trainer: AdamW over next-token cross-entropy. */
export class EvermindLMTrainer {
  private readonly adam: AdamW;
  constructor(
    private readonly model: EvermindLM,
    private readonly opts: AdamWOptions & { epochs?: number } = {},
  ) {
    this.adam = new AdamW(model, opts);
  }
  /** Train on a set of token sequences; returns per-epoch mean loss. */
  fit(sequences: number[][]): number[] {
    const epochs = this.opts.epochs ?? 1;
    const history: number[] = [];
    for (let e = 0; e < epochs; e++) {
      let total = 0;
      let n = 0;
      for (const seq of sequences) {
        if (seq.length < 2) continue;
        this.model.zeroGrad();
        total += this.model.lossAndBackward(seq);
        this.adam.step();
        n++;
      }
      history.push(n > 0 ? total / n : 0);
    }
    return history;
  }
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
