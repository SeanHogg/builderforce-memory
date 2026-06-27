/**
 * moe_model.ts — SharedExpertMoE: a shared-expert hybrid Mixture-of-Experts FFN.
 *
 * The sparsity design behind Evermind's generator. Each token is processed by:
 *   • a DENSE shared expert that is ALWAYS active (carries continuous learning;
 *     the part the online-distillation signal flows into), plus
 *   • the top-k of N routed experts, gated by a learned router and combined by a
 *     softmax over the selected experts.
 *
 *   y = SharedFFN(x) + Σ_{e ∈ topk(x)} gate_e · Expert_e(x)
 *
 * This is the DeepSeekMoE "shared-expert isolation" pattern: the dense backbone
 * resolves the online-learning attribution problem (you distil into ONE always-on
 * path), while the routed experts add web-pageable capacity (each expert's
 * weights are an independent checkpoint — see {@link SharedExpertMoE.exportExpert}
 * — so a host can stream only the experts a token activates).
 *
 * Pure-TS CPU reference (Float32Array, exact forward + backward), mirroring
 * {@link LimbicModel}'s WebGPU-or-fallback contract — the WGSL kernel path
 * (router gate + expert FFN GEMM) is a numerically-identical future acceleration.
 *
 * Activation is ReLU for an exact, unambiguous gradient in the reference path;
 * production may swap GELU/SwiGLU behind the same shapes.
 */

import { SeededRng } from "../utils/rng.js";
import { quantizeFp16, dequantizeFp16 } from "../utils/quantization.js";

export interface MoEConfig {
  /** Model (token) dimension — FFN input/output width. Default 64. */
  modelDim: number;
  /** Hidden width of each expert FFN. Default 128. */
  hiddenDim: number;
  /** Number of routed experts. Default 8. */
  numExperts: number;
  /** Experts activated per token (top-k). Default 2. Must be ≤ numExperts. */
  topK: number;
  /** Deterministic init seed for reproducible cold-start weights. */
  seed?: number;
}

export const DEFAULT_MOE_CONFIG: Required<Omit<MoEConfig, "seed">> = {
  modelDim: 64,
  hiddenDim: 128,
  numExperts: 8,
  topK: 2,
};

/** Fixed default init seed — reproducible byte-identical cold start across machines. */
export const DEFAULT_MOE_SEED = 0x4d6f4501; // "MoE\x01"

const MAGIC = 0x4d6f4530; // "MoE0"

/** A named trainable parameter tensor (flat row-major). */
export interface MoEParam {
  name: string;
  data: Float32Array;
  numel: number;
}

/** Result of routing a token: which experts fire and with what combine weights. */
export interface RouteResult {
  /** Indices of the selected top-k experts, highest router logit first. */
  experts: number[];
  /** Combine weights (softmax over the selected logits), index-aligned to `experts`. */
  gates: number[];
  /** Full softmax over ALL experts — the load-balancing signal. */
  probs: Float32Array;
}

function relu(x: number): number {
  return x > 0 ? x : 0;
}

/** A 2-layer FFN expert: y = W2·relu(W1·x + b1) + b2. */
class Expert {
  // Parameters (flat, row-major).
  w1: Float32Array; // hidden × model
  b1: Float32Array; // hidden
  w2: Float32Array; // model × hidden
  b2: Float32Array; // model
  // Gradient accumulators.
  gW1: Float32Array;
  gB1: Float32Array;
  gW2: Float32Array;
  gB2: Float32Array;

  constructor(
    private readonly modelDim: number,
    private readonly hiddenDim: number,
    gauss: (n: number, std: number) => Float32Array,
  ) {
    // He-style init for the ReLU layer; small output init so an untrained expert
    // contributes little until it has learned.
    this.w1 = gauss(hiddenDim * modelDim, Math.sqrt(2 / modelDim));
    this.b1 = new Float32Array(hiddenDim);
    this.w2 = gauss(modelDim * hiddenDim, 0.02);
    this.b2 = new Float32Array(modelDim);
    this.gW1 = new Float32Array(this.w1.length);
    this.gB1 = new Float32Array(this.b1.length);
    this.gW2 = new Float32Array(this.w2.length);
    this.gB2 = new Float32Array(this.b2.length);
  }

  /** Forward. Returns the output plus the cache needed for {@link backward}. */
  forward(x: Float32Array): { y: Float32Array; pre: Float32Array; h: Float32Array } {
    const { modelDim, hiddenDim } = this;
    const pre = new Float32Array(hiddenDim);
    const h = new Float32Array(hiddenDim);
    for (let j = 0; j < hiddenDim; j++) {
      let acc = this.b1[j]!;
      const off = j * modelDim;
      for (let i = 0; i < modelDim; i++) acc += this.w1[off + i]! * x[i]!;
      pre[j] = acc;
      h[j] = relu(acc);
    }
    const y = new Float32Array(modelDim);
    for (let d = 0; d < modelDim; d++) {
      let acc = this.b2[d]!;
      const off = d * hiddenDim;
      for (let j = 0; j < hiddenDim; j++) acc += this.w2[off + j]! * h[j]!;
      y[d] = acc;
    }
    return { y, pre, h };
  }

  /** Accumulate gradients for one token given dL/dy. Returns dL/dx. */
  backward(dy: Float32Array, x: Float32Array, pre: Float32Array, h: Float32Array): Float32Array {
    const { modelDim, hiddenDim } = this;
    const dh = new Float32Array(hiddenDim);
    for (let d = 0; d < modelDim; d++) {
      const dyd = dy[d]!;
      this.gB2[d] = this.gB2[d]! + dyd;
      const off = d * hiddenDim;
      for (let j = 0; j < hiddenDim; j++) {
        this.gW2[off + j] = this.gW2[off + j]! + dyd * h[j]!;
        dh[j] = dh[j]! + dyd * this.w2[off + j]!;
      }
    }
    const dx = new Float32Array(modelDim);
    for (let j = 0; j < hiddenDim; j++) {
      const dpre = pre[j]! > 0 ? dh[j]! : 0; // relu'
      this.gB1[j] = this.gB1[j]! + dpre;
      const off = j * modelDim;
      for (let i = 0; i < modelDim; i++) {
        this.gW1[off + i] = this.gW1[off + i]! + dpre * x[i]!;
        dx[i] = dx[i]! + dpre * this.w1[off + i]!;
      }
    }
    return dx;
  }

  params(): Float32Array[] {
    return [this.w1, this.b1, this.w2, this.b2];
  }
  grads(): Float32Array[] {
    return [this.gW1, this.gB1, this.gW2, this.gB2];
  }
}

/** Per-token forward intermediates retained for the backward pass. */
interface MoECache {
  x: Float32Array;
  route: RouteResult;
  sharedPre: Float32Array;
  sharedH: Float32Array;
  expertOut: Float32Array[]; // per selected expert, index-aligned to route.experts
  expertPre: Float32Array[];
  expertH: Float32Array[];
}

/**
 * Accumulates router statistics over a batch to compute the load-balancing
 * auxiliary loss `E · Σ_e f_e · P_e` (Switch/GShard). Minimised (→ near 1) when
 * dispatch is uniform; large (→ near E) when the router collapses onto few
 * experts. Add it to the task loss with a small coefficient to keep experts busy.
 */
export class LoadBalanceAccumulator {
  private readonly counts: Float32Array;
  private readonly probSum: Float32Array;
  private tokens = 0;
  constructor(private readonly numExperts: number) {
    this.counts = new Float32Array(numExperts);
    this.probSum = new Float32Array(numExperts);
  }
  observe(route: RouteResult): void {
    this.tokens++;
    for (const e of route.experts) this.counts[e] = this.counts[e]! + 1;
    for (let e = 0; e < this.numExperts; e++) this.probSum[e] = this.probSum[e]! + route.probs[e]!;
  }
  /** The load-balance loss over everything observed so far (0 if no tokens). */
  loss(): number {
    if (this.tokens === 0) return 0;
    const E = this.numExperts;
    const dispatched = this.counts.reduce((a, b) => a + b, 0) || 1; // = tokens·topK
    let sum = 0;
    for (let e = 0; e < E; e++) {
      const f = this.counts[e]! / dispatched; // fraction of dispatches to e
      const p = this.probSum[e]! / this.tokens; // mean router prob for e
      sum += f * p;
    }
    return E * sum;
  }
}

export class SharedExpertMoE {
  readonly config: Required<Omit<MoEConfig, "seed">>;

  /** Router weights: numExperts × modelDim (no bias). */
  wr: Float32Array;
  private gWr: Float32Array;

  private readonly shared: Expert;
  private readonly experts: Expert[];

  constructor(config: Partial<MoEConfig> = {}) {
    const cfg = { ...DEFAULT_MOE_CONFIG, ...config };
    if (cfg.topK > cfg.numExperts) {
      throw new Error(`MoE topK (${cfg.topK}) must be ≤ numExperts (${cfg.numExperts})`);
    }
    if (cfg.topK < 1) throw new Error(`MoE topK must be ≥ 1 (got ${cfg.topK})`);
    this.config = cfg;

    const rng = new SeededRng(((config.seed ?? DEFAULT_MOE_SEED) >>> 0) || 1);
    const gauss = (n: number, std: number): Float32Array => {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const u1 = Math.max(rng.next(), 1e-12);
        const u2 = rng.next();
        a[i] = std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      return a;
    };

    this.wr = gauss(cfg.numExperts * cfg.modelDim, 0.02);
    this.gWr = new Float32Array(this.wr.length);
    this.shared = new Expert(cfg.modelDim, cfg.hiddenDim, gauss);
    this.experts = Array.from({ length: cfg.numExperts }, () => new Expert(cfg.modelDim, cfg.hiddenDim, gauss));
  }

  /** Route a token: router logits → top-k → combine gates + full softmax probs. */
  route(x: Float32Array): RouteResult {
    const { numExperts, topK, modelDim } = this.config;
    const logits = new Float32Array(numExperts);
    for (let e = 0; e < numExperts; e++) {
      let acc = 0;
      const off = e * modelDim;
      for (let i = 0; i < modelDim; i++) acc += this.wr[off + i]! * x[i]!;
      logits[e] = acc;
    }
    // Full softmax over all experts (load-balancing signal).
    const probs = softmax(logits);
    // Top-k experts by logit.
    const order = Array.from({ length: numExperts }, (_, e) => e).sort((a, b) => logits[b]! - logits[a]!);
    const experts = order.slice(0, topK);
    // Combine gates = softmax over ONLY the selected logits.
    const selLogits = experts.map((e) => logits[e]!);
    const selSoft = softmax(Float32Array.from(selLogits));
    return { experts, gates: Array.from(selSoft), probs };
  }

  /** Forward a single token. Returns the output and a cache for {@link backward}. */
  forward(input: ArrayLike<number>): { output: Float32Array; route: RouteResult; cache: MoECache } {
    const { modelDim } = this.config;
    const x = Float32Array.from({ length: modelDim }, (_, i) => input[i] ?? 0);
    const route = this.route(x);

    const s = this.shared.forward(x);
    const output = Float32Array.from(s.y);

    const expertOut: Float32Array[] = [];
    const expertPre: Float32Array[] = [];
    const expertH: Float32Array[] = [];
    for (let m = 0; m < route.experts.length; m++) {
      const e = this.experts[route.experts[m]!]!;
      const r = e.forward(x);
      const g = route.gates[m]!;
      for (let d = 0; d < modelDim; d++) output[d] = output[d]! + g * r.y[d]!;
      expertOut.push(r.y);
      expertPre.push(r.pre);
      expertH.push(r.h);
    }

    return {
      output,
      route,
      cache: { x, route, sharedPre: s.pre, sharedH: s.h, expertOut, expertPre, expertH },
    };
  }

  /**
   * Accumulate gradients for one token given dL/d(output). Trains the shared
   * expert, the selected routed experts, and the router (so it learns to weight
   * the experts that reduce loss). Call {@link zeroGrad} before a batch and apply
   * an optimiser after. Load balancing is a separate signal (see
   * {@link LoadBalanceAccumulator}).
   */
  backward(dOutput: ArrayLike<number>, cache: MoECache): void {
    const { modelDim } = this.config;
    const dOut = Float32Array.from({ length: modelDim }, (_, d) => dOutput[d] ?? 0);

    // Shared expert (always active) sees the full upstream gradient.
    this.shared.backward(dOut, cache.x, cache.sharedPre, cache.sharedH);

    // Routed experts: each scaled by its gate; collect dL/dgate for the router.
    const k = cache.route.experts.length;
    const dGate = new Float32Array(k);
    for (let m = 0; m < k; m++) {
      const g = cache.route.gates[m]!;
      const scaled = new Float32Array(modelDim);
      let dg = 0;
      for (let d = 0; d < modelDim; d++) {
        scaled[d] = g * dOut[d]!;
        dg += dOut[d]! * cache.expertOut[m]![d]!;
      }
      this.experts[cache.route.experts[m]!]!.backward(scaled, cache.x, cache.expertPre[m]!, cache.expertH[m]!);
      dGate[m] = dg;
    }

    // Router: gates = softmax(selected logits). Backprop dGate through the
    // softmax Jacobian to the selected logits, then to Wr.
    const gates = cache.route.gates;
    let dot = 0;
    for (let m = 0; m < k; m++) dot += gates[m]! * dGate[m]!;
    for (let m = 0; m < k; m++) {
      const dLogit = gates[m]! * (dGate[m]! - dot);
      const e = cache.route.experts[m]!;
      const off = e * modelDim;
      for (let i = 0; i < modelDim; i++) this.gWr[off + i] = this.gWr[off + i]! + dLogit * cache.x[i]!;
    }
  }

  // ── Parameters / checkpoint ────────────────────────────────────────────────

  /** All trainable parameters in canonical order: router, shared, then experts. */
  parameters(): MoEParam[] {
    const out: MoEParam[] = [{ name: "wr", data: this.wr, numel: this.wr.length }];
    const push = (prefix: string, e: Expert) => {
      const names = ["w1", "b1", "w2", "b2"];
      e.params().forEach((p, i) => out.push({ name: `${prefix}.${names[i]}`, data: p, numel: p.length }));
    };
    push("shared", this.shared);
    this.experts.forEach((e, idx) => push(`expert${idx}`, e));
    return out;
  }

  /** Gradient buffers, index-aligned with {@link parameters}. */
  gradients(): MoEParam[] {
    const out: MoEParam[] = [{ name: "wr", data: this.gWr, numel: this.gWr.length }];
    const push = (prefix: string, e: Expert) => {
      const names = ["w1", "b1", "w2", "b2"];
      e.grads().forEach((g, i) => out.push({ name: `${prefix}.${names[i]}`, data: g, numel: g.length }));
    };
    push("shared", this.shared);
    this.experts.forEach((e, idx) => push(`expert${idx}`, e));
    return out;
  }

  zeroGrad(): void {
    for (const g of this.gradients()) g.data.fill(0);
  }

  /** One routed expert's weights as a standalone checkpoint (the web-paging unit). */
  exportExpert(index: number): MoEParam[] {
    const e = this.experts[index];
    if (!e) throw new Error(`exportExpert: index ${index} out of range (0..${this.config.numExperts - 1})`);
    const names = ["w1", "b1", "w2", "b2"];
    return e.params().map((p, i) => ({ name: names[i]!, data: p, numel: p.length }));
  }

  /**
   * Serialise all weights to a compact "MoE0" binary. Layout: magic, version,
   * [modelDim, hiddenDim, numExperts, topK], then params in {@link parameters}
   * order. fp16 (v2) halves the size; f32 (v1) is exact.
   */
  exportWeights(opts: { fp16?: boolean } = {}): ArrayBuffer {
    const fp16 = opts.fp16 ?? false;
    const params = this.parameters();
    const total = params.reduce((n, p) => n + p.numel, 0);
    const headerEls = 6; // magic, version, modelDim, hiddenDim, numExperts, topK
    const headerBytes = headerEls * 4;
    const buf = new ArrayBuffer(headerBytes + (fp16 ? total * 2 : total * 4));
    const head = new Uint32Array(buf, 0, headerEls);
    head[0] = MAGIC;
    head[1] = fp16 ? 2 : 1;
    head[2] = this.config.modelDim;
    head[3] = this.config.hiddenDim;
    head[4] = this.config.numExperts;
    head[5] = this.config.topK;

    const flat = new Float32Array(total);
    let o = 0;
    for (const p of params) {
      flat.set(p.data, o);
      o += p.numel;
    }
    if (fp16) new Uint16Array(buf, headerBytes, total).set(quantizeFp16(flat));
    else new Float32Array(buf, headerBytes, total).set(flat);
    return buf;
  }

  /** Load weights from an "MoE0" binary. Validates magic + dims. */
  loadWeights(buffer: ArrayBuffer): void {
    const head = new Uint32Array(buffer, 0, 6);
    if (head[0] !== MAGIC) throw new Error("SharedExpertMoE.loadWeights: bad magic (not an MoE0 checkpoint)");
    const [, version, modelDim, hiddenDim, numExperts, topK] = head;
    if (
      modelDim !== this.config.modelDim ||
      hiddenDim !== this.config.hiddenDim ||
      numExperts !== this.config.numExperts ||
      topK !== this.config.topK
    ) {
      throw new Error("SharedExpertMoE.loadWeights: config mismatch with checkpoint");
    }
    const params = this.parameters();
    const total = params.reduce((n, p) => n + p.numel, 0);
    const headerBytes = 24;
    const flat =
      version === 2
        ? dequantizeFp16(new Uint16Array(buffer, headerBytes, total))
        : new Float32Array(buffer.slice(headerBytes, headerBytes + total * 4));
    let o = 0;
    for (const p of params) {
      p.data.set(flat.subarray(o, o + p.numel));
      o += p.numel;
    }
  }
}

/** Numerically-stable softmax over a flat array. */
function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i]! > max) max = logits[i]!;
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i]! - max);
    sum += out[i]!;
  }
  for (let i = 0; i < logits.length; i++) out[i] = out[i]! / sum;
  return out;
}
