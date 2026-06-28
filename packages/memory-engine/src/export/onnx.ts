/**
 * export/onnx.ts — export a trained {@link EvermindLM} to a runnable ONNX graph.
 *
 * Produces a standard causal-LM ONNX `ModelProto`:  input `input_ids` [batch,seq]
 * (int64) → output `logits` [batch,seq,vocab] (float). No KV cache — the model is
 * stateless and recomputes the full context each step, so a host (onnxruntime-web,
 * onnxruntime-node, or transformers.js driving the generation loop) re-runs the
 * whole sequence per token. The graph reproduces the CPU reference forward
 * EXACTLY:
 *
 *   • embedding  → Gather(emb, input_ids)
 *   • RMSNorm    → Mul/ReduceMean/Add/Sqrt/Div/Mul (eps = 1e-5)
 *   • causal     → Transpose → Conv(group=D, pads=[K-1,0], kernel reversed) → Transpose
 *     depthwise conv (so out[t] = Σⱼ ker[j]·x[t−j], matching the reference tap order)
 *   • MoE        → router MatMul → TopK → Softmax → ScatterElements (dense top-k mask)
 *                  → shared FFN + Σₑ combineₑ · expertₑ(x)   (numerically identical to
 *                  the sparse reference: non-selected experts get combine 0)
 *   • tied head  → MatMul(x, embᵀ)
 *
 * Opset 18 / IR v8. Weights are emitted as raw little-endian f32 initializers that
 * are byte-identical to {@link namedTensors} (matmul weights are transposed by a
 * graph node at runtime, not duplicated on disk).
 *
 * Zero dependencies — the protobuf is hand-encoded by {@link ./protobuf}.
 */

import type { EvermindLM } from "../lm/evermind_lm.js";
import { archOf, namedTensors, type EvermindArch, type NamedTensor } from "./tensors.js";
import { ProtoWriter, float32ToBytes } from "./protobuf.js";

const OPSET = 18;
const IR_VERSION = 8;
const ELEM_FLOAT = 1;
const ELEM_INT64 = 7;

// AttributeProto.AttributeType
const ATTR_FLOAT = 1;
const ATTR_INT = 2;
const ATTR_STRING = 3;
const ATTR_INTS = 7;

type Attr =
  | { kind: "i"; name: string; value: number }
  | { kind: "ints"; name: string; value: number[] }
  | { kind: "f"; name: string; value: number }
  | { kind: "s"; name: string; value: string };

interface NodeDef {
  op: string;
  inputs: string[];
  outputs: string[];
  name: string;
  attrs: Attr[];
}

interface InitDef {
  name: string;
  dims: number[];
  dataType: number;
  raw?: Uint8Array;
  int64?: number[];
}

/** Accumulates a single ONNX GraphProto (nodes + initializers) with auto-naming. */
class GraphBuilder {
  readonly nodes: NodeDef[] = [];
  readonly inits: InitDef[] = [];
  private uid = 0;

  /** A fresh intermediate value name. */
  tmp(hint: string): string {
    return `${hint}_${this.uid++}`;
  }

  initFloat(name: string, dims: number[], data: Float32Array): string {
    this.inits.push({ name, dims, dataType: ELEM_FLOAT, raw: float32ToBytes(data) });
    return name;
  }

  initInt64(name: string, dims: number[], values: number[]): string {
    this.inits.push({ name, dims, dataType: ELEM_INT64, int64: values });
    return name;
  }

  node(op: string, inputs: string[], outputs: string[], attrs: Attr[] = []): string[] {
    this.nodes.push({ op, inputs, outputs, name: `${op}_${this.uid++}`, attrs });
    return outputs;
  }

  /** Single-output convenience: returns the output value name. */
  op(op: string, inputs: string[], hint: string, attrs: Attr[] = []): string {
    const out = this.tmp(hint);
    this.node(op, inputs, [out], attrs);
    return out;
  }
}

/** Transpose a 2-D row-major matrix [rows, cols] → [cols, rows]. */
function transpose2d(data: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(data.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = data[r * cols + c]!;
  return out;
}

/** RMSNorm(x, gain) as graph ops. Returns the normalised value name. */
function rmsNorm(g: GraphBuilder, x: string, gainInit: string, eps: string, axes: string): string {
  const sq = g.op("Mul", [x, x], "rms_sq");
  const ms = g.op("ReduceMean", [sq, axes], "rms_mean", [{ kind: "i", name: "keepdims", value: 1 }]);
  const msE = g.op("Add", [ms, eps], "rms_eps");
  const r = g.op("Sqrt", [msE], "rms_r");
  const div = g.op("Div", [x, r], "rms_div");
  return g.op("Mul", [div, gainInit], "rms_y");
}

/** A 2-layer FFN expert: y = (relu(x·W1ᵀ + b1))·W2ᵀ + b2. Weights transposed at runtime. */
function ffn(
  g: GraphBuilder,
  x: string,
  w1: string,
  b1: string,
  w2: string,
  b2: string,
): string {
  const w1t = g.op("Transpose", [w1], "w1t", [{ kind: "ints", name: "perm", value: [1, 0] }]);
  const pre = g.op("Add", [g.op("MatMul", [x, w1t], "ffn_mm1"), b1], "ffn_pre");
  const h = g.op("Relu", [pre], "ffn_h");
  const w2t = g.op("Transpose", [w2], "w2t", [{ kind: "ints", name: "perm", value: [1, 0] }]);
  return g.op("Add", [g.op("MatMul", [h, w2t], "ffn_mm2"), b2], "ffn_y");
}

/** Build the ONNX bytes for a trained LM. */
export function exportOnnx(lm: EvermindLM, opts: { producerName?: string; producerVersion?: string } = {}): Uint8Array {
  const arch = archOf(lm);
  const tensors = namedTensors(lm);
  const byName = new Map<string, NamedTensor>(tensors.map((t) => [t.name, t]));
  const get = (name: string): NamedTensor => {
    const t = byName.get(name);
    if (!t) throw new Error(`export/onnx: missing tensor ${name}`);
    return t;
  };

  const g = new GraphBuilder();
  const { vocabSize: V, dModel: D, numLayers: L, convKernel: K, numExperts: E, topK } = arch;

  // Shared scalar/axis initializers.
  const emb = g.initFloat("token_embedding.weight", [V, D], get("token_embedding.weight").data);
  const eps = g.initFloat("rms_eps", [1], new Float32Array([1e-5]));
  const axesLast = g.initInt64("axes_last", [1], [2]); // feature axis of [B,S,*]
  const topkK = g.initInt64("topk_k", [1], [topK]);

  // input_ids → embedded [B,S,D]
  let x = g.op("Gather", [emb, "input_ids"], "embedded", [{ kind: "i", name: "axis", value: 0 }]);

  for (let l = 0; l < L; l++) {
    const p = `layers.${l}`;

    // ── Causal depthwise conv sub-block ──────────────────────────────────────
    const nConv = g.initFloat(`${p}.norm_conv.weight`, [D], get(`${p}.norm_conv.weight`).data);
    const normedC = rmsNorm(g, x, nConv, eps, axesLast);
    // Conv weight: reference ker[c,j] → ONNX W[c,0,K-1-j] (reverse taps), shape [D,1,K].
    const ker = get(`${p}.conv.weight`).data; // [D,K]
    const cw = new Float32Array(D * K);
    for (let c = 0; c < D; c++) for (let j = 0; j < K; j++) cw[c * K + (K - 1 - j)] = ker[c * K + j]!;
    const convW = g.initFloat(`${p}.conv.onnx_weight`, [D, 1, K], cw);
    const xt = g.op("Transpose", [normedC], "conv_in", [{ kind: "ints", name: "perm", value: [0, 2, 1] }]); // [B,D,S]
    const cOut = g.op("Conv", [xt, convW], "conv_out", [
      { kind: "i", name: "group", value: D },
      { kind: "ints", name: "kernel_shape", value: [K] },
      { kind: "ints", name: "pads", value: [K - 1, 0] },
      { kind: "ints", name: "strides", value: [1] },
      { kind: "ints", name: "dilations", value: [1] },
    ]);
    const cOutT = g.op("Transpose", [cOut], "conv_back", [{ kind: "ints", name: "perm", value: [0, 2, 1] }]); // [B,S,D]
    const afterConv = g.op("Add", [x, cOutT], "after_conv");

    // ── MoE channel-mixer sub-block ──────────────────────────────────────────
    const nMoe = g.initFloat(`${p}.norm_moe.weight`, [D], get(`${p}.norm_moe.weight`).data);
    const normedM = rmsNorm(g, afterConv, nMoe, eps, axesLast);

    const wr = g.initFloat(`${p}.moe.router.weight`, [E, D], get(`${p}.moe.router.weight`).data);
    const wrT = g.op("Transpose", [wr], "router_t", [{ kind: "ints", name: "perm", value: [1, 0] }]); // [D,E]
    const routerLogits = g.op("MatMul", [normedM, wrT], "router_logits"); // [B,S,E]
    const topkV = g.tmp("topk_v");
    const topkI = g.tmp("topk_i");
    g.node("TopK", [routerLogits, topkK], [topkV, topkI], [
      { kind: "i", name: "axis", value: 2 },
      { kind: "i", name: "largest", value: 1 },
      { kind: "i", name: "sorted", value: 1 },
    ]);
    const gates = g.op("Softmax", [topkV], "gates", [{ kind: "i", name: "axis", value: 2 }]); // [B,S,k]
    const zeros = g.op("Sub", [routerLogits, routerLogits], "zeros"); // [B,S,E]
    const combine = g.op("ScatterElements", [zeros, topkI, gates], "combine", [{ kind: "i", name: "axis", value: 2 }]); // [B,S,E]

    // shared expert (always active)
    const sharedY = ffn(
      g,
      normedM,
      g.initFloat(`${p}.moe.shared.w1`, [arch.hiddenDim, D], get(`${p}.moe.shared.w1`).data),
      g.initFloat(`${p}.moe.shared.b1`, [arch.hiddenDim], get(`${p}.moe.shared.b1`).data),
      g.initFloat(`${p}.moe.shared.w2`, [D, arch.hiddenDim], get(`${p}.moe.shared.w2`).data),
      g.initFloat(`${p}.moe.shared.b2`, [D], get(`${p}.moe.shared.b2`).data),
    );

    // per-expert combine weight: split [B,S,E] → E × [B,S,1]
    const splitOuts = Array.from({ length: E }, (_, e) => g.tmp(`combine_${e}`));
    g.node("Split", [combine], splitOuts, [
      { kind: "i", name: "axis", value: 2 },
      { kind: "i", name: "num_outputs", value: E },
    ]);

    let moeOut = sharedY;
    for (let e = 0; e < E; e++) {
      const ep = `${p}.moe.experts.${e}`;
      const ye = ffn(
        g,
        normedM,
        g.initFloat(`${ep}.w1`, [arch.hiddenDim, D], get(`${ep}.w1`).data),
        g.initFloat(`${ep}.b1`, [arch.hiddenDim], get(`${ep}.b1`).data),
        g.initFloat(`${ep}.w2`, [D, arch.hiddenDim], get(`${ep}.w2`).data),
        g.initFloat(`${ep}.b2`, [D], get(`${ep}.b2`).data),
      );
      const weighted = g.op("Mul", [splitOuts[e]!, ye], "expert_w"); // [B,S,1]·[B,S,D] broadcast
      moeOut = g.op("Add", [moeOut, weighted], "moe_acc");
    }

    x = g.op("Add", [afterConv, moeOut], "after_moe");
  }

  // Tied head: logits = x · embᵀ.
  const embT = g.op("Transpose", [emb], "emb_t", [{ kind: "ints", name: "perm", value: [1, 0] }]); // [D,V]
  g.node("MatMul", [x, embT], ["logits"]);

  return encodeModel(g, arch, opts);
}

// ── protobuf encoding ──────────────────────────────────────────────────────────

function attrProto(a: Attr): ProtoWriter {
  const w = new ProtoWriter();
  w.string(1, a.name);
  switch (a.kind) {
    case "i":
      w.varint(20, ATTR_INT);
      w.varint(3, a.value);
      break;
    case "ints":
      w.varint(20, ATTR_INTS);
      for (const v of a.value) w.varint(8, v); // repeated (unpacked) int64
      break;
    case "f":
      w.varint(20, ATTR_FLOAT);
      w.float(2, a.value);
      break;
    case "s":
      w.varint(20, ATTR_STRING);
      w.string(4, a.value);
      break;
  }
  return w;
}

function nodeProto(n: NodeDef): ProtoWriter {
  const w = new ProtoWriter();
  for (const i of n.inputs) w.string(1, i);
  for (const o of n.outputs) w.string(2, o);
  w.string(3, n.name);
  w.string(4, n.op);
  for (const a of n.attrs) w.message(5, attrProto(a));
  return w;
}

function tensorProto(t: InitDef): ProtoWriter {
  const w = new ProtoWriter();
  for (const d of t.dims) w.varint(1, d); // repeated int64 dims
  w.varint(2, t.dataType);
  w.string(8, t.name);
  if (t.int64) for (const v of t.int64) w.varint(7, v); // int64_data (unpacked)
  if (t.raw) w.bytes(9, t.raw); // raw_data (TensorProto field 9)
  return w;
}

function dimParam(name: string): ProtoWriter {
  const w = new ProtoWriter();
  w.string(2, name); // Dimension.dim_param
  return w;
}
function dimValue(v: number): ProtoWriter {
  const w = new ProtoWriter();
  w.varint(1, v); // Dimension.dim_value
  return w;
}

/** ValueInfoProto for a tensor of the given element type and (symbolic) dims. */
function valueInfo(name: string, elemType: number, dims: (number | string)[]): ProtoWriter {
  const shape = new ProtoWriter();
  for (const d of dims) shape.message(1, typeof d === "string" ? dimParam(d) : dimValue(d));
  const tensorType = new ProtoWriter();
  tensorType.varint(1, elemType); // elem_type
  tensorType.message(2, shape); // shape
  const type = new ProtoWriter();
  type.message(1, tensorType); // tensor_type
  const vi = new ProtoWriter();
  vi.string(1, name);
  vi.message(2, type);
  return vi;
}

function encodeModel(
  g: GraphBuilder,
  arch: EvermindArch,
  opts: { producerName?: string; producerVersion?: string },
): Uint8Array {
  const graph = new ProtoWriter();
  for (const n of g.nodes) graph.message(1, nodeProto(n));
  graph.string(2, "evermind");
  for (const t of g.inits) graph.message(5, tensorProto(t));
  graph.message(11, valueInfo("input_ids", ELEM_INT64, ["batch", "seq"]));
  graph.message(12, valueInfo("logits", ELEM_FLOAT, ["batch", "seq", arch.vocabSize]));

  const opset = new ProtoWriter();
  opset.string(1, ""); // default domain
  opset.varint(2, OPSET);

  const model = new ProtoWriter();
  model.varint(1, IR_VERSION);
  model.string(2, opts.producerName ?? "builderforce-memory-engine");
  model.string(3, opts.producerVersion ?? "evermind");
  model.message(7, graph);
  model.message(8, opset);
  return model.finish();
}
