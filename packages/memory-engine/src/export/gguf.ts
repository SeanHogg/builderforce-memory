/**
 * export/gguf.ts — export weights to a GGUF v3 container (the llama.cpp format).
 *
 * Emits a spec-valid GGUF file: magic + version, KV metadata (architecture, dims,
 * expert counts…), aligned tensor-info table, then aligned tensor data. ggml
 * stores dimensions innermost-first, so row-major shapes are reversed here.
 *
 * NOTE: GGUF is a *container*. llama.cpp can only RUN a file whose
 * `general.architecture` it has a graph for; "evermind" is custom, so this file
 * is a valid, inspectable artifact (gguf tooling reads it) but won't execute in
 * stock llama.cpp until the architecture is upstreamed. It exists so the same
 * weights can travel through GGUF-native tooling. For an executable export use
 * {@link ./onnx} or {@link ./safetensors}.
 */

import type { EvermindLM } from "../lm/evermind_lm.js";
import { archOf, namedTensors, type NamedTensor } from "./tensors.js";
import { floatToFp16 } from "../utils/quantization.js";

const GGUF_MAGIC = 0x46554747; // "GGUF" little-endian
const GGUF_VERSION = 3;
const ALIGNMENT = 32;

// GGUF metadata value types
const T_UINT32 = 4;
const T_STRING = 8;
// ggml tensor types
const GGML_F32 = 0;
const GGML_F16 = 1;

export interface GgufOptions {
  name?: string;
  /** Store tensors as float16 (ggml type F16). Default false (F32). */
  fp16?: boolean;
}

/** A growable little-endian byte buffer. */
class ByteBuf {
  private buf = new Uint8Array(1024);
  private len = 0;
  private ensure(n: number): void {
    if (this.len + n <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.len + n) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.len));
    this.buf = next;
  }
  u32(v: number): void {
    this.ensure(4);
    new DataView(this.buf.buffer).setUint32(this.len, v, true);
    this.len += 4;
  }
  u64(v: number): void {
    this.ensure(8);
    new DataView(this.buf.buffer).setBigUint64(this.len, BigInt(v), true);
    this.len += 8;
  }
  raw(bytes: Uint8Array): void {
    this.ensure(bytes.length);
    this.buf.set(bytes, this.len);
    this.len += bytes.length;
  }
  /** Pad with zeros until the length is a multiple of `align`. */
  alignTo(align: number): void {
    const pad = (align - (this.len % align)) % align;
    if (pad > 0) {
      this.ensure(pad);
      this.len += pad;
    }
  }
  string(s: string): void {
    const bytes = new TextEncoder().encode(s);
    this.u64(bytes.length);
    this.raw(bytes);
  }
  get length(): number {
    return this.len;
  }
  bytes(): Uint8Array {
    return this.buf.subarray(0, this.len);
  }
}

function tensorBytes(data: Float32Array, fp16: boolean): Uint8Array {
  if (!fp16) {
    const out = new Uint8Array(data.length * 4);
    const dv = new DataView(out.buffer);
    for (let i = 0; i < data.length; i++) dv.setFloat32(i * 4, data[i]!, true);
    return out;
  }
  const out = new Uint8Array(data.length * 2);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < data.length; i++) dv.setUint16(i * 2, floatToFp16(data[i]!), true);
  return out;
}

/** Export a trained LM to GGUF bytes. */
export function exportGguf(lm: EvermindLM, opts: GgufOptions = {}): Uint8Array {
  const fp16 = opts.fp16 ?? false;
  const a = archOf(lm);
  const tensors = namedTensors(lm);
  const ggmlType = fp16 ? GGML_F16 : GGML_F32;
  const bytesPer = fp16 ? 2 : 4;

  const kvU32: [string, number][] = [
    ["evermind.vocab_size", a.vocabSize],
    ["evermind.embedding_length", a.dModel],
    ["evermind.block_count", a.numLayers],
    ["evermind.conv_kernel", a.convKernel],
    ["evermind.feed_forward_length", a.hiddenDim],
    ["evermind.expert_count", a.numExperts],
    ["evermind.expert_used_count", a.topK],
    ["general.alignment", ALIGNMENT],
  ];
  const kvStr: [string, string][] = [
    ["general.architecture", "evermind"],
    ["general.name", opts.name ?? "Evermind"],
  ];

  // Header: magic, version, tensor_count, metadata_kv_count, KVs.
  const head = new ByteBuf();
  head.u32(GGUF_MAGIC);
  head.u32(GGUF_VERSION);
  head.u64(tensors.length);
  head.u64(kvU32.length + kvStr.length);
  for (const [k, v] of kvStr) {
    head.string(k);
    head.u32(T_STRING);
    head.string(v);
  }
  for (const [k, v] of kvU32) {
    head.string(k);
    head.u32(T_UINT32);
    head.u32(v);
  }

  // Tensor-info table. Offsets are relative to the start of the (aligned) tensor
  // data section; each tensor starts at an ALIGNMENT boundary within it.
  const dims = (t: NamedTensor): number[] => [...t.shape].reverse(); // ggml: innermost first
  let dataOffset = 0;
  const offsets: number[] = [];
  for (const t of tensors) {
    offsets.push(dataOffset);
    const nbytes = t.data.length * bytesPer;
    dataOffset += nbytes;
    dataOffset += (ALIGNMENT - (dataOffset % ALIGNMENT)) % ALIGNMENT;
  }
  tensors.forEach((t, i) => {
    head.string(t.name);
    const d = dims(t);
    head.u32(d.length);
    for (const dim of d) head.u64(dim);
    head.u32(ggmlType);
    head.u64(offsets[i]!);
  });

  // Pad header to the alignment boundary, then write aligned tensor data.
  head.alignTo(ALIGNMENT);
  const dataStart = head.length;
  for (let i = 0; i < tensors.length; i++) {
    // Re-establish alignment relative to the data section start.
    const want = dataStart + offsets[i]!;
    while (head.length < want) head.raw(new Uint8Array(1));
    head.raw(tensorBytes(tensors[i]!.data, fp16));
  }
  return head.bytes();
}
