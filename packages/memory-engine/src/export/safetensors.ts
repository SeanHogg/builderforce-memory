/**
 * export/safetensors.ts — export weights to the HF-native `.safetensors` format.
 *
 * Layout (the safetensors spec): an 8-byte little-endian u64 header length, then a
 * JSON header mapping each tensor name → {dtype, shape, data_offsets:[start,end]}
 * (offsets relative to the start of the byte buffer that follows the header), then
 * the concatenated raw tensor bytes. Round-trips losslessly in F32; F16 halves the
 * size. Loadable by `safetensors`, `transformers`, and transformers.js.
 */

import type { EvermindLM } from "../lm/evermind_lm.js";
import { namedTensors, type NamedTensor } from "./tensors.js";
import { floatToFp16 } from "../utils/quantization.js";

export interface SafetensorsOptions {
  /** Store tensors as float16 (half the size). Default false (float32). */
  fp16?: boolean;
  /** Extra `__metadata__` string entries (e.g. format provenance). */
  metadata?: Record<string, string>;
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

/** Serialise the given named tensors to a `.safetensors` byte buffer. */
export function tensorsToSafetensors(tensors: NamedTensor[], opts: SafetensorsOptions = {}): Uint8Array {
  const fp16 = opts.fp16 ?? false;
  const dtype = fp16 ? "F16" : "F32";
  const bytesPer = fp16 ? 2 : 4;

  const header: Record<string, unknown> = {};
  if (opts.metadata) header.__metadata__ = opts.metadata;

  const chunks: Uint8Array[] = [];
  let offset = 0;
  for (const t of tensors) {
    const nbytes = t.data.length * bytesPer;
    header[t.name] = { dtype, shape: t.shape, data_offsets: [offset, offset + nbytes] };
    chunks.push(tensorBytes(t.data, fp16));
    offset += nbytes;
  }

  let headerJson = JSON.stringify(header);
  // safetensors requires the header be padded to an 8-byte boundary (with spaces).
  const headerBytesRaw = new TextEncoder().encode(headerJson);
  const pad = (8 - (headerBytesRaw.length % 8)) % 8;
  if (pad > 0) headerJson += " ".repeat(pad);
  const headerBytes = new TextEncoder().encode(headerJson);

  const out = new Uint8Array(8 + headerBytes.length + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  out.set(headerBytes, 8);
  let p = 8 + headerBytes.length;
  for (const c of chunks) {
    out.set(c, p);
    p += c.length;
  }
  return out;
}

/** Export a trained LM's weights to `.safetensors`. */
export function exportSafetensors(lm: EvermindLM, opts: SafetensorsOptions = {}): Uint8Array {
  return tensorsToSafetensors(namedTensors(lm), {
    ...opts,
    metadata: { format: "pt", producer: "builderforce-memory-engine", ...(opts.metadata ?? {}) },
  });
}
