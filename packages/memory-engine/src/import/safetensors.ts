/**
 * import/safetensors.ts — read weights FROM the HF-native `.safetensors` format.
 *
 * The exact inverse of {@link ../export/safetensors.ts}: parse the 8-byte
 * little-endian u64 header length, the JSON header (name → {dtype, shape,
 * data_offsets}), then decode each tensor's raw bytes into a Float32Array.
 * F32 and F16 are supported (F16 is widened to f32 on read). This is the reader
 * half of the warm-start / weight-port path — export was one-directional before.
 */

import type { NamedTensor } from "../export/tensors.js";
import { fp16ToFloat } from "../utils/quantization.js";

interface HeaderEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

/** Decode a `.safetensors` byte buffer into named F32 tensors (header order). */
export function safetensorsToTensors(bytes: Uint8Array): NamedTensor[] {
  if (bytes.length < 8) throw new Error("safetensors: buffer too short for header length");
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true));
  const headerEnd = 8 + headerLen;
  if (headerEnd > bytes.length) throw new Error(`safetensors: header length ${headerLen} exceeds buffer`);

  const headerJson = new TextDecoder().decode(bytes.subarray(8, headerEnd));
  let header: Record<string, unknown>;
  try {
    header = JSON.parse(headerJson) as Record<string, unknown>;
  } catch (e) {
    throw new Error(`safetensors: malformed JSON header (${e instanceof Error ? e.message : String(e)})`);
  }

  const dataBase = headerEnd;
  const out: NamedTensor[] = [];
  for (const [name, raw] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const entry = raw as HeaderEntry;
    if (!entry || !Array.isArray(entry.shape) || !Array.isArray(entry.data_offsets)) {
      throw new Error(`safetensors: malformed entry for "${name}"`);
    }
    const [start, end] = entry.data_offsets;
    const numel = entry.shape.reduce((n, d) => n * d, 1);
    const seg = bytes.subarray(dataBase + start, dataBase + end);
    out.push({ name, shape: entry.shape, data: decodeTensor(seg, entry.dtype, numel, name) });
  }
  return out;
}

function decodeTensor(seg: Uint8Array, dtype: string, numel: number, name: string): Float32Array {
  const dv = new DataView(seg.buffer, seg.byteOffset, seg.byteLength);
  const data = new Float32Array(numel);
  const d = dtype.toUpperCase();
  if (d === "F32" || d === "FLOAT32") {
    if (seg.length < numel * 4) throw new Error(`safetensors: "${name}" F32 bytes ${seg.length} < ${numel * 4}`);
    for (let i = 0; i < numel; i++) data[i] = dv.getFloat32(i * 4, true);
    return data;
  }
  if (d === "F16" || d === "FLOAT16") {
    if (seg.length < numel * 2) throw new Error(`safetensors: "${name}" F16 bytes ${seg.length} < ${numel * 2}`);
    for (let i = 0; i < numel; i++) data[i] = fp16ToFloat(dv.getUint16(i * 2, true));
    return data;
  }
  throw new Error(`safetensors: unsupported dtype "${dtype}" for "${name}" (expected F32 or F16)`);
}
