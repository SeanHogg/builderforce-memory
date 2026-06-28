/**
 * export/protobuf.ts — a minimal, dependency-free protobuf (proto3) wire writer.
 *
 * Just enough of the encoding to emit an ONNX `ModelProto` (see {@link ./onnx}):
 * varint, fixed32 (for float tensors), and length-delimited (string / bytes /
 * nested message) fields. Keeping it tiny and self-contained preserves the
 * engine's zero-runtime-dependency guarantee — no `protobufjs`, no `onnx` proto.
 *
 * Only non-negative integers are emitted (all ONNX ids here are ≥ 0), so the
 * varint path uses plain division and stays correct up to 2^53.
 */

const WIRE_VARINT = 0;
const WIRE_FIXED32 = 5;
const WIRE_LEN = 2;

/** Accumulates protobuf-encoded bytes for one message. */
export class ProtoWriter {
  private readonly parts: number[] = [];

  private byte(b: number): void {
    this.parts.push(b & 0xff);
  }

  private rawVarint(value: number): void {
    if (value < 0 || !Number.isFinite(value)) throw new Error(`ProtoWriter: bad varint ${value}`);
    let v = value;
    while (v >= 0x80) {
      this.byte((v % 0x80) | 0x80);
      v = Math.floor(v / 0x80);
    }
    this.byte(v);
  }

  private tag(field: number, wire: number): void {
    this.rawVarint(field * 8 + wire);
  }

  private rawBytes(bytes: Uint8Array): void {
    for (let i = 0; i < bytes.length; i++) this.byte(bytes[i]!);
  }

  /** A varint field (int32 / int64 / bool / enum). */
  varint(field: number, value: number): void {
    this.tag(field, WIRE_VARINT);
    this.rawVarint(value);
  }

  /** A 32-bit IEEE-754 float field (wire type fixed32, little-endian). */
  float(field: number, value: number): void {
    this.tag(field, WIRE_FIXED32);
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    this.rawBytes(new Uint8Array(buf));
  }

  /** A length-delimited raw-bytes field. */
  bytes(field: number, data: Uint8Array): void {
    this.tag(field, WIRE_LEN);
    this.rawVarint(data.length);
    this.rawBytes(data);
  }

  /** A length-delimited UTF-8 string field. */
  string(field: number, value: string): void {
    this.bytes(field, new TextEncoder().encode(value));
  }

  /** A nested-message field (the sub-message encoded as length-delimited bytes). */
  message(field: number, sub: ProtoWriter): void {
    this.bytes(field, sub.finish());
  }

  /** The encoded bytes of this message. */
  finish(): Uint8Array {
    return Uint8Array.from(this.parts);
  }
}

/** Pack a Float32Array into little-endian raw bytes (for TensorProto.raw_data). */
export function float32ToBytes(data: Float32Array): Uint8Array {
  const out = new Uint8Array(data.length * 4);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < data.length; i++) dv.setFloat32(i * 4, data[i]!, true);
  return out;
}
