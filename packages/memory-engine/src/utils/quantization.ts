/**
 * quantization.ts – FP16 and Int8 quantization utilities.
 */

export interface QuantizeInt8Result {
  data: Int8Array;
  scale: number;
}

export interface QuantizeInt8PerChannelResult {
  data: Int8Array;
  scales: Float32Array;
}

export interface MemoryEstimate {
  fp32: number;
  fp16: number;
  int8: number;
}

export function floatToFp16(val: number): number {
    const buf = new ArrayBuffer(4);
    const f32 = new Float32Array(buf);
    const u32 = new Uint32Array(buf);
    f32[0] = val;
    const bits = u32[0]!;

    const sign     = (bits >>> 31) & 0x1;
    const exponent = (bits >>> 23) & 0xFF;
    const mantissa =  bits         & 0x7FFFFF;

    if (exponent === 255) {
        return (sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0);
    }

    const expAdj = exponent - 127 + 15;

    if (expAdj >= 31) {
        return (sign << 15) | 0x7C00;
    }

    if (expAdj <= 0) {
        if (expAdj < -10) { return sign << 15; }
        const shift = 14 - expAdj;
        return (sign << 15) | ((mantissa | 0x800000) >> shift);
    }

    return (sign << 15) | (expAdj << 10) | (mantissa >> 13);
}

export function fp16ToFloat(val: number): number {
    const sign     = (val >>> 15) & 0x1;
    const exponent = (val >>> 10) & 0x1F;
    const mantissa =  val         & 0x3FF;

    if (exponent === 0) {
        const f = mantissa / 1024.0;
        return sign ? -f : f;
    }

    if (exponent === 31) {
        return sign ? -Infinity : (mantissa ? NaN : Infinity);
    }

    const expUnbiased = exponent - 15;
    const f = (1 + mantissa / 1024.0) * Math.pow(2, expUnbiased);
    return sign ? -f : f;
}

export function quantizeFp16(f32: Float32Array): Uint16Array {
    const out = new Uint16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
        out[i] = floatToFp16(f32[i]!);
    }
    return out;
}

export function dequantizeFp16(fp16: Uint16Array): Float32Array {
    const out = new Float32Array(fp16.length);
    for (let i = 0; i < fp16.length; i++) {
        out[i] = fp16ToFloat(fp16[i]!);
    }
    return out;
}

export function quantizeInt8(f32: Float32Array): QuantizeInt8Result {
    let maxAbs = 0;
    for (let i = 0; i < f32.length; i++) {
        const a = Math.abs(f32[i]!);
        if (a > maxAbs) maxAbs = a;
    }

    const scale = maxAbs / 127.0 || 1.0;
    const data  = new Int8Array(f32.length);

    for (let i = 0; i < f32.length; i++) {
        data[i] = Math.max(-128, Math.min(127, Math.round(f32[i]! / scale)));
    }

    return { data, scale };
}

export function dequantizeInt8(int8: Int8Array, scale: number): Float32Array {
    const out = new Float32Array(int8.length);
    for (let i = 0; i < int8.length; i++) {
        out[i] = int8[i]! * scale;
    }
    return out;
}

export function quantizeInt8PerChannel(f32: Float32Array, numChannels: number): QuantizeInt8PerChannelResult {
    const channelSize = f32.length / numChannels;
    const scales = new Float32Array(numChannels);
    const data   = new Int8Array(f32.length);

    for (let c = 0; c < numChannels; c++) {
        let maxAbs = 0;
        const base = c * channelSize;
        for (let j = 0; j < channelSize; j++) {
            const a = Math.abs(f32[base + j]!);
            if (a > maxAbs) maxAbs = a;
        }
        scales[c] = maxAbs / 127.0 || 1.0;
        for (let j = 0; j < channelSize; j++) {
            data[base + j] = Math.max(-128, Math.min(127,
                Math.round(f32[base + j]! / scales[c]!)
            ));
        }
    }

    return { data, scales };
}

export function dequantizeInt8PerChannel(int8: Int8Array, scales: Float32Array, numChannels: number): Float32Array {
    const channelSize = int8.length / numChannels;
    const out = new Float32Array(int8.length);

    for (let c = 0; c < numChannels; c++) {
        const base = c * channelSize;
        for (let j = 0; j < channelSize; j++) {
            out[base + j] = int8[base + j]! * scales[c]!;
        }
    }

    return out;
}

export function estimateMemory(numElements: number): MemoryEstimate {
    return {
        fp32: numElements * 4,
        fp16: numElements * 2,
        int8: numElements * 1,
    };
}
