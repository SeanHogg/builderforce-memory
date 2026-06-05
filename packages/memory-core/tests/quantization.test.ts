/**
 * tests/quantization.test.ts
 * Unit tests for the quantization utilities (no GPU required).
 */

import {
    floatToFp16,
    fp16ToFloat,
    quantizeFp16,
    dequantizeFp16,
    quantizeInt8,
    dequantizeInt8,
    quantizeInt8PerChannel,
    dequantizeInt8PerChannel,
    estimateMemory,
} from '../src/utils/quantization';

// ── FP16 round-trip ───────────────────────────────────────────────────────────

test('floatToFp16 and fp16ToFloat are inverse for ordinary values', () => {
    const cases = [0.0, 1.0, -1.0, 0.5, -0.5, 3.14, -100.0, 0.001];
    for (const v of cases) {
        const roundTripped = fp16ToFloat(floatToFp16(v));
        // FP16 has ~3 decimal digits of precision
        expect(Math.abs(roundTripped - v)).toBeLessThan(Math.abs(v) * 0.01 + 1e-4);
    }
});

test('fp16ToFloat returns Infinity for overflow', () => {
    expect(fp16ToFloat(0x7C00)).toBe(Infinity);
    expect(fp16ToFloat(0xFC00)).toBe(-Infinity);
});

test('floatToFp16 clamps overflow to Inf', () => {
    const huge = 1e10;
    const bits = floatToFp16(huge);
    expect(fp16ToFloat(bits)).toBe(Infinity);
});

test('quantizeFp16 / dequantizeFp16 round-trip', () => {
    const data = new Float32Array([1.0, 2.5, -0.75, 0.0, 100.0, -100.0]);
    const fp16 = quantizeFp16(data);
    const back = dequantizeFp16(fp16);

    expect(back.length).toBe(data.length);
    for (let i = 0; i < data.length; i++) {
        const rel = Math.abs(data[i]) > 1 ? Math.abs(data[i]) * 0.02 : 0.02;
        expect(Math.abs(back[i] - data[i])).toBeLessThan(rel + 1e-3);
    }
});

// ── Int8 quantization ─────────────────────────────────────────────────────────

test('quantizeInt8 / dequantizeInt8 round-trip', () => {
    const data = new Float32Array([0.1, -0.2, 0.5, -0.5, 1.0, -1.0]);
    const { data: q, scale } = quantizeInt8(data);
    const back = dequantizeInt8(q, scale);

    expect(scale).toBeGreaterThan(0);
    expect(back.length).toBe(data.length);

    for (let i = 0; i < data.length; i++) {
        // Int8 has ~1/127 relative error
        expect(Math.abs(back[i] - data[i])).toBeLessThan(0.02);
    }
});

test('quantizeInt8 clamps to [-128, 127]', () => {
    const data = new Float32Array([1.0, -1.0, 0.5]);
    const { data: q } = quantizeInt8(data);
    for (let i = 0; i < q.length; i++) {
        expect(q[i]).toBeGreaterThanOrEqual(-128);
        expect(q[i]).toBeLessThanOrEqual(127);
    }
});

test('quantizeInt8PerChannel produces per-channel scales', () => {
    const data   = new Float32Array([10, 20, 30, 0.1, 0.2, 0.3]);  // 2 channels, 3 elements each
    const { data: q, scales } = quantizeInt8PerChannel(data, 2);
    expect(scales.length).toBe(2);
    expect(scales[0]).toBeGreaterThan(scales[1]);   // first channel has larger range
});

test('quantizeInt8PerChannel / dequantizeInt8PerChannel round-trip', () => {
    const data = new Float32Array([1, 2, 3, 4, -1, -2, -3, -4]);  // 2 channels
    const { data: q, scales } = quantizeInt8PerChannel(data, 2);
    const back = dequantizeInt8PerChannel(q, scales, 2);
    for (let i = 0; i < data.length; i++) {
        expect(Math.abs(back[i] - data[i])).toBeLessThan(0.1);
    }
});

test('quantizeInt8 handles all-zeros', () => {
    const data = new Float32Array([0, 0, 0]);
    const { data: q, scale } = quantizeInt8(data);
    expect(scale).toBe(1.0);
    expect([...q]).toEqual([0, 0, 0]);
});

// ── Memory estimation ─────────────────────────────────────────────────────────

test('estimateMemory returns correct byte counts', () => {
    const { fp32, fp16, int8 } = estimateMemory(100);
    expect(fp32).toBe(400);
    expect(fp16).toBe(200);
    expect(int8).toBe(100);
});

test('estimateMemory with 0 elements returns all zeros', () => {
    const { fp32, fp16, int8 } = estimateMemory(0);
    expect(fp32).toBe(0);
    expect(fp16).toBe(0);
    expect(int8).toBe(0);
});

test('estimateMemory fp32 is always 4× int8', () => {
    const { fp32, int8 } = estimateMemory(256);
    expect(fp32).toBe(int8 * 4);
});

test('estimateMemory fp16 is always 2× int8', () => {
    const { fp16, int8 } = estimateMemory(256);
    expect(fp16).toBe(int8 * 2);
});

// ── FP16 – additional cases ───────────────────────────────────────────────────

test('floatToFp16 and fp16ToFloat round-trip for zero', () => {
    expect(fp16ToFloat(floatToFp16(0.0))).toBe(0.0);
});

test('fp16ToFloat handles NaN bit pattern (exponent=31, mantissa>0)', () => {
    // exponent bits all 1, non-zero mantissa → NaN
    expect(Number.isNaN(fp16ToFloat(0x7E00))).toBe(true);
});

test('floatToFp16 sign bit is set for negative values', () => {
    expect((floatToFp16(-1.0) >>> 15) & 1).toBe(1);
});

test('floatToFp16 sign bit is clear for positive values', () => {
    expect((floatToFp16(1.0) >>> 15) & 1).toBe(0);
});

test('quantizeFp16 output length matches input', () => {
    const data = new Float32Array(50).fill(1.5);
    expect(quantizeFp16(data).length).toBe(50);
});

test('dequantizeFp16 output length matches input', () => {
    const fp16 = new Uint16Array(20).fill(floatToFp16(2.0));
    expect(dequantizeFp16(fp16).length).toBe(20);
});

// ── Int8 – additional cases ───────────────────────────────────────────────────

test('quantizeInt8 single element round-trip', () => {
    const data = new Float32Array([0.75]);
    const { data: q, scale } = quantizeInt8(data);
    const back = dequantizeInt8(q, scale);
    expect(Math.abs(back[0]! - 0.75)).toBeLessThan(0.01);
});

test('quantizeInt8 scale equals maxAbs / 127', () => {
    const data = new Float32Array([0.0, 0.254, -0.127]);
    const { scale } = quantizeInt8(data);
    expect(Math.abs(scale - 0.254 / 127)).toBeLessThan(1e-6);
});

test('dequantizeInt8 output is zeros when int8 data is all zeros', () => {
    const q    = new Int8Array([0, 0, 0]);
    const back = dequantizeInt8(q, 1.0);
    expect([...back]).toEqual([0, 0, 0]);
});

test('quantizeInt8PerChannel single channel matches quantizeInt8', () => {
    const data = new Float32Array([0.5, -0.5, 0.25]);
    const { data: qPC, scales } = quantizeInt8PerChannel(data, 1);
    const { data: q1,  scale  } = quantizeInt8(data);
    expect(Math.abs(scales[0]! - scale)).toBeLessThan(1e-6);
    expect([...qPC]).toEqual([...q1]);
});

test('dequantizeInt8PerChannel output length matches input', () => {
    const data = new Float32Array(6).fill(0.5);
    const { data: q, scales } = quantizeInt8PerChannel(data, 3);
    const back = dequantizeInt8PerChannel(q, scales, 3);
    expect(back.length).toBe(6);
});
