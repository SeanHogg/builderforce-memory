/**
 * tests/crc32.test.ts — EVM-7: checkpoint integrity (CRC-32 + trailer).
 */

import { crc32, appendCrcTrailer, verifyCrcTrailer, CRC_TRAILER_MAGIC } from '../src/utils/crc32';
import { EvermindLM } from '../src/lm/evermind_lm';

describe('crc32', () => {
    it('matches the standard check value for "123456789"', () => {
        const bytes = new TextEncoder().encode('123456789');
        expect(crc32(bytes) >>> 0).toBe(0xcbf43926);
    });

    it('is empty-safe and order-sensitive', () => {
        expect(crc32(new Uint8Array(0)) >>> 0).toBe(0);
        expect(crc32(new Uint8Array([1, 2]))).not.toBe(crc32(new Uint8Array([2, 1])));
    });
});

describe('CRC trailer', () => {
    it('round-trips: append then verify ok, with the body recoverable', () => {
        const payload = new Uint8Array([10, 20, 30, 40, 50]).buffer;
        const withCrc = appendCrcTrailer(payload);
        expect(withCrc.byteLength).toBe(payload.byteLength + 8);
        const { hasTrailer, ok, body } = verifyCrcTrailer(withCrc);
        expect(hasTrailer).toBe(true);
        expect(ok).toBe(true);
        expect(new Uint8Array(body)).toEqual(new Uint8Array(payload));
    });

    it('detects corruption in the payload', () => {
        const withCrc = appendCrcTrailer(new Uint8Array([1, 2, 3, 4]).buffer);
        const u8 = new Uint8Array(withCrc);
        u8[0] = u8[0]! ^ 0xff; // flip a bit in the body
        expect(verifyCrcTrailer(withCrc).ok).toBe(false);
    });

    it('treats a buffer without a trailer as legacy (hasTrailer=false, ok=true)', () => {
        const { hasTrailer, ok } = verifyCrcTrailer(new Uint8Array([9, 9, 9, 9, 9, 9, 9, 9, 9]).buffer);
        expect(hasTrailer).toBe(false);
        expect(ok).toBe(true);
    });

    it('treats a too-short buffer as legacy', () => {
        const { hasTrailer, ok } = verifyCrcTrailer(new Uint8Array([1, 2]).buffer);
        expect(hasTrailer).toBe(false);
        expect(ok).toBe(true);
    });

    it('exposes a recognisable magic', () => {
        expect(CRC_TRAILER_MAGIC >>> 0).toBe(0x52435645);
    });
});

describe('EvermindLM checkpoint integrity', () => {
    const CFG = { vocabSize: 24, dModel: 8, numLayers: 1, hiddenDim: 12, seed: 3 };

    it('exports a CRC-trailed checkpoint that re-loads cleanly', () => {
        const m = new EvermindLM(CFG);
        const buf = m.exportWeights();
        expect(verifyCrcTrailer(buf).hasTrailer).toBe(true);
        const m2 = new EvermindLM(CFG);
        expect(() => m2.loadWeights(buf)).not.toThrow();
    });

    it('rejects a corrupted checkpoint with an integrity error', () => {
        const m = new EvermindLM(CFG);
        const buf = m.exportWeights();
        new Uint8Array(buf)[40] = new Uint8Array(buf)[40]! ^ 0xff; // corrupt a weight byte
        const m2 = new EvermindLM(CFG);
        expect(() => m2.loadWeights(buf)).toThrow(/CRC integrity check/);
    });
});
