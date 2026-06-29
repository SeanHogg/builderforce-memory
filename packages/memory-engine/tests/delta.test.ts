/**
 * tests/delta.test.ts — EVM-6: sparse, row-granular weight delta checkpoints.
 */

import {
    computeRowDelta,
    applyRowDelta,
    serializeRowDelta,
    deserializeRowDelta,
    diffCheckpoints,
    applyCheckpointDiff,
} from '../src/utils/delta';
import { EvermindLM, EvermindLMTrainer } from '../src/lm/evermind_lm';

describe('row delta', () => {
    it('captures only changed rows and reconstructs exactly', () => {
        const base = new Float32Array([1, 1, 2, 2, 3, 3]); // 3 rows × 2
        const current = new Float32Array([1, 1, 9, 9, 3, 3]); // only row 1 changed
        const delta = computeRowDelta(base, current, 2);
        expect(delta.rows).toEqual([1]);
        expect(Array.from(delta.data)).toEqual([9, 9]);
        expect(Array.from(applyRowDelta(base, delta))).toEqual([1, 1, 9, 9, 3, 3]);
    });

    it('respects eps when deciding a row changed', () => {
        const base = new Float32Array([1, 1]);
        const current = new Float32Array([1.0001, 1]);
        expect(computeRowDelta(base, current, 2, 0.001).rows).toEqual([]); // within eps
        expect(computeRowDelta(base, current, 2, 0).rows).toEqual([0]);
    });

    it('an identical model yields an empty delta', () => {
        const v = new Float32Array([1, 2, 3, 4]);
        expect(computeRowDelta(v, Float32Array.from(v), 2).rows).toEqual([]);
    });

    it('validates lengths and rowSize', () => {
        expect(() => computeRowDelta(new Float32Array(4), new Float32Array(5), 2)).toThrow(/length mismatch/);
        expect(() => computeRowDelta(new Float32Array(5), new Float32Array(5), 2)).toThrow(/multiple of rowSize/);
        expect(() => applyRowDelta(new Float32Array(5), { rowSize: 2, rows: [], data: new Float32Array(0) })).toThrow(/multiple of rowSize/);
    });

    it('serialize/deserialize round-trips with CRC', () => {
        const base = new Float32Array([0, 0, 0, 0, 0, 0]);
        const current = new Float32Array([0, 0, 7, 8, 0, 0]);
        const delta = computeRowDelta(base, current, 2);
        const bin = serializeRowDelta(delta);
        const back = deserializeRowDelta(bin);
        expect(back.rowSize).toBe(2);
        expect(back.rows).toEqual([1]);
        expect(Array.from(applyRowDelta(base, back))).toEqual([0, 0, 7, 8, 0, 0]);
    });

    it('rejects a corrupted serialized delta', () => {
        const bin = serializeRowDelta(computeRowDelta(new Float32Array([0, 0]), new Float32Array([1, 1]), 2));
        new Uint8Array(bin)[16] ^= 0xff; // corrupt a data byte
        expect(() => deserializeRowDelta(bin)).toThrow(/CRC integrity check/);
    });

    it('applyRowDelta guards an out-of-range row index', () => {
        const bad = { rowSize: 2, rows: [99], data: new Float32Array([1, 2]) };
        expect(() => applyRowDelta(new Float32Array([0, 0]), bad)).toThrow(/out of range/);
    });
});

describe('EvermindLM delta checkpoints (EVM-6)', () => {
    const CFG = { vocabSize: 32, dModel: 8, numLayers: 1, hiddenDim: 12, seed: 4 };

    const codec = {
        encode: (s: string) => [...s].map((c) => c.charCodeAt(0) % CFG.vocabSize),
        decode: (ids: number[]) => ids.join(','),
    };

    it('a no-op delta is far smaller than a full checkpoint (the size win)', () => {
        const model = new EvermindLM(CFG);
        const base = model.exportWeights();
        const noChange = model.exportDelta(base); // nothing changed → ~empty delta
        const full = model.exportWeights();
        expect(noChange.byteLength).toBeLessThan(full.byteLength);
    });

    it('a delta reconstructs an adapted model exactly (base + delta == current)', () => {
        const model = new EvermindLM(CFG);
        const base = model.exportWeights();
        new EvermindLMTrainer(model, { lr: 0.05, epochs: 5 }).fit([[1, 2, 3, 4], [2, 3, 4, 5]]);

        const delta = model.exportDelta(base); // eps 0 → exact
        const restored = new EvermindLM(CFG);
        restored.loadDelta(base, delta);

        const a = model.generateText('a', codec, { maxNewTokens: 4, temperature: 0 });
        const b = restored.generateText('a', codec, { maxNewTokens: 4, temperature: 0 });
        expect(b).toBe(a);
    });
});

describe('checkpoint diff (EVM-6b — GPU adapt() save path)', () => {
    const CFG = { vocabSize: 40, dModel: 8, numLayers: 1, hiddenDim: 12, seed: 9 };

    it('diff + apply reconstructs the current checkpoint byte-for-byte', () => {
        const model = new EvermindLM(CFG);
        const base = model.exportWeights({ fp16: false });
        new EvermindLMTrainer(model, { lr: 0.05, epochs: 4 }).fit([[1, 2, 3], [3, 4, 5]]);
        const current = model.exportWeights({ fp16: false });

        const diff = diffCheckpoints(base, current);
        const restored = applyCheckpointDiff(base, diff);
        expect(new Uint8Array(restored)).toEqual(new Uint8Array(current));
    });

    it('an unchanged model yields a tiny diff vs the full checkpoint', () => {
        const model = new EvermindLM(CFG);
        const base = model.exportWeights({ fp16: false });
        const diff = diffCheckpoints(base, model.exportWeights({ fp16: false }));
        expect(diff.byteLength).toBeLessThan(base.byteLength);
    });

    it('rejects diffing checkpoints of different shapes', () => {
        const a = new EvermindLM(CFG).exportWeights({ fp16: false });
        const b = new EvermindLM({ ...CFG, dModel: 16 }).exportWeights({ fp16: false });
        expect(() => diffCheckpoints(a, b)).toThrow(/shape mismatch/);
    });

    it('a reconstructed checkpoint loads cleanly into a fresh model', () => {
        const model = new EvermindLM(CFG);
        const base = model.exportWeights({ fp16: false });
        new EvermindLMTrainer(model, { lr: 0.05, epochs: 3 }).fit([[2, 3, 4]]);
        const diff = diffCheckpoints(base, model.exportWeights({ fp16: false }));
        const restored = applyCheckpointDiff(base, diff);
        const fresh = new EvermindLM(CFG);
        expect(() => fresh.loadWeights(restored)).not.toThrow();
    });
});
