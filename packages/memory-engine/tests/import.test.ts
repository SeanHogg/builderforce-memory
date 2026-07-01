/**
 * tests/import.test.ts
 * The warm-start / weight-port path: read weights back FROM .safetensors into a
 * live EvermindLM. The inverse of export/ — must round-trip losslessly.
 */

import { EvermindLM } from '../src/lm/evermind_lm';
import { BPETokenizer } from '../src/tokenizer/bpe';
import { exportSafetensors } from '../src/export/safetensors';
import { namedTensors } from '../src/export/tensors';
import { importEvermind, importEvermindTensors, inferArchFromTensors, safetensorsToTensors } from '../src/import';

function tinyModel() {
    return new EvermindLM({ vocabSize: 40, dModel: 16, numLayers: 2, hiddenDim: 24, numExperts: 4, topK: 2, seed: 11 });
}

test('exportSafetensors → importEvermind round-trips weights losslessly (F32)', () => {
    const lm = tinyModel();
    const bytes = exportSafetensors(lm);
    const restored = importEvermind(bytes);

    // Same architecture inferred from the tensors alone.
    expect(restored.config.dModel).toBe(16);
    expect(restored.config.numLayers).toBe(2);
    expect(restored.config.numExperts).toBe(4);

    // Every parameter buffer is bit-identical (F32 is lossless).
    const a = lm.parameters();
    const b = restored.parameters();
    expect(b.length).toBe(a.length);
    for (let i = 0; i < a.length; i++) {
        expect(Array.from(b[i]!.data)).toEqual(Array.from(a[i]!.data));
    }
});

test('an imported model generates identical text to the original (serve-equivalent)', () => {
    const lm = tinyModel();
    const tok = new BPETokenizer();
    tok.train('the quick brown fox jumps over the lazy dog. a b c d e f g.', { numMerges: 30 });

    const restored = importEvermind(exportSafetensors(lm));
    const before = lm.generateText('the', tok, { maxNewTokens: 6, temperature: 0 });
    const after = restored.generateText('the', tok, { maxNewTokens: 6, temperature: 0 });
    expect(after).toBe(before);
});

test('F16 export imports back within fp16 tolerance', () => {
    const lm = tinyModel();
    const restored = importEvermind(exportSafetensors(lm, { fp16: true }));
    const a = lm.parameters();
    const b = restored.parameters();
    for (let i = 0; i < a.length; i++) {
        for (let j = 0; j < a[i]!.data.length; j++) {
            expect(b[i]!.data[j]!).toBeCloseTo(a[i]!.data[j]!, 2);
        }
    }
});

test('inferArchFromTensors recovers arch from shapes; topK defaults but is overridable', () => {
    const lm = tinyModel();
    const tensors = namedTensors(lm);
    expect(inferArchFromTensors(tensors)).toMatchObject({ vocabSize: 40, dModel: 16, numLayers: 2, convKernel: 3, hiddenDim: 24, numExperts: 4 });
    expect(inferArchFromTensors(tensors, 3).topK).toBe(3);
});

test('rename map warm-starts a foreign checkpoint whose tensor names differ', () => {
    const lm = tinyModel();
    // Simulate a foreign checkpoint: prefix every tensor name with "backbone.".
    const foreign = namedTensors(lm).map((t) => ({ ...t, name: `backbone.${t.name}` }));
    const restored = importEvermindTensors(foreign, {
        rename: (name) => (name.startsWith('backbone.') ? name.slice('backbone.'.length) : name),
    });
    expect(restored.config.numLayers).toBe(2);
    expect(Array.from(restored.parameters()[0]!.data)).toEqual(Array.from(lm.parameters()[0]!.data));
});

test('a missing or mis-shaped tensor fails with the exact offending name', () => {
    const lm = tinyModel();
    const tensors = namedTensors(lm);

    const dropped = tensors.filter((t) => t.name !== 'layers.1.moe.router.weight');
    expect(() => importEvermindTensors(dropped)).toThrow(/layers\.1\.moe\.router\.weight/);

    const badShape = tensors.map((t) =>
        t.name === 'token_embedding.weight' ? { ...t, data: t.data.subarray(0, t.data.length - 1) } : t,
    );
    expect(() => importEvermindTensors(badShape)).toThrow(/token_embedding\.weight|elements/);
});

test('safetensorsToTensors rejects a truncated buffer', () => {
    expect(() => safetensorsToTensors(new Uint8Array(4))).toThrow(/too short/);
});
