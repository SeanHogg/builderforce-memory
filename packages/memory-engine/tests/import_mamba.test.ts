/**
 * tests/import_mamba.test.ts
 * The plain Mamba (`state-spaces/mamba-*`) route into the foreign port: BF16
 * safetensors, a config read off the tensors, and fitting the vocabulary — the
 * pieces tools/convert.html now runs instead of its own converter.
 */

import type { NamedTensor } from '../src/export/tensors';
import { foreignMambaAdapterFor, mambaConfigFromTensors, portForeignMamba, resizePortedVocab, safetensorsToTensors } from '../src/import';

const E = 8, L = 2, V = 12, N = 4, K = 3, D = 16, R = 2;

function ramp(n: number, seed: number): Float32Array {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = seed + i / 1000;
    return out;
}

/** A state-spaces/mamba checkpoint: native `backbone.embedding` spelling, no lm_head. */
function mambaTensors(): NamedTensor[] {
    const spec: Array<[string, number[]]> = [['backbone.embedding.weight', [V, E]]];
    for (let i = 0; i < L; i++) {
        const m = `backbone.layers.${i}.mixer`;
        spec.push(
            [`backbone.layers.${i}.norm.weight`, [E]],
            [`${m}.in_proj.weight`, [2 * D, E]],
            [`${m}.conv1d.weight`, [D, 1, K]],
            [`${m}.conv1d.bias`, [D]],
            [`${m}.x_proj.weight`, [R + 2 * N, D]],
            [`${m}.dt_proj.weight`, [D, R]],
            [`${m}.dt_proj.bias`, [D]],
            [`${m}.A_log`, [D, N]],
            [`${m}.D`, [D]],
            [`${m}.out_proj.weight`, [E, D]],
        );
    }
    spec.push(['backbone.norm_f.weight', [E]]);
    return spec.map(([name, shape], i) => ({ name, shape, data: ramp(shape.reduce((a, b) => a * b, 1), i + 1) }));
}

/** A one-tensor BF16 safetensors buffer. */
function bf16Safetensors(name: string, values: number[]): Uint8Array {
    const header = new TextEncoder().encode(JSON.stringify({ [name]: { dtype: 'BF16', shape: [values.length], data_offsets: [0, values.length * 2] } }));
    const out = new Uint8Array(8 + header.length + values.length * 2);
    const dv = new DataView(out.buffer);
    dv.setBigUint64(0, BigInt(header.length), true);
    out.set(header, 8);
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);
    values.forEach((v, i) => { f32[0] = v; dv.setUint16(8 + header.length + i * 2, u32[0]! >>> 16, true); });
    return out;
}

test('BF16 tensors decode to the f32 values they truncate', () => {
    const [t] = safetensorsToTensors(bf16Safetensors('w', [1, -2.5, 0.15625, 0]));
    expect(Array.from(t!.data)).toEqual([1, -2.5, 0.15625, 0]);
});

test('a Mamba config is read off the tensor shapes', () => {
    expect(mambaConfigFromTensors(mambaTensors())).toEqual({
        model_type: 'mamba',
        architectures: ['MambaForCausalLM'],
        vocab_size: V,
        hidden_size: E,
        num_hidden_layers: L,
        intermediate_size: D,
        state_size: N,
        conv_kernel: K,
        time_step_rank: R,
    });
    expect(() => mambaConfigFromTensors([])).toThrow(/embedding/);
});

test('plain Mamba is claimed and ports every parameter with nothing left over', () => {
    const t = mambaTensors();
    const config = mambaConfigFromTensors(t);
    expect(foreignMambaAdapterFor(config).id).toBe('mamba');
    const port = portForeignMamba(config, t);
    expect(port.adapter).toBe('mamba');
    expect(port.unmappedSources).toEqual([]);
    expect(port.modelConfig).toMatchObject({ vocabSize: V, dModel: E, numLayers: L, dState: N, dConv: K, expand: 2 });
    expect(port.weights.get('embedding')!.length).toBe(V * E);
});

test('resizing the vocabulary pads with seeded rows, records them, and truncates', () => {
    const port = portForeignMamba(mambaConfigFromTensors(mambaTensors()), mambaTensors());
    const grown = resizePortedVocab(port, V + 3, { seed: 7 });
    const emb = grown.weights.get('embedding')!;
    expect(grown.modelConfig.vocabSize).toBe(V + 3);
    expect(emb.length).toBe((V + 3) * E);
    expect(Array.from(emb.subarray(0, V * E))).toEqual(Array.from(port.weights.get('embedding')!));
    expect(Array.from(resizePortedVocab(port, V + 3, { seed: 7 }).weights.get('embedding')!)).toEqual(Array.from(emb));
    expect(grown.synthesisedTargets.at(-1)).toMatchObject({ target: 'embedding', value: 0.02 });

    const shrunk = resizePortedVocab(port, V - 2);
    expect(shrunk.weights.get('embedding')!.length).toBe((V - 2) * E);
    expect(resizePortedVocab(port, V)).toBe(port);
    expect(port.weights.get('embedding')!.length).toBe(V * E); // the input is not mutated
});
