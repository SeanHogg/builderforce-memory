/**
 * mamba_model.ts – HybridMambaModel: Mamba-1/2/3 and Attention layer scheduling.
 *
 * Replaces the fixed MambaBlock[] array with a SequenceLayer[] built from a
 * per-layer type schedule.  MambaModel is kept as a backward-compatible alias
 * (all-mamba1 schedule).
 *
 * MBJS binary format:
 *   Version 1 (legacy): [magic][v=1][nParams][numel[]][ f32 data ]
 *   Version 2 (new):    [magic][v=2][nLayers][layerType[]][padding][nParams][numel[]][ f32 data ]
 *     layerType: 0=mamba1, 1=mamba2, 2=mamba3, 3=attention
 */

import { Mamba1Block }              from './mamba1_block.js';
import { Mamba2Block }              from './mamba2_block.js';
import { Mamba3Block }              from './mamba3_block.js';
import { AttentionBlock }           from './attention_block.js';
import type { SequenceLayer, LayerParam, LayerType } from './sequence_layer.js';
import type { Mamba1BlockConfig }   from './mamba1_block.js';
import type { Mamba2BlockConfig }   from './mamba2_block.js';
import type { Mamba3BlockConfig }   from './mamba3_block.js';
import type { AttentionBlockConfig } from './attention_block.js';

import {
    createStorageBuffer,
    createEmptyStorageBuffer,
    createUniformBuffer,
    createComputePipeline,
    createBindGroup,
    dispatchKernel,
    readBuffer,
    uploadBuffer,
    cdiv,
} from '../utils/gpu_utils.js';
import { LINEAR_FORWARD_WGSL } from '../kernels/linear_projection.js';
import { ACTIVATIONS_WGSL }    from '../kernels/activations.js';
import { gaussianArray, setInitSeed } from '../utils/rng.js';
import { quantizeFp16, dequantizeFp16 } from '../utils/quantization.js';
import { appendCrcTrailer, verifyCrcTrailer } from '../utils/crc32.js';

// ── Public types ──────────────────────────────────────────────────────────────

export interface LayerSpec {
    type    : LayerType;
    config? : Partial<Mamba1BlockConfig | Mamba2BlockConfig | Mamba3BlockConfig | AttentionBlockConfig>;
}

export interface HybridMambaModelConfig {
    vocabSize        : number;
    dModel           : number;
    numLayers        : number;

    /**
     * Per-layer type schedule.  Length must equal numLayers.
     * Defaults to all 'mamba1' (backward-compatible).
     */
    layers?          : LayerSpec[];

    // Shared defaults per variant type (individual LayerSpec.config overrides take precedence)
    defaultMamba1?   : Partial<Mamba1BlockConfig>;
    defaultMamba2?   : Partial<Mamba2BlockConfig>;
    defaultMamba3?   : Partial<Mamba3BlockConfig>;
    defaultAttention?: Partial<AttentionBlockConfig>;

    // Mamba-1 compatible shorthand fields (applied to all mamba1 layers)
    dState?          : number;
    dConv?           : number;
    expand?          : number;

    // Mamba-2/3 defaults
    nHeads?          : number;
    nGroups?         : number;
    chunkLen?        : number;
    mimoGroup?       : number;

    eosId?           : number;

    /**
     * Optional deterministic seed for weight initialisation. When set, the
     * embedding table and all block weights are initialised reproducibly — the
     * same seed yields byte-identical initial weights on any machine. When
     * omitted, weights use `Math.random` (non-reproducible) as before.
     */
    seed?            : number;
}

/** Legacy Mamba-1-only config (fully backward-compatible). */
export interface MambaModelConfig {
    vocabSize  : number;
    dModel     : number;
    numLayers  : number;
    dState?    : number;
    dConv?     : number;
    expand?    : number;
    eosId?     : number;
}

export interface ModelForwardResult {
    logits    : Float32Array;
    gpuLogits : GPUBuffer;
    caches    : unknown[];
}

export interface SamplingOptions {
    temperature? : number;
    topK?        : number;
    topP?        : number;
}

// ── MBJS format constants ─────────────────────────────────────────────────────

const MBJS_MAGIC   = 0x4D424A53;  // 'MBJS'
const LAYER_TYPE_ID: Record<LayerType, number> = {
    mamba1   : 0,
    mamba2   : 1,
    mamba3   : 2,
    attention: 3,
};
const ID_TO_LAYER_TYPE: LayerType[] = ['mamba1', 'mamba2', 'mamba3', 'attention'];

// ── HybridMambaModel ──────────────────────────────────────────────────────────

export class HybridMambaModel {
    device        : GPUDevice;
    config        : Required<HybridMambaModelConfig>;
    gpuEmbedding  : GPUBuffer;
    layers        : SequenceLayer[];
    layerSpecs    : LayerSpec[];
    gpuFinalNorm  : GPUBuffer;
    tiedEmbedding : boolean;
    gpuLMHeadBias : GPUBuffer;

    private _lmHeadPipeline  : GPUComputePipeline;
    private _rmsnormPipeline : GPUComputePipeline;
    private _embedPipeline   : GPUComputePipeline;
    private _wslaMode = false;

    constructor(device: GPUDevice, config: HybridMambaModelConfig) {
        this.device = device;
        this.config = {
            dState        : 16,
            dConv         : 4,
            expand        : 2,
            nHeads        : 4,
            nGroups       : 1,
            chunkLen      : 256,
            mimoGroup     : 1,
            eosId         : -1,
            defaultMamba1 : {},
            defaultMamba2 : {},
            defaultMamba3 : {},
            defaultAttention: {},
            layers        : undefined as unknown as LayerSpec[],
            seed          : undefined as unknown as number,
            ...config,
        } as Required<HybridMambaModelConfig>;

        // Install the deterministic init seed (if any) for the duration of
        // construction, so the embedding table and every block initialise
        // reproducibly. Restored to Math.random once all weights are built.
        setInitSeed(this.config.seed);

        // Resolve layer schedule
        const layerSchedule: LayerSpec[] = config.layers
            ?? Array.from({ length: config.numLayers }, () => ({ type: 'mamba1' as LayerType }));

        if (layerSchedule.length !== config.numLayers) {
            throw new Error(
                `HybridMambaModel: layers schedule length (${layerSchedule.length}) must equal numLayers (${config.numLayers}).`
            );
        }
        this.layerSpecs = layerSchedule;

        // Embedding table
        const { vocabSize, dModel } = this.config;
        const embedData = gaussianArray(vocabSize * dModel, 1.0 / Math.sqrt(dModel));
        this.gpuEmbedding = createStorageBuffer(device, embedData, true);

        // Build layers (block constructors also draw from the seeded source)
        this.layers = layerSchedule.map(spec => this._buildLayer(spec));

        // Restore the default Math.random source now that all weights are built.
        setInitSeed(undefined);

        // Final RMSNorm
        this.gpuFinalNorm = createStorageBuffer(device, new Float32Array(dModel).fill(1.0), true);

        this.tiedEmbedding = true;
        this.gpuLMHeadBias = createStorageBuffer(device, new Float32Array(vocabSize), true);

        this._lmHeadPipeline  = createComputePipeline(device, LINEAR_FORWARD_WGSL, 'linear_forward');
        this._rmsnormPipeline = createComputePipeline(device, ACTIVATIONS_WGSL,    'rmsnorm_forward');
        this._embedPipeline   = createComputePipeline(device, EMBED_LOOKUP_WGSL,   'embed_lookup');
    }

    private _buildLayer(spec: LayerSpec): SequenceLayer {
        const c = this.config;
        switch (spec.type) {
            case 'mamba1': {
                const base: Mamba1BlockConfig = {
                    dModel : c.dModel,
                    dState : c.dState,
                    dConv  : c.dConv,
                    expand : c.expand,
                    ...c.defaultMamba1,
                };
                return new Mamba1Block(this.device, { ...base, ...(spec.config ?? {}) } as Mamba1BlockConfig);
            }
            case 'mamba2': {
                const base: Mamba2BlockConfig = {
                    dModel  : c.dModel,
                    dState  : c.dState,
                    dConv   : c.dConv,
                    expand  : c.expand,
                    nHeads  : c.nHeads,
                    nGroups : c.nGroups,
                    chunkLen: c.chunkLen,
                    ...c.defaultMamba2,
                };
                return new Mamba2Block(this.device, { ...base, ...(spec.config ?? {}) } as Mamba2BlockConfig);
            }
            case 'mamba3': {
                const base: Mamba3BlockConfig = {
                    dModel   : c.dModel,
                    dState   : c.dState,
                    dConv    : c.dConv,
                    expand   : c.expand,
                    nHeads   : c.nHeads,
                    nGroups  : c.nGroups,
                    chunkLen : c.chunkLen,
                    mimoGroup: c.mimoGroup,
                    ...c.defaultMamba3,
                };
                return new Mamba3Block(this.device, { ...base, ...(spec.config ?? {}) } as Mamba3BlockConfig);
            }
            case 'attention': {
                const base: AttentionBlockConfig = {
                    dModel : c.dModel,
                    nHeads : c.nHeads,
                    ...c.defaultAttention,
                };
                return new AttentionBlock(this.device, { ...base, ...(spec.config ?? {}) } as AttentionBlockConfig);
            }
        }
    }

    embedTokens(tokenIds: number[] | Uint32Array, batch: number, seqLen: number): GPUBuffer {
        const { dModel } = this.config;
        const M = batch * seqLen;

        const idsBuf = createStorageBuffer(this.device,
            tokenIds instanceof Uint32Array ? tokenIds : new Uint32Array(tokenIds), false);
        const outBuf = createEmptyStorageBuffer(this.device, M * dModel * 4, true);

        const pBuf = createUniformBuffer(this.device, new Uint32Array([M, dModel]).buffer);
        const bg   = createBindGroup(this.device, this._embedPipeline,
            [pBuf, idsBuf, this.gpuEmbedding, outBuf]);
        dispatchKernel(this.device, this._embedPipeline, bg, [cdiv(M, 64), 1, 1]);

        idsBuf.destroy();
        pBuf.destroy();
        return outBuf;
    }

    async forward(tokenIds: number[] | Uint32Array, batch: number, seqLen: number): Promise<ModelForwardResult> {
        const { dModel, vocabSize } = this.config;
        const M = batch * seqLen;

        let hidden = this.embedTokens(tokenIds, batch, seqLen);

        const caches: unknown[] = [];
        for (const layer of this.layers) {
            const { output, cache } = layer.forward(hidden, batch, seqLen);
            caches.push(cache);
            hidden.destroy();
            hidden = output;
        }

        // Final RMSNorm
        const normOut = createEmptyStorageBuffer(this.device, M * dModel * 4, true);
        const normInv = createEmptyStorageBuffer(this.device, M * 4, false);
        {
            const params = new ArrayBuffer(16);
            new Uint32Array(params, 0, 2).set([M, dModel]);
            new Float32Array(params, 8, 1).set([1e-6]);
            const pBuf = createUniformBuffer(this.device, params);
            const bg = createBindGroup(this.device, this._rmsnormPipeline,
                [pBuf, hidden, this.gpuFinalNorm, normOut, normInv]);
            dispatchKernel(this.device, this._rmsnormPipeline, bg, [cdiv(M, 64), 1, 1]);
        }
        hidden.destroy();

        // LM head (tied embedding)
        const gpuLogits = createEmptyStorageBuffer(this.device, M * vocabSize * 4, true);
        {
            const params = new Uint32Array([M, dModel, vocabSize]).buffer;
            const pBuf   = createUniformBuffer(this.device, params);
            const bg = createBindGroup(this.device, this._lmHeadPipeline,
                [pBuf, normOut, this.gpuEmbedding, this.gpuLMHeadBias, gpuLogits]);
            dispatchKernel(this.device, this._lmHeadPipeline, bg,
                [cdiv(M, 16), cdiv(vocabSize, 16), 1]);
        }
        normOut.destroy();
        normInv.destroy();

        const logits = await readBuffer(this.device, gpuLogits, M * vocabSize * 4);
        return { logits, gpuLogits, caches };
    }

    /**
     * Produces a single fixed-length embedding vector for a token sequence.
     *
     * Runs the full layer stack plus the final RMSNorm — i.e. the same hidden
     * state the LM head consumes — then mean-pools across sequence positions and
     * L2-normalises the result. The returned vector has length `dModel` and is
     * suitable for cosine-similarity semantic search.
     *
     * Unlike `forward()`, this skips the (expensive) LM-head projection: it only
     * needs the `dModel`-wide hidden state, not `vocabSize` logits.
     *
     * The embedding reflects whatever the model currently knows — an untrained
     * model behaves like a random projection of the token embeddings (still
     * lexically discriminative), and the representation sharpens automatically as
     * the model is adapted/distilled.
     */
    async embed(tokenIds: number[] | Uint32Array): Promise<Float32Array> {
        const { dModel } = this.config;
        const seqLen = tokenIds.length;
        const batch  = 1;
        const M      = batch * seqLen;
        if (M === 0) return new Float32Array(dModel);

        let hidden = this.embedTokens(tokenIds, batch, seqLen);
        for (const layer of this.layers) {
            const { output } = layer.forward(hidden, batch, seqLen);
            hidden.destroy();
            hidden = output;
        }

        // Final RMSNorm — mirrors forward(), but we stop here (no LM head).
        const normOut = createEmptyStorageBuffer(this.device, M * dModel * 4, true);
        const normInv = createEmptyStorageBuffer(this.device, M * 4, false);
        {
            const params = new ArrayBuffer(16);
            new Uint32Array(params, 0, 2).set([M, dModel]);
            new Float32Array(params, 8, 1).set([1e-6]);
            const pBuf = createUniformBuffer(this.device, params);
            const bg = createBindGroup(this.device, this._rmsnormPipeline,
                [pBuf, hidden, this.gpuFinalNorm, normOut, normInv]);
            dispatchKernel(this.device, this._rmsnormPipeline, bg, [cdiv(M, 64), 1, 1]);
        }
        hidden.destroy();

        const normed = await readBuffer(this.device, normOut, M * dModel * 4);
        normOut.destroy();
        normInv.destroy();

        // Mean-pool across sequence positions → dModel vector.
        const out = new Float32Array(dModel);
        for (let t = 0; t < seqLen; t++) {
            const base = t * dModel;
            for (let d = 0; d < dModel; d++) out[d]! += normed[base + d]!;
        }
        for (let d = 0; d < dModel; d++) out[d]! /= seqLen;

        // L2-normalise so cosine similarity reduces to a dot product.
        let norm = 0;
        for (let d = 0; d < dModel; d++) norm += out[d]! * out[d]!;
        norm = Math.sqrt(norm) || 1;
        for (let d = 0; d < dModel; d++) out[d]! /= norm;

        return out;
    }

    async generate(promptIds: number[], maxNewTokens = 200, samplingOpts: SamplingOptions = {}): Promise<number[]> {
        const { temperature = 1.0, topK = 50, topP = 0.9 } = samplingOpts;
        const { vocabSize } = this.config;

        const ids = [...promptIds];

        for (let step = 0; step < maxNewTokens; step++) {
            const { logits } = await this.forward(new Uint32Array(ids), 1, ids.length);
            const lastLogits = logits.slice((ids.length - 1) * vocabSize, ids.length * vocabSize);
            const nextId = sampleToken(lastLogits, { temperature, topK, topP });
            ids.push(nextId);
            if (nextId === this.config.eosId) break;
        }

        return ids;
    }

    parameters(): LayerParam[] {
        const params: LayerParam[] = [];

        params.push({
            buf  : this.gpuEmbedding,
            numel: this.config.vocabSize * this.config.dModel,
            name : 'embedding',
        });

        for (let i = 0; i < this.layers.length; i++) {
            for (const p of this.layers[i]!.parameters()) {
                params.push({ ...p, name: `layer${i}.${p.name}` });
            }
        }

        params.push({
            buf  : this.gpuFinalNorm,
            numel: this.config.dModel,
            name : 'final_norm',
        });

        return params;
    }

    setWSLAMode(enabled: boolean): void {
        for (const layer of this.layers) layer.setWSLAMode(enabled);
        this._wslaMode = enabled;
    }

    // ── Serialisation (MBJS v2 / v3) ──────────────────────────────────────────

    /**
     * Export all parameters to an ArrayBuffer.
     *
     * MBJS v2/v3 format (identical header; only the data encoding differs):
     *   [0..3]   magic    : uint32 = 0x4D424A53
     *   [4..7]   version  : uint32 = 2 (fp32 data) | 3 (fp16 data)
     *   [8..11]  nLayers  : uint32
     *   [12 .. 12+nLayers-1]  layerType[i]: uint8 (0=m1, 1=m2, 2=m3, 3=attn)
     *   aligned to 4 bytes: padding
     *   [next 4] nParams  : uint32
     *   [next 4*nParams]  numel[i]: uint32
     *   [data]  float32 values (v2)  |  float16 values (v3, half the size)
     *
     * Pass `{ fp16: true }` to emit a v3 checkpoint — roughly half the bytes,
     * with a small precision loss that is negligible for SSM weights.
     */
    async exportWeights(opts: { fp16?: boolean } = {}): Promise<ArrayBuffer> {
        const fp16     = opts.fp16 ?? false;
        const params   = this.parameters();
        const nParams  = params.length;
        const nLayers  = this.layers.length;

        const arrays: Float32Array[] = await Promise.all(
            params.map(p => readBuffer(this.device, p.buf, p.numel * 4))
        );

        // Header: magic(4) + version(4) + nLayers(4) + layerTypes(nLayers, padded to 4) + nParams(4) + numels(4*nParams)
        const layerTypeBytes = Math.ceil(nLayers / 4) * 4;  // align to 4
        const headerBytes    = 4 + 4 + 4 + layerTypeBytes + 4 + nParams * 4;
        const bytesPerEl     = fp16 ? 2 : 4;
        const totalEls       = arrays.reduce((a, arr) => a + arr.length, 0);
        const dataBytes      = totalEls * bytesPerEl;
        const out  = new ArrayBuffer(headerBytes + dataBytes);
        const view = new DataView(out);

        let off = 0;
        view.setUint32(off, MBJS_MAGIC, true);   off += 4;
        view.setUint32(off, fp16 ? 3 : 2, true); off += 4;   // version 2 (fp32) | 3 (fp16)
        view.setUint32(off, nLayers,     true);  off += 4;

        for (let i = 0; i < nLayers; i++) {
            const lt = this.layers[i]!.layerType;
            view.setUint8(off + i, LAYER_TYPE_ID[lt]);
        }
        off += layerTypeBytes;

        view.setUint32(off, nParams, true); off += 4;
        for (const p of params) {
            view.setUint32(off, p.numel, true);
            off += 4;
        }
        // Header bytes are a multiple of 4, so both Float32Array and Uint16Array
        // views below are correctly aligned at `off`.
        if (fp16) {
            for (const arr of arrays) {
                const half = quantizeFp16(arr);
                new Uint16Array(out, off, half.length).set(half);
                off += half.length * 2;
            }
        } else {
            for (const arr of arrays) {
                new Float32Array(out, off, arr.length).set(arr);
                off += arr.byteLength;
            }
        }

        // Append a CRC-32 trailer for integrity-on-load (backward compatible —
        // length-bounded readers ignore the trailing bytes). (EVM-7)
        return appendCrcTrailer(out);
    }

    /**
     * Load parameters from an MBJS v1, v2, or v3 ArrayBuffer.
     *
     * v1: assumes all layers are mamba1 (backward compatible).
     * v2: reads layer type array and validates per-layer parameter counts (fp32 data).
     * v3: identical layout to v2 but the data section is fp16 (dequantised on load).
     */
    async loadWeights(buffer: ArrayBuffer): Promise<void> {
        // Verify the CRC trailer when present (corrupt/truncated guard, EVM-7).
        const crc = verifyCrcTrailer(buffer);
        if (crc.hasTrailer && !crc.ok) {
            throw new Error('Invalid weight file: failed CRC integrity check (corrupt or truncated).');
        }
        const view = new DataView(buffer);
        let off    = 0;

        const magic = view.getUint32(off, true); off += 4;
        if (magic !== MBJS_MAGIC) {
            throw new Error('Invalid weight file: bad magic number. Expected MBJS file.');
        }

        const version = view.getUint32(off, true); off += 4;

        if (version === 1) {
            // Legacy path: all-mamba1, no layer metadata
            const nParams = view.getUint32(off, true); off += 4;
            const params  = this.parameters();

            if (nParams !== params.length) {
                throw new Error(
                    `Weight file has ${nParams} parameters but this model has ${params.length}.`
                );
            }

            const numels: number[] = [];
            for (let i = 0; i < nParams; i++) {
                numels.push(view.getUint32(off, true));
                off += 4;
            }

            for (let i = 0; i < nParams; i++) {
                const p     = params[i]!;
                const numel = numels[i]!;
                if (numel !== p.numel) {
                    throw new Error(`Parameter ${i} ("${p.name}") size mismatch: file=${numel}, model=${p.numel}.`);
                }
                uploadBuffer(this.device, p.buf, new Float32Array(buffer, off, p.numel));
                off += p.numel * 4;
            }
            return;
        }

        if (version === 2 || version === 3) {
            const fp16    = version === 3;
            const nLayers = view.getUint32(off, true); off += 4;

            if (nLayers !== this.layers.length) {
                throw new Error(`Weight file has ${nLayers} layers but this model has ${this.layers.length}.`);
            }

            // Read layer types and validate
            for (let i = 0; i < nLayers; i++) {
                const typeId = view.getUint8(off + i);
                const expectedType = this.layers[i]!.layerType;
                const fileType     = ID_TO_LAYER_TYPE[typeId] ?? 'mamba1';
                if (fileType !== expectedType) {
                    throw new Error(
                        `Layer ${i} type mismatch: file="${fileType}", model="${expectedType}".`
                    );
                }
            }
            const layerTypeBytes = Math.ceil(nLayers / 4) * 4;
            off += layerTypeBytes;

            const nParams = view.getUint32(off, true); off += 4;
            const params  = this.parameters();

            if (nParams !== params.length) {
                throw new Error(
                    `Weight file has ${nParams} parameters but this model has ${params.length}.`
                );
            }

            const numels: number[] = [];
            for (let i = 0; i < nParams; i++) {
                numels.push(view.getUint32(off, true));
                off += 4;
            }

            for (let i = 0; i < nParams; i++) {
                const p     = params[i]!;
                const numel = numels[i]!;
                if (numel !== p.numel) {
                    throw new Error(`Parameter ${i} ("${p.name}") size mismatch: file=${numel}, model=${p.numel}.`);
                }
                if (fp16) {
                    const half = new Uint16Array(buffer, off, numel);
                    uploadBuffer(this.device, p.buf, dequantizeFp16(half));
                    off += numel * 2;
                } else {
                    uploadBuffer(this.device, p.buf, new Float32Array(buffer, off, p.numel));
                    off += numel * 4;
                }
            }
            return;
        }

        throw new Error(`Unsupported MBJS version: ${version}. Expected 1, 2, or 3.`);
    }

    destroy(): void {
        this.gpuEmbedding.destroy();
        for (const layer of this.layers) layer.destroy();
        this.gpuFinalNorm.destroy();
        this.gpuLMHeadBias.destroy();
    }
}

// ── MambaModel – backward-compatible alias ────────────────────────────────────

export class MambaModel extends HybridMambaModel {
    constructor(device: GPUDevice, config: MambaModelConfig) {
        super(device, {
            ...config,
            layers: Array.from({ length: config.numLayers }, () => ({ type: 'mamba1' as LayerType })),
        });
    }
}

// ── Embed lookup WGSL ─────────────────────────────────────────────────────────

const EMBED_LOOKUP_WGSL: string = /* wgsl */`
struct EmbedParams {
    num_tokens : u32,
    d_model    : u32,
};

@group(0) @binding(0) var<uniform>            params  : EmbedParams;
@group(0) @binding(1) var<storage, read>      ids     : array<u32>;
@group(0) @binding(2) var<storage, read>      table   : array<f32>;
@group(0) @binding(3) var<storage, read_write> out    : array<f32>;

@compute @workgroup_size(64, 1, 1)
fn embed_lookup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;
    if (token_idx >= params.num_tokens) { return; }

    let D   = params.d_model;
    let tok = ids[token_idx];
    let src = tok * D;
    let dst = token_idx * D;

    for (var i: u32 = 0u; i < D; i = i + 1u) {
        out[dst + i] = table[src + i];
    }
}
`;

// ── Token sampling ────────────────────────────────────────────────────────────

function sampleToken(logits: Float32Array, { temperature = 1.0, topK = 50, topP = 0.9 } = {}): number {
    const n = logits.length;

    const scaled = new Float32Array(n);
    for (let i = 0; i < n; i++) scaled[i] = logits[i]! / Math.max(temperature, 1e-7);

    let maxL = -Infinity;
    for (let i = 0; i < n; i++) if (scaled[i]! > maxL) maxL = scaled[i]!;
    let sumE = 0;
    const exps = new Float32Array(n);
    for (let i = 0; i < n; i++) { exps[i] = Math.exp(scaled[i]! - maxL); sumE += exps[i]!; }

    const indices = Array.from({ length: n }, (_, i) => i).sort((a, b) => exps[b]! - exps[a]!);
    const topKIdx = indices.slice(0, topK);

    let cumSum = 0;
    const nucleus: number[] = [];
    for (const idx of topKIdx) {
        cumSum += exps[idx]! / sumE;
        nucleus.push(idx);
        if (cumSum >= topP) break;
    }

    let nucleusSum = 0;
    for (const idx of nucleus) nucleusSum += exps[idx]!;
    const threshold = Math.random() * nucleusSum;
    let acc = 0;
    for (const idx of nucleus) {
        acc += exps[idx]!;
        if (acc >= threshold) return idx;
    }
    return nucleus[nucleus.length - 1]!;
}
