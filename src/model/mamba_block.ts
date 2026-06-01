/**
 * mamba_block.ts – Mamba Mixer Block
 */

import {
    createComputePipeline,
    createBindGroup,
    createStorageBuffer,
    createEmptyStorageBuffer,
    createUniformBuffer,
    dispatchKernel,
    cdiv,
} from '../utils/gpu_utils.js';

import { SELECTIVE_SCAN_FORWARD_WGSL }  from '../kernels/selective_scan.js';
import { gaussianArray }                from '../utils/rng.js';
import { CONV1D_FORWARD_WGSL }          from '../kernels/conv1d.js';
import { LINEAR_FORWARD_WGSL }          from '../kernels/linear_projection.js';
import { ACTIVATIONS_WGSL }             from '../kernels/activations.js';

export interface MambaBlockConfig {
  dModel: number;
  dState?: number;
  dConv?: number;
  expand?: number;
  dtRank?: number;
  biasConv?: boolean;
}

export interface BlockParam {
  buf: GPUBuffer;
  numel: number;
  name: string;
}

export interface BlockCache {
  normInv: GPUBuffer;
  normIn: GPUBuffer;
  normOut: GPUBuffer;
  zBuf: GPUBuffer;
  xConvIn: GPUBuffer;
  convOut: GPUBuffer;
  siluOut: GPUBuffer;
  deltaFull: GPUBuffer;
  B_raw: GPUBuffer;
  C_raw: GPUBuffer;
  hCache: GPUBuffer;
}

export interface BlockForwardResult {
  output: GPUBuffer;
  cache: BlockCache;
}

export class MambaBlock {
    device: GPUDevice;
    config: Required<MambaBlockConfig>;
    dInner: number;
    dtRank: number;
    wInProj: Float32Array;
    bInProj: Float32Array;
    wConv: Float32Array;
    bConv: Float32Array;
    wXProj: Float32Array;
    bXProj: Float32Array;
    wDtProj: Float32Array;
    bDtProj: Float32Array;
    A_log: Float32Array;
    D_vec: Float32Array;
    wOutProj: Float32Array;
    bOutProj: Float32Array;
    normWeight: Float32Array;
    gpuWeights: Record<string, GPUBuffer>;
    pipelines: Record<string, GPUComputePipeline>;
    private _wslaMode = false;

    constructor(device: GPUDevice, config: MambaBlockConfig) {
        this.device  = device;
        this.config  = {
            dState  : 16,
            dConv   : 4,
            expand  : 2,
            biasConv: true,
            dtRank  : Math.ceil(config.dModel / 16),
            ...config,
        } as Required<MambaBlockConfig>;

        const { dModel, expand } = this.config;
        this.dInner  = expand * dModel;
        this.dtRank  = config.dtRank ?? Math.ceil(dModel / 16);

        // Initialize these before _initWeights so TypeScript is happy
        this.wInProj = new Float32Array(0);
        this.bInProj = new Float32Array(0);
        this.wConv = new Float32Array(0);
        this.bConv = new Float32Array(0);
        this.wXProj = new Float32Array(0);
        this.bXProj = new Float32Array(0);
        this.wDtProj = new Float32Array(0);
        this.bDtProj = new Float32Array(0);
        this.A_log = new Float32Array(0);
        this.D_vec = new Float32Array(0);
        this.wOutProj = new Float32Array(0);
        this.bOutProj = new Float32Array(0);
        this.normWeight = new Float32Array(0);
        this.gpuWeights = {};
        this.pipelines = {};

        this._initWeights();
        this._buildPipelines();
    }

    private _initWeights(): void {
        const { dModel, dState, dConv } = this.config;
        const D = this.dInner;
        const N = dState;
        const K = dConv;
        const R = this.dtRank;

        const randn = (n: number, std = 0.02): Float32Array => gaussianArray(n, std);

        const zeros  = (n: number): Float32Array    => new Float32Array(n);
        const ones   = (n: number): Float32Array    => new Float32Array(n).fill(1.0);

        this.wInProj  = randn(2 * D * dModel);
        this.bInProj  = zeros(2 * D);
        this.wConv    = randn(D * K, 0.01);
        this.bConv    = zeros(D);
        this.wXProj   = randn((R + 2 * N) * D, 0.01);
        this.bXProj   = zeros(R + 2 * N);
        this.wDtProj  = randn(D * R, 0.02);
        this.bDtProj  = zeros(D);

        this.A_log    = new Float32Array(D * N);
        for (let d = 0; d < D; d++) {
            for (let n = 0; n < N; n++) {
                this.A_log[d * N + n] = Math.log(n + 1);
            }
        }

        this.D_vec    = ones(D);
        this.wOutProj = randn(dModel * D, 0.02);
        this.bOutProj = zeros(dModel);
        this.normWeight = ones(dModel);

        this._uploadWeightsToGPU();
    }

    private _uploadWeightsToGPU(): void {
        const d  = this.device;
        const mk = (arr: Float32Array, readable = true): GPUBuffer => createStorageBuffer(d, arr, readable);

        this.gpuWeights = {
            wInProj  : mk(this.wInProj),
            bInProj  : mk(this.bInProj),
            wConv    : mk(this.wConv),
            bConv    : mk(this.bConv),
            wXProj   : mk(this.wXProj),
            bXProj   : mk(this.bXProj),
            wDtProj  : mk(this.wDtProj),
            bDtProj  : mk(this.bDtProj),
            A_log    : mk(this.A_log),
            D_vec    : mk(this.D_vec),
            wOutProj : mk(this.wOutProj),
            bOutProj : mk(this.bOutProj),
            normWeight: mk(this.normWeight),
        };
    }

    private _buildPipelines(): void {
        const d = this.device;

        this.pipelines = {
            linear    : createComputePipeline(d, LINEAR_FORWARD_WGSL,           'linear_forward'),
            conv1d    : createComputePipeline(d, CONV1D_FORWARD_WGSL,           'conv1d_forward'),
            silu      : createComputePipeline(d, ACTIVATIONS_WGSL,              'silu_forward'),
            rmsnorm   : createComputePipeline(d, ACTIVATIONS_WGSL,              'rmsnorm_forward'),
            scan_fwd  : createComputePipeline(d, SELECTIVE_SCAN_FORWARD_WGSL,   'forward_scan'),
            scan_reduce: createComputePipeline(d, SELECTIVE_SCAN_FORWARD_WGSL,  'forward_reduce'),
        };
    }

    forward(xBuf: GPUBuffer, batch: number, seqLen: number): BlockForwardResult {
        const d = this.device;
        const { dModel, dState, dConv } = this.config;
        const D = this.dInner;
        const N = dState;
        const B = batch;
        const L = seqLen;
        const M = B * L;
        const R = this.dtRank;

        const cache = {} as BlockCache;

        const normOut  = createEmptyStorageBuffer(d, M * dModel * 4, true);
        const normInv  = createEmptyStorageBuffer(d, M * 4,          true);
        cache.normInv  = normInv;
        cache.normIn   = xBuf;

        {
            const params = new ArrayBuffer(16);
            new Uint32Array(params, 0, 2).set([M, dModel]);
            new Float32Array(params, 8, 1).set([1e-6]);
            const pBuf = createUniformBuffer(d, params);

            const bg = createBindGroup(d, this.pipelines['rmsnorm']!,
                [pBuf, xBuf, this.gpuWeights['normWeight']!, normOut, normInv]);
            dispatchKernel(d, this.pipelines['rmsnorm']!, bg, [cdiv(M, 64), 1, 1]);
        }

        const inProjOut = createEmptyStorageBuffer(d, M * 2 * D * 4, true);
        cache.normOut   = normOut;
        {
            const params = new Uint32Array([M, dModel, 2 * D]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, normOut, this.gpuWeights['wInProj']!, this.gpuWeights['bInProj']!, inProjOut]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(2 * D, 16), 1]);
        }

        const xConvIn  = createEmptyStorageBuffer(d, M * D * 4, true);
        const zBuf     = createEmptyStorageBuffer(d, M * D * 4, true);
        {
            const enc = d.createCommandEncoder();
            enc.copyBufferToBuffer(inProjOut, 0,           xConvIn, 0, M * D * 4);
            enc.copyBufferToBuffer(inProjOut, M * D * 4,   zBuf,    0, M * D * 4);
            d.queue.submit([enc.finish()]);
        }
        cache.zBuf = zBuf;

        const convOut = createEmptyStorageBuffer(d, M * D * 4, true);
        cache.xConvIn = xConvIn;
        {
            const params = new Uint32Array([L, D, dConv, B]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['conv1d']!,
                [pBuf, xConvIn, this.gpuWeights['wConv']!, this.gpuWeights['bConv']!, convOut]);
            dispatchKernel(d, this.pipelines['conv1d']!, bg, [cdiv(L, 16), cdiv(D, 16), B]);
        }

        const siluOut = createEmptyStorageBuffer(d, M * D * 4, true);
        cache.convOut = convOut;
        {
            const params = new Uint32Array([M * D]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['silu']!,
                [pBuf, convOut, siluOut]);
            dispatchKernel(d, this.pipelines['silu']!, bg, [cdiv(M * D, 256), 1, 1]);
        }

        const xProjOut = createEmptyStorageBuffer(d, M * (R + 2 * N) * 4, true);
        {
            const params = new Uint32Array([M, D, R + 2 * N]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, siluOut, this.gpuWeights['wXProj']!, this.gpuWeights['bXProj']!, xProjOut]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(R + 2 * N, 16), 1]);
        }

        const dtRaw = createEmptyStorageBuffer(d, M * R * 4,     true);
        const B_raw = createEmptyStorageBuffer(d, B * L * N * 4, true);
        const C_raw = createEmptyStorageBuffer(d, B * L * N * 4, true);
        {
            const enc = d.createCommandEncoder();
            enc.copyBufferToBuffer(xProjOut, 0,                  dtRaw, 0, M * R * 4);
            enc.copyBufferToBuffer(xProjOut, M * R * 4,          B_raw, 0, B * L * N * 4);
            enc.copyBufferToBuffer(xProjOut, M * (R + N) * 4,    C_raw, 0, B * L * N * 4);
            d.queue.submit([enc.finish()]);
        }

        const deltaFull = createEmptyStorageBuffer(d, M * D * 4, true);
        {
            const params = new Uint32Array([M, R, D]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, dtRaw, this.gpuWeights['wDtProj']!, this.gpuWeights['bDtProj']!, deltaFull]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(D, 16), 1]);
        }

        const scanY      = createEmptyStorageBuffer(d, B * L * D * 4,         true);
        const hCache     = createEmptyStorageBuffer(d, 2 * B * L * D * N * 4, true);
        cache.siluOut    = siluOut;
        cache.deltaFull  = deltaFull;
        cache.B_raw      = B_raw;
        cache.C_raw      = C_raw;
        cache.hCache     = hCache;

        {
            const params = new Uint32Array([L, N, D, B]).buffer;
            const pBuf   = createUniformBuffer(d, params);

            const bg = createBindGroup(d, this.pipelines['scan_fwd']!,
                [pBuf, siluOut, deltaFull, this.gpuWeights['A_log']!, B_raw, C_raw,
                 this.gpuWeights['D_vec']!, scanY, hCache]);
            dispatchKernel(d, this.pipelines['scan_fwd']!, bg,
                [cdiv(D, 8), cdiv(N, 8), B]);

            const bg2 = createBindGroup(d, this.pipelines['scan_reduce']!,
                [pBuf, siluOut, deltaFull, this.gpuWeights['A_log']!, B_raw, C_raw,
                 this.gpuWeights['D_vec']!, scanY, hCache]);
            dispatchKernel(d, this.pipelines['scan_reduce']!, bg2,
                [cdiv(L, 64), D, B]);
        }

        const siluZ   = createEmptyStorageBuffer(d, M * D * 4, true);
        const gatedOut = createEmptyStorageBuffer(d, M * D * 4, true);
        {
            const params = new Uint32Array([M * D]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['silu']!,
                [pBuf, zBuf, siluZ]);
            dispatchKernel(d, this.pipelines['silu']!, bg, [cdiv(M * D, 256), 1, 1]);

            const mulShader = /* wgsl */`
                @group(0) @binding(0) var<storage, read>       a : array<f32>;
                @group(0) @binding(1) var<storage, read>       b : array<f32>;
                @group(0) @binding(2) var<storage, read_write> c : array<f32>;
                @group(0) @binding(3) var<uniform>             n : u32;
                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    let i = gid.x;
                    if (i < n) { c[i] = a[i] * b[i]; }
                }
            `;
            const mulPipeline = createComputePipeline(d, mulShader, 'main');
            const nBuf = createUniformBuffer(d, new Uint32Array([M * D]).buffer);
            const bgMul = createBindGroup(d, mulPipeline,
                [scanY, siluZ, gatedOut, nBuf]);
            dispatchKernel(d, mulPipeline, bgMul, [cdiv(M * D, 256), 1, 1]);
        }

        const outProjOut = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            const params = new Uint32Array([M, D, dModel]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, gatedOut, this.gpuWeights['wOutProj']!, this.gpuWeights['bOutProj']!, outProjOut]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(dModel, 16), 1]);
        }

        const output = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            const addShader = /* wgsl */`
                @group(0) @binding(0) var<storage, read>       a : array<f32>;
                @group(0) @binding(1) var<storage, read>       b : array<f32>;
                @group(0) @binding(2) var<storage, read_write> c : array<f32>;
                @group(0) @binding(3) var<uniform>             n : u32;
                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    let i = gid.x;
                    if (i < n) { c[i] = a[i] + b[i]; }
                }
            `;
            const addPipeline = createComputePipeline(d, addShader, 'main');
            const nBuf = createUniformBuffer(d, new Uint32Array([M * dModel]).buffer);
            const bgAdd = createBindGroup(d, addPipeline,
                [outProjOut, xBuf, output, nBuf]);
            dispatchKernel(d, addPipeline, bgAdd, [cdiv(M * dModel, 256), 1, 1]);
        }

        return { output, cache };
    }

    parameters(): BlockParam[] {
        const { dModel, dState, dConv } = this.config;
        const D = this.dInner;
        const N = dState;
        const K = dConv;
        const R = this.dtRank;

        return [
            { buf: this.gpuWeights['wInProj']!,   numel: 2 * D * dModel, name: 'wInProj'   },
            { buf: this.gpuWeights['bInProj']!,   numel: 2 * D,          name: 'bInProj'   },
            { buf: this.gpuWeights['wConv']!,     numel: D * K,           name: 'wConv'     },
            { buf: this.gpuWeights['bConv']!,     numel: D,               name: 'bConv'     },
            { buf: this.gpuWeights['wXProj']!,    numel: (R + 2*N) * D,   name: 'wXProj'   },
            { buf: this.gpuWeights['bXProj']!,    numel: R + 2 * N,       name: 'bXProj'   },
            { buf: this.gpuWeights['wDtProj']!,   numel: D * R,           name: 'wDtProj'  },
            { buf: this.gpuWeights['bDtProj']!,   numel: D,               name: 'bDtProj'  },
            { buf: this.gpuWeights['A_log']!,     numel: D * N,           name: 'A_log'    },
            { buf: this.gpuWeights['D_vec']!,     numel: D,               name: 'D_vec'    },
            { buf: this.gpuWeights['wOutProj']!,  numel: dModel * D,      name: 'wOutProj' },
            { buf: this.gpuWeights['bOutProj']!,  numel: dModel,          name: 'bOutProj' },
            { buf: this.gpuWeights['normWeight']!, numel: dModel,          name: 'normWeight'},
        ];
    }

    setWSLAMode(enabled: boolean): void {
        this._wslaMode = enabled;
    }

    getTrainableParams(): BlockParam[] {
        if (this._wslaMode) {
            return [
                { buf: this.gpuWeights['wXProj']!, numel: this.wXProj.length, name: 'wXProj' },
                { buf: this.gpuWeights['bXProj']!, numel: this.bXProj.length, name: 'bXProj' },
            ];
        }
        return this.parameters();
    }
}
