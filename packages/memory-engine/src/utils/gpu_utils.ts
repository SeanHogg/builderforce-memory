/**
 * gpu_utils.ts – WebGPU device management and buffer helpers.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
const _gpu = globalThis as any;
const UNIFORM: number  = _gpu.GPUBufferUsage?.UNIFORM  ?? 0x40;
const STORAGE: number  = _gpu.GPUBufferUsage?.STORAGE  ?? 0x80;
const COPY_SRC: number = _gpu.GPUBufferUsage?.COPY_SRC ?? 0x04;
const COPY_DST: number = _gpu.GPUBufferUsage?.COPY_DST ?? 0x08;
const MAP_READ: number = _gpu.GPUBufferUsage?.MAP_READ ?? 0x01;

/**
 * Shape-keyed GPUBuffer pool (EVM-8). Hot paths (e.g. `readBuffer` staging)
 * allocated and destroyed a GPUBuffer on EVERY call — costly under load. The
 * pool reuses buffers keyed by (byteSize, usage): `acquire` hands back a free
 * buffer of that shape or creates one; `release` returns it for reuse. Buffers
 * of one shape are interchangeable, so reuse is always safe.
 */
export class BufferPool {
    private readonly _free = new Map<string, GPUBuffer[]>();
    constructor(private readonly device: GPUDevice) {}

    private _key(byteSize: number, usage: number): string {
        return `${byteSize}:${usage}`;
    }

    /** A free buffer of this shape, or a freshly created one. */
    acquire(byteSize: number, usage: number): GPUBuffer {
        const list = this._free.get(this._key(byteSize, usage));
        const reused = list?.pop();
        return reused ?? this.device.createBuffer({ size: byteSize, usage });
    }

    /** Return a buffer to the pool for later reuse (do not destroy it yourself). */
    release(buffer: GPUBuffer, byteSize: number, usage: number): void {
        const key = this._key(byteSize, usage);
        let list = this._free.get(key);
        if (!list) { list = []; this._free.set(key, list); }
        list.push(buffer);
    }

    /** Number of pooled (free) buffers across all shapes. */
    get freeCount(): number {
        let n = 0;
        for (const list of this._free.values()) n += list.length;
        return n;
    }

    /** Destroy every pooled buffer and empty the pool. */
    clear(): void {
        for (const list of this._free.values()) {
            for (const b of list) b.destroy();
        }
        this._free.clear();
    }
}

/** One staging-buffer pool per device, used by {@link readBuffer}. */
const _stagingPools = new WeakMap<GPUDevice, BufferPool>();
function stagingPool(device: GPUDevice): BufferPool {
    let pool = _stagingPools.get(device);
    if (!pool) { pool = new BufferPool(device); _stagingPools.set(device, pool); }
    return pool;
}

export interface InitWebGPUOptions {
  powerPreference?: 'high-performance' | 'low-power';
}

export interface InitWebGPUResult {
  device: GPUDevice;
  adapter: GPUAdapter;
}

export async function initWebGPU(opts: InitWebGPUOptions = {}): Promise<InitWebGPUResult> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
        throw new Error(
            'WebGPU is not available in this environment. ' +
            'Use Chrome 113+, Edge 113+, or Firefox Nightly with WebGPU enabled.'
        );
    }

    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: opts.powerPreference ?? 'high-performance',
    });

    if (!adapter) {
        throw new Error('Failed to acquire a GPUAdapter. Your GPU may not support WebGPU.');
    }

    const adapterLimits = adapter.limits;
    const requested3GB  = 3 * 1024 * 1024 * 1024;
    const device = await adapter.requestDevice({
        requiredLimits: {
            maxBufferSize: Math.min(
                requested3GB,
                adapterLimits.maxBufferSize
            ),
            maxStorageBufferBindingSize: Math.min(
                requested3GB,
                adapterLimits.maxStorageBufferBindingSize
            ),
            maxComputeInvocationsPerWorkgroup: Math.min(
                256,
                adapterLimits.maxComputeInvocationsPerWorkgroup
            ),
        },
    });

    device.lost.then((info) => {
        console.error('WebGPU device lost:', info.message);
    });

    return { device, adapter };
}

export function createStorageBuffer(device: GPUDevice, data: Float32Array | Uint32Array | number[], readable = false): GPUBuffer {
    const arr    = data instanceof Float32Array || data instanceof Uint32Array ? data : new Float32Array(data);
    const usage  = STORAGE | COPY_DST | (readable ? COPY_SRC : 0);
    const buffer = device.createBuffer({ size: arr.byteLength, usage, mappedAtCreation: true });
    if (arr instanceof Uint32Array) {
        new Uint32Array(buffer.getMappedRange()).set(arr);
    } else {
        new Float32Array(buffer.getMappedRange()).set(arr as Float32Array);
    }
    buffer.unmap();
    return buffer;
}

export function createEmptyStorageBuffer(device: GPUDevice, byteSize: number, readable = false): GPUBuffer {
    const usage = STORAGE | COPY_DST | (readable ? COPY_SRC : 0);
    return device.createBuffer({ size: byteSize, usage });
}

export function createUniformBuffer(device: GPUDevice, data: ArrayBuffer | ArrayBufferView): GPUBuffer {
    const bytes  = ArrayBuffer.isView(data) ? data.buffer : data;
    const buffer = device.createBuffer({
        size  : bytes.byteLength,
        usage : UNIFORM | COPY_DST,
        mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(bytes));
    buffer.unmap();
    return buffer;
}

export async function readBuffer(device: GPUDevice, srcBuffer: GPUBuffer, byteSize: number): Promise<Float32Array> {
    const MAP_READ_FLAG: number = _gpu.GPUMapMode?.READ ?? 0x01;
    // EVM-8: reuse a pooled staging buffer instead of allocating + destroying one
    // per call. After unmap a staging buffer is fully reusable for the next read.
    const stagingUsage = MAP_READ | COPY_DST;
    const pool = stagingPool(device);
    const stagingBuffer = pool.acquire(byteSize, stagingUsage);

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(srcBuffer, 0, stagingBuffer, 0, byteSize);
    device.queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(MAP_READ_FLAG);
    const result = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    pool.release(stagingBuffer, byteSize, stagingUsage);
    return result;
}

export function uploadBuffer(device: GPUDevice, buffer: GPUBuffer, data: Float32Array, byteOffset = 0): void {
    device.queue.writeBuffer(buffer, byteOffset, data.buffer, data.byteOffset, data.byteLength);
}

export function createComputePipeline(device: GPUDevice, wgslSource: string, entryPoint: string): GPUComputePipeline {
    const shaderModule = device.createShaderModule({ code: wgslSource });
    return device.createComputePipeline({
        layout : 'auto',
        compute: { module: shaderModule, entryPoint },
    });
}

export function createBindGroup(device: GPUDevice, pipeline: GPUComputePipeline, buffers: GPUBuffer[], groupIndex = 0): GPUBindGroup {
    const entries = buffers.map((buf, i) => ({
        binding : i,
        resource: { buffer: buf },
    }));
    return device.createBindGroup({
        layout : pipeline.getBindGroupLayout(groupIndex),
        entries,
    });
}

export function dispatchKernel(device: GPUDevice, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: [number, number, number]): void {
    const encoder = device.createCommandEncoder();
    const pass    = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
}

export function cdiv(a: number, b: number): number {
    return Math.ceil(a / b);
}
