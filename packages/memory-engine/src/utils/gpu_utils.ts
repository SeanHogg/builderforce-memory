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
    const stagingBuffer = device.createBuffer({
        size  : byteSize,
        usage : MAP_READ | COPY_DST,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(srcBuffer, 0, stagingBuffer, 0, byteSize);
    device.queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(MAP_READ_FLAG);
    const result = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    stagingBuffer.destroy();
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
