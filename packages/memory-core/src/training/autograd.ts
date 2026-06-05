/**
 * autograd.ts – Lightweight tape-based automatic differentiation engine.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
const _gpu = globalThis as any;

interface TapeEntry {
  backward: () => void | Promise<void>;
}

let _tape: TapeEntry[] = [];
let _gradEnabled = true;

export class Tensor {
    data: GPUBuffer | null;
    shape: number[];
    numel: number;
    requiresGrad: boolean;
    grad: GPUBuffer | null;
    _gradFn: number | null;

    constructor(data: GPUBuffer | null, shape: number[], requiresGrad = false) {
        this.data         = data;
        this.shape        = shape;
        this.numel        = shape.reduce((a, b) => a * b, 1);
        this.requiresGrad = requiresGrad;
        this.grad         = null;
        this._gradFn      = null;
    }

    get byteSize(): number { return this.numel * 4; }

    zeroGrad(device: GPUDevice): void {
        if (this.grad) {
            device.queue.writeBuffer(this.grad, 0, new Float32Array(this.numel));
        }
    }

    destroy(): void {
        this.data?.destroy();
        this.grad?.destroy();
        this.data = null;
        this.grad = null;
    }
}

export function enableGrad(): void  { _gradEnabled = true;  }
export function noGrad(): void      { _gradEnabled = false; }
export function clearTape(): void   { _tape = []; }

export function recordOperation(backwardFn: () => void | Promise<void>): number {
    if (!_gradEnabled) return -1;
    _tape.push({ backward: backwardFn });
    return _tape.length - 1;
}

export async function backward(): Promise<void> {
    for (let i = _tape.length - 1; i >= 0; i--) {
        await _tape[i]!.backward();
    }
    clearTape();
}

export function ensureGradBuffer(device: GPUDevice, tensor: Tensor): void {
    if (!tensor.grad) {
        const STORAGE_USAGE: number = (_gpu.GPUBufferUsage?.STORAGE ?? 0x80) |
                                      (_gpu.GPUBufferUsage?.COPY_DST ?? 0x08) |
                                      (_gpu.GPUBufferUsage?.COPY_SRC ?? 0x04);
        tensor.grad = device.createBuffer({
            size  : tensor.byteSize,
            usage : STORAGE_USAGE,
        });
        device.queue.writeBuffer(tensor.grad, 0, new Float32Array(tensor.numel));
    }
}

export function allocateGradients(device: GPUDevice, tensors: Tensor[]): void {
    for (const t of tensors) {
        if (t.requiresGrad) ensureGradBuffer(device, t);
    }
}

export function zeroGradients(device: GPUDevice, tensors: Tensor[]): void {
    for (const t of tensors) {
        if (t.grad) {
            device.queue.writeBuffer(t.grad, 0, new Float32Array(t.numel));
        }
    }
}

export function onesLikeScalar(device: GPUDevice): GPUBuffer {
    const USAGE: number = (_gpu.GPUBufferUsage?.STORAGE ?? 0x80) |
                          (_gpu.GPUBufferUsage?.COPY_DST ?? 0x08);
    const buf = device.createBuffer({
        size  : 4,
        usage : USAGE,
        mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange()).set([1.0]);
    buf.unmap();
    return buf;
}

export function crossEntropyLoss(logits: Float32Array, targetId: number): number {
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i]! > maxLogit) maxLogit = logits[i]!;
    }
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) {
        sumExp += Math.exp(logits[i]! - maxLogit);
    }
    const logSumExp = Math.log(sumExp) + maxLogit;
    return logSumExp - logits[targetId]!;
}

export function crossEntropyGrad(logits: Float32Array, targetId: number): Float32Array {
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i]! > maxLogit) maxLogit = logits[i]!;
    }
    let sumExp = 0;
    const exp_shifted = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
        exp_shifted[i] = Math.exp(logits[i]! - maxLogit);
        sumExp += exp_shifted[i]!;
    }
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
        probs[i] = exp_shifted[i]! / sumExp;
    }
    probs[targetId] = (probs[targetId] ?? 0) - 1.0;
    return probs;
}
