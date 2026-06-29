/**
 * tests/bufferpool.test.ts — EVM-8: shape-keyed GPUBuffer pool.
 * Uses a fake device so the pool logic is testable without WebGPU.
 */

import { BufferPool } from '../src/utils/gpu_utils';

interface FakeBuffer { id: number; size: number; usage: number; destroyed: boolean; destroy(): void }

function fakeDevice() {
    let counter = 0;
    const all: FakeBuffer[] = [];
    const device = {
        createBuffer: (desc: { size: number; usage: number }): FakeBuffer => {
            const b: FakeBuffer = { id: ++counter, size: desc.size, usage: desc.usage, destroyed: false, destroy() { this.destroyed = true; } };
            all.push(b);
            return b;
        },
    };
    return { device: device as unknown as GPUDevice, all, get created() { return counter; } };
}

test('acquire creates a buffer when the pool is empty', () => {
    const { device, all } = fakeDevice();
    const pool = new BufferPool(device);
    const b = pool.acquire(64, 0x09) as unknown as FakeBuffer;
    expect(b.size).toBe(64);
    expect(all).toHaveLength(1);
    expect(pool.freeCount).toBe(0);
});

test('release then acquire reuses the same buffer (no new allocation)', () => {
    const ctx = fakeDevice();
    const pool = new BufferPool(ctx.device);
    const b1 = pool.acquire(64, 0x09);
    pool.release(b1, 64, 0x09);
    expect(pool.freeCount).toBe(1);
    const b2 = pool.acquire(64, 0x09);
    expect(b2).toBe(b1);          // reused
    expect(ctx.created).toBe(1);  // only one ever created
});

test('different shapes do not collide', () => {
    const ctx = fakeDevice();
    const pool = new BufferPool(ctx.device);
    const a = pool.acquire(64, 0x09);
    pool.release(a, 64, 0x09);
    const b = pool.acquire(128, 0x09); // different size → fresh buffer
    expect(b).not.toBe(a);
    expect(ctx.created).toBe(2);
    const c = pool.acquire(64, 0x01); // same size, different usage → fresh
    expect(c).not.toBe(a);
    expect(ctx.created).toBe(3);
});

test('clear destroys all pooled buffers and empties the pool', () => {
    const ctx = fakeDevice();
    const pool = new BufferPool(ctx.device);
    pool.release(pool.acquire(64, 0x09), 64, 0x09);
    pool.release(pool.acquire(128, 0x09), 128, 0x09);
    expect(pool.freeCount).toBe(2);
    pool.clear();
    expect(pool.freeCount).toBe(0);
    expect(ctx.all.every((b) => b.destroyed)).toBe(true);
});
