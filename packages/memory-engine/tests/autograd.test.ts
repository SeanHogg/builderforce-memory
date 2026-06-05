/**
 * tests/autograd.test.ts
 * Unit tests for the tape-based autograd engine.
 */

import {
    Tensor,
    enableGrad,
    noGrad,
    clearTape,
    recordOperation,
    backward,
    crossEntropyLoss,
    crossEntropyGrad,
} from '../src/training/autograd';

// ── crossEntropyLoss ──────────────────────────────────────────────────────────

test('crossEntropyLoss returns positive scalar', () => {
    const logits = new Float32Array([1.0, 2.0, 0.5, -1.0]);
    const loss   = crossEntropyLoss(logits, 1);
    expect(loss).toBeGreaterThan(0);
    expect(Number.isFinite(loss)).toBe(true);
});

test('crossEntropyLoss is lower when target is highest logit', () => {
    const logits = new Float32Array([0.0, 10.0, 0.0]);
    const lossCorrect   = crossEntropyLoss(logits, 1);   // target is highest
    const lossIncorrect = crossEntropyLoss(logits, 0);   // target is low-prob
    expect(lossCorrect).toBeLessThan(lossIncorrect);
});

test('crossEntropyLoss handles uniform logits', () => {
    const n      = 4;
    const logits = new Float32Array(n).fill(0.0);  // uniform
    const loss   = crossEntropyLoss(logits, 0);
    expect(Math.abs(loss - Math.log(n))).toBeLessThan(1e-4);
});

// ── crossEntropyGrad ──────────────────────────────────────────────────────────

test('crossEntropyGrad sums to 0', () => {
    const logits = new Float32Array([1.0, 2.0, 0.5, -1.0]);
    const grad   = crossEntropyGrad(logits, 1);
    const sum    = grad.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum)).toBeLessThan(1e-5);
});

test('crossEntropyGrad target component is negative', () => {
    const logits = new Float32Array([1.0, 2.0, 0.5]);
    const grad   = crossEntropyGrad(logits, 1);
    // dL/d logit_target = prob_target - 1 < 0 (since prob_target < 1)
    expect(grad[1]).toBeLessThan(0);
});

test('crossEntropyGrad non-target components are positive', () => {
    const logits = new Float32Array([1.0, 2.0, 0.5]);
    const grad   = crossEntropyGrad(logits, 1);
    expect(grad[0]).toBeGreaterThan(0);
    expect(grad[2]).toBeGreaterThan(0);
});

test('crossEntropyGrad length matches logits length', () => {
    const logits = new Float32Array(100).fill(0.1);
    const grad   = crossEntropyGrad(logits, 50);
    expect(grad.length).toBe(100);
});

// ── Tape recording ────────────────────────────────────────────────────────────

test('recordOperation returns -1 when grad is disabled', () => {
    noGrad();
    const idx = recordOperation(() => {});
    expect(idx).toBe(-1);
    enableGrad();
    clearTape();
});

test('recordOperation returns a valid index when grad is enabled', () => {
    enableGrad();
    clearTape();
    const idx = recordOperation(() => {});
    expect(idx).toBe(0);
    clearTape();
});

test('backward calls closures in reverse order', async () => {
    enableGrad();
    clearTape();

    const order: number[] = [];
    recordOperation(async () => { order.push(1); });
    recordOperation(async () => { order.push(2); });
    recordOperation(async () => { order.push(3); });

    await backward();
    expect(order).toEqual([3, 2, 1]);
});

test('backward clears the tape', async () => {
    enableGrad();
    clearTape();
    recordOperation(async () => {});
    await backward();
    // After backward the tape should be empty, so calling backward again is a no-op
    const order: number[] = [];
    await backward();
    expect(order.length).toBe(0);
});

// ── Tensor ────────────────────────────────────────────────────────────────────

test('Tensor.numel computes product of shape', () => {
    // No GPU needed for shape arithmetic
    const t = new Tensor(null, [2, 3, 4]);
    expect(t.numel).toBe(24);
});

test('Tensor.byteSize is numel * 4', () => {
    const t = new Tensor(null, [10]);
    expect(t.byteSize).toBe(40);
});

test('Tensor requiresGrad defaults to false', () => {
    const t = new Tensor(null, [4]);
    expect(t.requiresGrad).toBe(false);
});

test('Tensor requiresGrad can be set to true', () => {
    const t = new Tensor(null, [4], true);
    expect(t.requiresGrad).toBe(true);
});

test('Tensor.grad is null on construction', () => {
    const t = new Tensor(null, [5]);
    expect(t.grad).toBeNull();
});

test('Tensor._gradFn is null on construction', () => {
    const t = new Tensor(null, [3]);
    expect(t._gradFn).toBeNull();
});

test('Tensor.numel is 1 for scalar (empty shape)', () => {
    const t = new Tensor(null, []);
    expect(t.numel).toBe(1);
});

test('Tensor.numel handles high-rank shapes', () => {
    const t = new Tensor(null, [2, 3, 4, 5]);
    expect(t.numel).toBe(120);
});

// ── crossEntropyLoss – additional cases ──────────────────────────────────────

test('crossEntropyLoss with single-element logits is ~0', () => {
    const logits = new Float32Array([5.0]);
    const loss   = crossEntropyLoss(logits, 0);
    // Only one class → probability = 1 → -log(1) = 0
    expect(Math.abs(loss)).toBeLessThan(1e-5);
});

test('crossEntropyLoss is numerically stable with large logits', () => {
    const logits = new Float32Array([1000.0, 1001.0, 999.0]);
    const loss   = crossEntropyLoss(logits, 1);
    expect(Number.isFinite(loss)).toBe(true);
    expect(loss).toBeGreaterThanOrEqual(0);
});

test('crossEntropyLoss is numerically stable with very negative logits', () => {
    const logits = new Float32Array([-1000.0, -999.0, -1001.0]);
    const loss   = crossEntropyLoss(logits, 1);
    expect(Number.isFinite(loss)).toBe(true);
});

test('crossEntropyLoss two uniform logits equals log(2)', () => {
    const logits = new Float32Array([0.0, 0.0]);
    const loss   = crossEntropyLoss(logits, 0);
    expect(Math.abs(loss - Math.log(2))).toBeLessThan(1e-5);
});

// ── crossEntropyGrad – additional cases ──────────────────────────────────────

test('crossEntropyGrad with single-element logits returns [0]', () => {
    const logits = new Float32Array([3.0]);
    const grad   = crossEntropyGrad(logits, 0);
    expect(grad.length).toBe(1);
    // prob = 1 → grad = 1 - 1 = 0
    expect(Math.abs(grad[0]!)).toBeLessThan(1e-6);
});

test('crossEntropyGrad uniform logits gives symmetric non-target entries', () => {
    const logits = new Float32Array(4).fill(0.0);
    const grad   = crossEntropyGrad(logits, 0);
    expect(Math.abs(grad[1]! - grad[2]!)).toBeLessThan(1e-6);
    expect(Math.abs(grad[2]! - grad[3]!)).toBeLessThan(1e-6);
});

test('crossEntropyGrad values lie in [-1, 1]', () => {
    const logits = new Float32Array([2.0, -1.0, 0.5, 3.0, -2.0]);
    const grad   = crossEntropyGrad(logits, 2);
    for (const g of grad) {
        expect(g).toBeGreaterThanOrEqual(-1);
        expect(g).toBeLessThanOrEqual(1);
    }
});

// ── Tape – additional cases ───────────────────────────────────────────────────

test('noGrad then enableGrad restores recording', () => {
    enableGrad();
    clearTape();
    noGrad();
    expect(recordOperation(() => {})).toBe(-1);
    enableGrad();
    expect(recordOperation(() => {})).toBe(0);
    clearTape();
});

test('tape index increments with each recorded operation', () => {
    enableGrad();
    clearTape();
    const i0 = recordOperation(() => {});
    const i1 = recordOperation(() => {});
    const i2 = recordOperation(() => {});
    expect(i0).toBe(0);
    expect(i1).toBe(1);
    expect(i2).toBe(2);
    clearTape();
});

test('multiple clearTape calls are idempotent', () => {
    enableGrad();
    clearTape();
    recordOperation(() => {});
    clearTape();
    clearTape();
    const idx = recordOperation(() => {});
    expect(idx).toBe(0);
    clearTape();
});
