/**
 * tests/kernels.test.ts
 * Smoke tests that verify the WGSL kernel sources export non-empty strings.
 * (Full GPU execution tests require a browser environment with WebGPU.)
 */

import { SELECTIVE_SCAN_FORWARD_WGSL, SELECTIVE_SCAN_BACKWARD_WGSL }
    from '../src/kernels/selective_scan';
import { CONV1D_FORWARD_WGSL, CONV1D_BACKWARD_WGSL }
    from '../src/kernels/conv1d';
import { LINEAR_FORWARD_WGSL, LINEAR_BACKWARD_WGSL }
    from '../src/kernels/linear_projection';
import { WEIGHT_UPDATE_WGSL, GRAD_CLIP_WGSL }
    from '../src/kernels/weight_update';
import { ACTIVATIONS_WGSL, ACTIVATIONS_BACKWARD_WGSL }
    from '../src/kernels/activations';

// ── Selective scan ────────────────────────────────────────────────────────────

test('SELECTIVE_SCAN_FORWARD_WGSL is a non-empty string', () => {
    expect(typeof SELECTIVE_SCAN_FORWARD_WGSL).toBe('string');
    expect(SELECTIVE_SCAN_FORWARD_WGSL.length).toBeGreaterThan(100);
});

test('SELECTIVE_SCAN_FORWARD_WGSL contains forward_scan entry point', () => {
    expect(SELECTIVE_SCAN_FORWARD_WGSL).toContain('fn forward_scan');
});

test('SELECTIVE_SCAN_FORWARD_WGSL contains forward_reduce entry point', () => {
    expect(SELECTIVE_SCAN_FORWARD_WGSL).toContain('fn forward_reduce');
});

test('SELECTIVE_SCAN_FORWARD_WGSL uses workgroup shared memory', () => {
    expect(SELECTIVE_SCAN_FORWARD_WGSL).toContain('var<workgroup>');
});

test('SELECTIVE_SCAN_BACKWARD_WGSL is a non-empty string', () => {
    expect(typeof SELECTIVE_SCAN_BACKWARD_WGSL).toBe('string');
    expect(SELECTIVE_SCAN_BACKWARD_WGSL.length).toBeGreaterThan(100);
});

test('SELECTIVE_SCAN_BACKWARD_WGSL contains backward_scan entry point', () => {
    expect(SELECTIVE_SCAN_BACKWARD_WGSL).toContain('fn backward_scan');
});

// ── Conv1D ────────────────────────────────────────────────────────────────────

test('CONV1D_FORWARD_WGSL is a non-empty string', () => {
    expect(typeof CONV1D_FORWARD_WGSL).toBe('string');
    expect(CONV1D_FORWARD_WGSL.length).toBeGreaterThan(50);
});

test('CONV1D_FORWARD_WGSL contains conv1d_forward entry point', () => {
    expect(CONV1D_FORWARD_WGSL).toContain('fn conv1d_forward');
});

test('CONV1D_FORWARD_WGSL implements causal padding', () => {
    expect(CONV1D_FORWARD_WGSL).toContain('t >= k');
});

test('CONV1D_BACKWARD_WGSL contains backward entry points', () => {
    expect(CONV1D_BACKWARD_WGSL).toContain('fn conv1d_backward_dx');
    expect(CONV1D_BACKWARD_WGSL).toContain('fn conv1d_backward_dw');
});

// ── Linear projection ─────────────────────────────────────────────────────────

test('LINEAR_FORWARD_WGSL is a non-empty string', () => {
    expect(typeof LINEAR_FORWARD_WGSL).toBe('string');
    expect(LINEAR_FORWARD_WGSL.length).toBeGreaterThan(50);
});

test('LINEAR_FORWARD_WGSL contains tiled matmul', () => {
    expect(LINEAR_FORWARD_WGSL).toContain('tile_X');
    expect(LINEAR_FORWARD_WGSL).toContain('tile_W');
});

test('LINEAR_FORWARD_WGSL contains linear_forward entry point', () => {
    expect(LINEAR_FORWARD_WGSL).toContain('fn linear_forward');
});

test('LINEAR_BACKWARD_WGSL contains all three backward entry points', () => {
    expect(LINEAR_BACKWARD_WGSL).toContain('fn linear_backward_dX');
    expect(LINEAR_BACKWARD_WGSL).toContain('fn linear_backward_dW');
    expect(LINEAR_BACKWARD_WGSL).toContain('fn linear_backward_db');
});

// ── Weight update (AdamW) ─────────────────────────────────────────────────────

test('WEIGHT_UPDATE_WGSL contains adamw_update entry point', () => {
    expect(WEIGHT_UPDATE_WGSL).toContain('fn adamw_update');
});

test('WEIGHT_UPDATE_WGSL implements bias correction', () => {
    expect(WEIGHT_UPDATE_WGSL).toContain('beta1_t');
    expect(WEIGHT_UPDATE_WGSL).toContain('beta2_t');
});

test('WEIGHT_UPDATE_WGSL implements weight decay', () => {
    expect(WEIGHT_UPDATE_WGSL).toContain('weight_decay');
});

test('GRAD_CLIP_WGSL contains both clip passes', () => {
    expect(GRAD_CLIP_WGSL).toContain('fn grad_norm_reduce');
    expect(GRAD_CLIP_WGSL).toContain('fn grad_clip_scale');
});

// ── Activations ───────────────────────────────────────────────────────────────

test('ACTIVATIONS_WGSL contains silu_forward', () => {
    expect(ACTIVATIONS_WGSL).toContain('fn silu_forward');
});

test('ACTIVATIONS_WGSL contains rmsnorm_forward', () => {
    expect(ACTIVATIONS_WGSL).toContain('fn rmsnorm_forward');
});

test('ACTIVATIONS_BACKWARD_WGSL contains silu_backward', () => {
    expect(ACTIVATIONS_BACKWARD_WGSL).toContain('fn silu_backward');
});

// ── WGSL structure validation ─────────────────────────────────────────────────

const ALL_KERNELS = [
    SELECTIVE_SCAN_FORWARD_WGSL,
    SELECTIVE_SCAN_BACKWARD_WGSL,
    CONV1D_FORWARD_WGSL,
    CONV1D_BACKWARD_WGSL,
    LINEAR_FORWARD_WGSL,
    LINEAR_BACKWARD_WGSL,
    WEIGHT_UPDATE_WGSL,
    GRAD_CLIP_WGSL,
    ACTIVATIONS_WGSL,
    ACTIVATIONS_BACKWARD_WGSL,
];

test('all kernels use @compute decorator', () => {
    for (const kernel of ALL_KERNELS) {
        expect(kernel).toContain('@compute');
    }
});

test('all kernels use @group(0) bindings', () => {
    for (const kernel of ALL_KERNELS) {
        expect(kernel).toContain('@group(0)');
    }
});

test('all kernels declare struct for uniform params', () => {
    for (const kernel of ALL_KERNELS) {
        expect(kernel).toMatch(/struct\s+\w+/);
    }
});

// ── Selective scan – deeper inspection ───────────────────────────────────────

test('SELECTIVE_SCAN_FORWARD_WGSL contains softplus helper', () => {
    expect(SELECTIVE_SCAN_FORWARD_WGSL).toContain('fn softplus');
});

test('SELECTIVE_SCAN_FORWARD_WGSL contains discretise_A helper', () => {
    expect(SELECTIVE_SCAN_FORWARD_WGSL).toContain('fn discretise_A');
});

test('SELECTIVE_SCAN_FORWARD_WGSL contains discretise_B helper', () => {
    expect(SELECTIVE_SCAN_FORWARD_WGSL).toContain('fn discretise_B');
});

test('SELECTIVE_SCAN_FORWARD_WGSL references ScanParams struct', () => {
    expect(SELECTIVE_SCAN_FORWARD_WGSL).toContain('ScanParams');
});

test('SELECTIVE_SCAN_FORWARD_WGSL declares h_cache binding', () => {
    expect(SELECTIVE_SCAN_FORWARD_WGSL).toContain('h_cache');
});

test('SELECTIVE_SCAN_BACKWARD_WGSL contains softplus_grad helper', () => {
    expect(SELECTIVE_SCAN_BACKWARD_WGSL).toContain('fn softplus_grad');
});

// ── Conv1D – deeper inspection ────────────────────────────────────────────────

test('CONV1D_FORWARD_WGSL references ConvParams struct', () => {
    expect(CONV1D_FORWARD_WGSL).toContain('ConvParams');
});

test('CONV1D_FORWARD_WGSL reads bias binding', () => {
    expect(CONV1D_FORWARD_WGSL).toContain('bias');
});

test('CONV1D_BACKWARD_WGSL is a non-empty string', () => {
    expect(typeof CONV1D_BACKWARD_WGSL).toBe('string');
    expect(CONV1D_BACKWARD_WGSL.length).toBeGreaterThan(50);
});

// ── Linear projection – deeper inspection ────────────────────────────────────

test('LINEAR_FORWARD_WGSL references LinearParams struct', () => {
    expect(LINEAR_FORWARD_WGSL).toContain('LinearParams');
});

test('LINEAR_FORWARD_WGSL adds bias to output', () => {
    expect(LINEAR_FORWARD_WGSL).toContain('bias');
});

test('LINEAR_BACKWARD_WGSL is a non-empty string', () => {
    expect(typeof LINEAR_BACKWARD_WGSL).toBe('string');
    expect(LINEAR_BACKWARD_WGSL.length).toBeGreaterThan(50);
});

// ── Weight update – deeper inspection ────────────────────────────────────────

test('WEIGHT_UPDATE_WGSL references AdamParams struct', () => {
    expect(WEIGHT_UPDATE_WGSL).toContain('AdamParams');
});

test('WEIGHT_UPDATE_WGSL is a non-empty string', () => {
    expect(typeof WEIGHT_UPDATE_WGSL).toBe('string');
    expect(WEIGHT_UPDATE_WGSL.length).toBeGreaterThan(50);
});

test('GRAD_CLIP_WGSL is a non-empty string', () => {
    expect(typeof GRAD_CLIP_WGSL).toBe('string');
    expect(GRAD_CLIP_WGSL.length).toBeGreaterThan(50);
});

// ── Activations – deeper inspection ──────────────────────────────────────────

test('ACTIVATIONS_WGSL contains RMSNormParams struct', () => {
    expect(ACTIVATIONS_WGSL).toContain('RMSNormParams');
});

test('ACTIVATIONS_WGSL contains eps for numerical stability', () => {
    expect(ACTIVATIONS_WGSL).toContain('eps');
});

test('ACTIVATIONS_BACKWARD_WGSL contains silu derivative formula', () => {
    // The backward kernel computes sigmoid(x) * (1 + x*(1 - sigmoid(x)))
    expect(ACTIVATIONS_BACKWARD_WGSL).toContain('sig');
});

test('ACTIVATIONS_BACKWARD_WGSL is a non-empty string', () => {
    expect(typeof ACTIVATIONS_BACKWARD_WGSL).toBe('string');
    expect(ACTIVATIONS_BACKWARD_WGSL.length).toBeGreaterThan(50);
});
