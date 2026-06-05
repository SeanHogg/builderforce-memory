/**
 * tests/rng.test.ts
 * Determinism guarantees for the shared weight-init RNG.
 */

import { SeededRng, setInitSeed, randn, gaussianArray } from '../src/utils/rng.js';

afterEach(() => setInitSeed(undefined)); // always restore Math.random

test('SeededRng is deterministic for a given seed', () => {
    const a = new SeededRng(42);
    const b = new SeededRng(42);
    const seqA = Array.from({ length: 8 }, () => a.next());
    const seqB = Array.from({ length: 8 }, () => b.next());
    expect(seqA).toEqual(seqB);
});

test('different seeds produce different sequences', () => {
    const a = new SeededRng(1);
    const b = new SeededRng(2);
    expect(a.next()).not.toEqual(b.next());
});

test('SeededRng output is in [0, 1)', () => {
    const r = new SeededRng(123);
    for (let i = 0; i < 1000; i++) {
        const v = r.next();
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
    }
});

test('gaussianArray is reproducible under the same seed', () => {
    setInitSeed(7);
    const first = Array.from(gaussianArray(16, 0.02));
    setInitSeed(7);
    const second = Array.from(gaussianArray(16, 0.02));
    expect(first).toEqual(second);
});

test('gaussianArray differs across seeds', () => {
    setInitSeed(7);
    const a = Array.from(gaussianArray(16, 0.02));
    setInitSeed(8);
    const b = Array.from(gaussianArray(16, 0.02));
    expect(a).not.toEqual(b);
});

test('clearing the seed restores non-deterministic Math.random draws', () => {
    setInitSeed(undefined);
    // Two consecutive draws being identical is astronomically unlikely with
    // Math.random; this asserts we are no longer on a reset seed.
    expect(randn()).not.toEqual(randn());
});

test('std scales the Gaussian spread', () => {
    setInitSeed(99);
    const wide = gaussianArray(2000, 1.0);
    setInitSeed(99);
    const narrow = gaussianArray(2000, 0.01);
    const std = (xs: Float32Array) => {
        const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
        return Math.sqrt(xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length);
    };
    expect(std(wide)).toBeGreaterThan(std(narrow) * 10);
});
