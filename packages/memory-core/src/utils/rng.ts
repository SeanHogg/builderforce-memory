/**
 * rng.ts – shared, optionally-seeded random source for weight initialisation.
 *
 * Weight init across the model and every block used to duplicate the same
 * `Math.random()` Box–Muller draw. That made cold-start weights
 * non-reproducible across machines. This module centralises the draw and lets
 * the model install a deterministic seed for the duration of construction, so
 * the same `seed` yields byte-identical initial weights everywhere.
 *
 * The default (unseeded) source delegates to `Math.random`, preserving the
 * original behaviour for callers that don't request a seed.
 *
 * The seeded generator uses the same LCG constants as tools/generate-bin.js so
 * tooling and runtime agree on what a "seed N" model looks like.
 */

/** Deterministic linear-congruential generator (Numerical Recipes constants). */
export class SeededRng {
    private _s: number;

    constructor(seed: number) {
        // Avoid the zero fixed point; keep state in uint32 range.
        this._s = (seed >>> 0) || 1;
    }

    /** Next float in [0, 1). */
    next(): number {
        this._s = (Math.imul(1664525, this._s) + 1013904223) >>> 0;
        return this._s / 0x1_0000_0000;
    }
}

/** Active uniform source. Swapped by setInitSeed; defaults to Math.random. */
let _next: () => number = Math.random;

/**
 * Installs (or clears) the deterministic init seed.
 * Pass a number to make subsequent `randn`/`gaussianArray` draws reproducible;
 * pass `undefined` to restore the default `Math.random` source.
 *
 * Construction is synchronous, so a process-wide source is safe: seed before
 * building a model and clear afterwards.
 */
export function setInitSeed(seed: number | undefined): void {
    if (seed == null) {
        _next = Math.random;
    } else {
        const rng = new SeededRng(seed);
        _next = () => rng.next();
    }
}

/** Box–Muller Gaussian sample from the active source. */
export function randn(std = 1): number {
    const u1 = Math.max(_next(), 1e-12);
    const u2 = _next();
    return std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Returns a Float32Array of `n` Gaussian samples with the given standard deviation. */
export function gaussianArray(n: number, std: number): Float32Array {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = randn(std);
    return a;
}
