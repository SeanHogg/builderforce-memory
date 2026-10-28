/**
 * quantize.ts — int8 codes with one symmetric per-index scale.
 *
 * Graph traversal compares int8 codes with integer dot products. A sum of
 * int8×int8 products stays far inside 2^53, so it is exact in a JS number and
 * bit-identical on every platform — the float noise between a CPU and a WebGPU
 * embedding only matters where a component sits on a rounding boundary.
 *
 * The scale is fitted once (at the index's first compaction) and then fixed, so a
 * code never changes meaning while the index lives.
 */

/** Largest code magnitude; symmetric, so -128 is never produced. */
export const CODE_MAX = 127;

/** How far from unit norm a vector may be and still be taken as already normalised. */
const UNIT_TOLERANCE = 1e-3;

/**
 * A unit-norm Float32 copy of `v`. Every embedder behind the port already emits
 * L2-normalised vectors; those (|‖v‖² − 1| ≤ 1e-3) are copied as-is, so a
 * last-bit difference between two hosts' embeddings is not amplified by a
 * renormalisation. Anything else is normalised; a zero vector stays zero.
 */
export function toUnit(v: ArrayLike<number>): Float32Array {
    let sum = 0;
    for (let i = 0; i < v.length; i++) sum += v[i]! * v[i]!;
    const out = Float32Array.from(v);
    if (sum === 0 || Math.abs(sum - 1) <= UNIT_TOLERANCE) return out;
    const norm = Math.sqrt(sum);
    for (let i = 0; i < out.length; i++) out[i] = out[i]! / norm;
    return out;
}

/**
 * The symmetric scale for a set of (normalised) vectors: CODE_MAX over the largest
 * component magnitude, rounded UP to 1/64 first so a little float noise in the
 * maximum does not move the scale.
 */
export function fitScale(vectors: Iterable<ArrayLike<number>>): number {
    let maxAbs = 0;
    for (const v of vectors) {
        for (let i = 0; i < v.length; i++) {
            const a = Math.abs(v[i]!);
            if (a > maxAbs) maxAbs = a;
        }
    }
    const bound = Math.max(1, Math.ceil(maxAbs * 64)) / 64;
    return CODE_MAX / bound;
}

/** int8 codes of `v` at `scale`, clamped to ±CODE_MAX. */
export function quantize(v: ArrayLike<number>, scale: number): Int8Array {
    const out = new Int8Array(v.length);
    for (let i = 0; i < v.length; i++) {
        const q = Math.round(v[i]! * scale);
        out[i] = q > CODE_MAX ? CODE_MAX : q < -CODE_MAX ? -CODE_MAX : q;
    }
    return out;
}

/** Exact integer dot product of two code vectors. */
export function dotInt8(a: Int8Array, b: Int8Array): number {
    const n = Math.min(a.length, b.length);
    let dot = 0;
    for (let i = 0; i < n; i++) dot += a[i]! * b[i]!;
    return dot;
}

/**
 * Cosine of two code vectors: an exact integer dot over exact integer norms, then
 * IEEE `sqrt` and division — both correctly rounded, so the result is the same
 * everywhere. 0 when either is a zero vector.
 */
export function cosineInt8(a: Int8Array, b: Int8Array): number {
    const denom = Math.sqrt(dotInt8(a, a)) * Math.sqrt(dotInt8(b, b));
    return denom === 0 ? 0 : dotInt8(a, b) / denom;
}
