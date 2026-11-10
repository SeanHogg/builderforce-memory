/**
 * Deterministic seeded PRNG for the stats module (PRD 41 Part D, D13/B8): every
 * stochastic routine here draws from THIS generator, seeded by the caller, so the
 * same inputs give byte-identical outputs on any machine. Algorithm: xoshiro128**.
 */

export type Rng = () => number;

/**
 * SplitMix32-style seed expansion to four uint32 words. Never the all-zero state (the one fixed point of
 * xoshiro): the finaliser is a bijection on uint32 (xor-shifts and odd multipliers), and its four inputs
 * s + k·0x9e3779b9 (k = 1..4) are distinct mod 2^32, so at most one of the four words can be 0.
 */
function seedWords(seed: number): [number, number, number, number] {
    let s = seed >>> 0;
    const next = () => {
        s = (s + 0x9e3779b9) >>> 0;
        let z = s;
        z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
        z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
        z = (z ^ (z >>> 15)) >>> 0;
        return z;
    };
    return [next(), next(), next(), next()];
}

/**
 * A deterministic RNG in [0, 1), seeded once. `seed = 0` is a valid seed. State (a, b, c, d) = (s0, s1, s2, s3)
 * of the reference xoshiro128** (Blackman & Vigna): result = rotl(s1·5, 7)·9; t = s1 << 9 (a SHIFT, not a
 * rotate — rotating collapsed every stream to 0 within a few hundred draws).
 */
export function createRng(seed: number): Rng {
    let [a, b, c, d] = seedWords(seed);
    return () => {
        const result = (Math.imul(b, 5) << 7 | Math.imul(b, 5) >>> 25) >>> 0;
        const out = ((result * 9) >>> 0) / 4294967296;
        const t = (b << 9) >>> 0;
        c = (c ^ a) >>> 0;
        d = (d ^ b) >>> 0;
        b = (b ^ c) >>> 0;
        a = (a ^ d) >>> 0;
        c = (c ^ t) >>> 0;
        d = ((d << 11) | (d >>> 21)) >>> 0;
        return out;
    };
}

/** Box–Muller standard-normal draw from a uniform RNG (deterministic under seed). */
export function normalDraw(rng: Rng): number {
    const u1 = Math.max(rng(), Number.EPSILON);
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
