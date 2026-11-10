/**
 * src/contract/regression.prng.test.ts — the stats module's xoshiro128** (stats/prng.ts `createRng`, shipped wrong in
 * 2026.10.72) computed `t = rotl(s1, 9)` where the algorithm shifts (`t = s1 << 9`). The state degenerated: every
 * stream collapsed to 0 within ~76–200 draws, so the paired bootstrap (2,000 × n draws from one stream) returned
 * collapsed CIs and Thompson draws were biased over long streams.
 */

import { describe, expect, it } from '@jest/globals';
import { createRng, normalDraw } from '../stats/prng.js';
import { pairedBootstrapCi } from '../stats/index.js';

const MASK = 0xffffffffn;

/** The seed expansion createRng documents (SplitMix32-style), in BigInt so it shares no code path with prng.ts. */
function referenceSeedWords(seed: number): bigint[] {
    let s = BigInt(seed >>> 0);
    const words: bigint[] = [];
    for (let k = 0; k < 4; k++) {
        s = (s + 0x9e3779b9n) & MASK;
        let z = s;
        z = ((z ^ (z >> 16n)) * 0x21f0aaadn) & MASK;
        z = ((z ^ (z >> 15n)) * 0x735a2d97n) & MASK;
        z = (z ^ (z >> 15n)) & MASK;
        words.push(z);
    }
    return words;
}

/** xoshiro128** transcribed from Blackman & Vigna's reference C (xoshiro128starstar.c), uint32 arithmetic via BigInt. */
function referenceXoshiro128ss(state: readonly bigint[]): () => bigint {
    const s = [...state];
    const rotl = (x: bigint, k: bigint) => ((x << k) | (x >> (32n - k))) & MASK;
    return () => {
        const result = (rotl((s[1]! * 5n) & MASK, 7n) * 9n) & MASK;
        const t = (s[1]! << 9n) & MASK;
        s[2] = s[2]! ^ s[0]!;
        s[3] = s[3]! ^ s[1]!;
        s[1] = s[1]! ^ s[2]!;
        s[0] = s[0]! ^ s[3]!;
        s[2] = s[2]! ^ t;
        s[3] = rotl(s[3]!, 11n);
        return result;
    };
}

describe('regression: createRng is the reference xoshiro128**', () => {
    it('regression: matches an inline reference implementation over 10,000 draws for 50 seeds', () => {
        for (let seed = 0; seed < 50; seed++) {
            const rng = createRng(seed);
            const ref = referenceXoshiro128ss(referenceSeedWords(seed));
            for (let i = 0; i < 10_000; i++) {
                const expected = Number(ref()) / 4294967296;
                const got = rng();
                if (got !== expected) expect({ seed, i, got }).toEqual({ seed, i, got: expected });
            }
        }
    });

    it('regression: known answer for the fixed state (1, 2, 3, 4)', () => {
        // Hand-derived from the reference C. Step 1: s1 = 2 → rotl(10, 7)·9 = 1280·9 = 11520; t = 1024;
        // s = (1^6, 2^2, (3^1)^1024, rotl(4^2, 11)) = (7, 0, 1026, 12288).
        const ref = referenceXoshiro128ss([1n, 2n, 3n, 4n]);
        expect(ref()).toBe(11520n);
        // Step 2: s1 = 0 → result 0; t = 0; s = (7^12288, 0^1029, 1029, rotl(12288, 11)) = (12295, 1029, 1029, 25165824).
        expect(ref()).toBe(0n);
        // Step 3: s1 = 1029 → rotl(5145, 7)·9 = 658560·9 = 5927040.
        expect(ref()).toBe(5927040n);
    });

    it('regression: seedWords never yields the all-zero state', () => {
        for (const seed of [0, 1, -1, 0x61c88647, 0xffffffff, 2 ** 31, -(2 ** 31)]) {
            expect({ seed, zero: referenceSeedWords(seed).every((w) => w === 0n) }).toEqual({ seed, zero: false });
        }
    });
});

describe('regression: no stream collapses to 0', () => {
    it('regression: 200 seeds × 2,000 draws (the rotate collapsed within ~200) — no run of ≥ 8 exact zeros, zero fraction < 1e-6', () => {
        let zeros = 0;
        let total = 0;
        let longestRun = 0;
        for (let seed = 0; seed < 200; seed++) {
            const rng = createRng(seed);
            let run = 0;
            for (let i = 0; i < 2_000; i++) {
                total++;
                if (rng() === 0) {
                    zeros++;
                    run++;
                    if (run > longestRun) longestRun = run;
                } else run = 0;
            }
        }
        expect(longestRun).toBeLessThan(8);
        expect(zeros / total).toBeLessThan(1e-6);
    });

    it('regression: 1M draws from one seed have mean ≈ 1/2 and variance ≈ 1/12', () => {
        const rng = createRng(12345);
        const N = 1_000_000;
        let sum = 0;
        let sumSq = 0;
        for (let i = 0; i < N; i++) {
            const u = rng();
            sum += u;
            sumSq += u * u;
        }
        const mean = sum / N;
        const variance = sumSq / N - mean * mean;
        expect(Math.abs(mean - 0.5)).toBeLessThan(0.001);
        expect(Math.abs(variance - 1 / 12)).toBeLessThan(0.001);
    });
});

describe('regression: the paired bootstrap CI does not collapse', () => {
    it('regression: a noisy no-shift series (seeded normal noise, n = 30) gives a CI that contains 0', () => {
        const rng = createRng(2026);
        const before = Array.from({ length: 30 }, () => 10 + normalDraw(rng));
        const after = Array.from({ length: 30 }, () => 10 + normalDraw(rng));
        const ci = pairedBootstrapCi(before, after, { seed: 7 });
        expect(ci.low).toBeLessThan(0);
        expect(ci.high).toBeGreaterThan(0);
        // diff sd ≈ √2, se ≈ 0.26: a 95% interval is ≈ 1 wide, never a collapsed point.
        expect(ci.high - ci.low).toBeGreaterThan(0.5);
    });
});
