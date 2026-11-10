/**
 * src/contract/regression.thompsonDeterminism.test.ts — Thompson sampling (stats/policies.ts) used a polar Box–Muller
 * normal that cached its spare draw at MODULE scope. The spare from one caller's RNG was handed to the next caller, so
 * the same seed did not give the same schedule: it depended on who had sampled before. The draw is now the stats
 * module's one stateless normal (prng.ts `normalDraw`).
 */

import { describe, expect, it } from '@jest/globals';
import { createRng, thompsonRank, thompsonSample } from '../stats/index.js';

const ARMS = Array.from({ length: 6 }, () => ({ alpha: 5, beta: 5 }));

describe('regression: thompson sampling is deterministic under its own seed', () => {
    it('regression: a schedule does not depend on draws made earlier with ANOTHER rng', () => {
        for (let seed = 1; seed <= 20; seed++) {
            const alone = thompsonRank(ARMS, createRng(seed));
            // Another caller samples in between (an odd number of normal draws is what left a spare behind).
            thompsonSample([{ alpha: 3, beta: 4 }], createRng(10_000 + seed));
            const afterOthers = thompsonRank(ARMS, createRng(seed));
            expect({ seed, rank: afterOthers }).toEqual({ seed, rank: alone });
        }
    });
});
