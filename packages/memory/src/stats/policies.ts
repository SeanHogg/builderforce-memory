/**
 * Policies for the stats module (PRD 41 Part D / D13): Thompson sampling over
 * Beta-bandit arms. Draws are made from the caller-supplied seeded RNG (prng.ts),
 * so a schedule is reproducible under its seed. State-free: the RNG is the only
 * source of randomness.
 */

import { normalDraw, type Rng } from "./prng.js";

export interface BetaArm {
    alpha: number;
    beta: number;
}

/** Beta draw by the Gamma-ratio method. */
function betaDraw(alpha: number, beta: number, rng: Rng): number {
    const x = gammaDraw(alpha, rng);
    const y = gammaDraw(beta, rng);
    return x / (x + y);
}

/** Gamma(shape, 1) draw by Marsaglia's method; shape < 1 boosted by a fractional power. */
function gammaDraw(shape: number, rng: Rng): number {
    if (shape < 1) {
        const u = Math.max(rng(), 1e-12);
        return gammaDraw(shape + 1, rng) * Math.pow(u, 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
        // The module's ONE normal (prng.ts), stateless: a spare cached across calls would
        // leak one caller's draw into the next, and the same seed would not give the same pick.
        const z = normalDraw(rng);
        const v = Math.pow(1 + c * z, 3);
        if (v <= 0) continue;
        const u = rng();
        if (Math.log(u) < 0.5 * z * z + d - d * v + d * Math.log(v)) return d * v;
    }
}

/**
 * Sample one arm index by Thompson sampling: draw θᵢ ~ Beta(αᵢ, βᵢ) per arm and
 * return the argmax. Deterministic under the supplied `rng`.
 */
export function thompsonSample(arms: readonly BetaArm[], rng: Rng): number {
    let best = 0;
    let bestDraw = -1;
    for (let i = 0; i < arms.length; i++) {
        const draw = betaDraw(arms[i]!.alpha, arms[i]!.beta, rng);
        if (draw > bestDraw) {
            bestDraw = draw;
            best = i;
        }
    }
    return best;
}

/** Rank all arms by one Thompson draw each, descending. Deterministic under `rng`. */
export function thompsonRank(arms: readonly BetaArm[], rng: Rng): number[] {
    const draws = arms.map((arm) => betaDraw(arm.alpha, arm.beta, rng));
    return arms
        .map((_, i) => i)
        .sort((a, b) => draws[b]! - draws[a]! || a - b);
}
