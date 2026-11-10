/**
 * Model fitting for the stats module (PRD 41 Part D / D13): Bayesian logistic with a
 * Laplace posterior, Platt scaling and isotonic regression (PAV). Pure, deterministic
 * (Newton iterations with fixed caps), and independent of ml/classical — these are the
 * calibration shapes the classical catalog does not carry.
 */

import { sigmoid } from "./logistic.js";
import { createRng, normalDraw, type Rng } from "./prng.js";

export interface LaplaceApproximation {
    mean: number[];
    covariance: number[][];
}

function invert2x2(m: number[][]): number[][] {
    const det = m[0]![0]! * m[1]![1]! - m[0]![1]! * m[1]![0]!;
    if (Math.abs(det) < 1e-12) {
        return [
            [1e6, 0],
            [0, 1e6],
        ];
    }
    return [
        [m[1]![1]! / det, -m[0]![1]! / det],
        [-m[1]![0]! / det, m[0]![0]! / det],
    ];
}

export interface BayesianLogisticOptions {
    /** L2 prior precision (1/σ²). Default 1. */
    priorPrecision?: number;
    maxIterations?: number;
}

function penalizedLogLikelihood(
    x: readonly number[],
    y: readonly number[],
    b0: number,
    b1: number,
    priorPrecision: number,
): number {
    let ll = -(priorPrecision / 2) * (b0 * b0 + b1 * b1);
    for (let i = 0; i < x.length; i++) {
        const eta = b0 + b1 * x[i]!;
        // log(1 + e^η) in a numerically stable form.
        const softplus = eta >= 0 ? eta + Math.log1p(Math.exp(-eta)) : Math.log1p(Math.exp(eta));
        ll += y[i]! * eta - softplus;
    }
    return ll;
}

/**
 * Bayesian logistic regression on one feature, fitted by Newton–Raphson with
 * step-halving on the penalized log-likelihood (IRLS is unstable on near-separable
 * data with a weak prior: the raw step can leap into the saturated boundary, where
 * the quadratic approximation no longer describes the surface). The posterior at
 * the mode is approximated by a Laplace (Gaussian) whose covariance is the inverse
 * observed information. Returns the mode, the covariance, a posterior-sample
 * log-odds sampler (deterministic under `rng`), and the posterior-mean predictive
 * probability at a point.
 */
export function fitBayesianLogistic(
    x: readonly number[],
    y: readonly number[],
    options: BayesianLogisticOptions = {},
): {
    intercept: number;
    slope: number;
    covariance: number[][];
    sampleLogOdds(x0: number, rng?: Rng): number;
    predict(x0: number): number;
} {
    const priorPrecision = options.priorPrecision ?? 1;
    const maxIterations = options.maxIterations ?? 50;
    let b0 = 0;
    let b1 = 0;
    let current = penalizedLogLikelihood(x, y, b0, b1, priorPrecision);
    let covariance: number[][] = [
        [1 / priorPrecision, 0],
        [0, 1 / priorPrecision],
    ];
    for (let iter = 0; iter < maxIterations; iter++) {
        let g0 = priorPrecision * b0;
        let g1 = priorPrecision * b1;
        let h00 = priorPrecision;
        let h01 = 0;
        let h11 = priorPrecision;
        for (let i = 0; i < x.length; i++) {
            const p = sigmoid(b0 + b1 * x[i]!);
            const w = Math.max(p * (1 - p), 1e-9);
            const r = y[i]! - p;
            g0 += r;
            g1 += r * x[i]!;
            h00 += w;
            h01 += w * x[i]!;
            h11 += w * x[i]! * x[i]!;
        }
        // Newton for the MODE of the penalized likelihood: δ = (−H)⁻¹g, where
        // −H = priorPrecision·I + Σ w·xxᵀ (the `h` accumulators, all positive).
        const step = invert2x2([
            [h00, h01],
            [h01, h11],
        ]);
        const d0 = step[0]![0]! * g0 + step[0]![1]! * g1;
        const d1 = step[1]![0]! * g0 + step[1]![1]! * g1;
        // Step-halving: accept the step only while the penalized likelihood rises.
        let scale = 1;
        for (let halving = 0; halving < 40; halving++) {
            const candidate = penalizedLogLikelihood(x, y, b0 + d0 * scale, b1 + d1 * scale, priorPrecision);
            if (candidate >= current - 1e-12) {
                b0 += d0 * scale;
                b1 += d1 * scale;
                current = candidate;
                break;
            }
            scale /= 2;
            if (halving === 39) {
                covariance = invert2x2([
                    [h00, h01],
                    [h01, h11],
                ]);
                return {
                    intercept: b0,
                    slope: b1,
                    covariance,
                    sampleLogOdds: (x0, rng = createRng(1)) => sampleLogOdds(b0, b1, covariance, x0, rng),
                    predict: (x0) => sigmoid(b0 + b1 * x0),
                };
            }
        }
        if (Math.abs(d0 * scale) < 1e-10 && Math.abs(d1 * scale) < 1e-10) {
            covariance = invert2x2([
                [h00, h01],
                [h01, h11],
            ]);
            break;
        }
    }
    return {
        intercept: b0,
        slope: b1,
        covariance,
        sampleLogOdds: (x0, rng = createRng(1)) => sampleLogOdds(b0, b1, covariance, x0, rng),
        predict: (x0) => sigmoid(b0 + b1 * x0),
    };
}

/** One posterior draw of the log-odds at `x0` from the Laplace approximation. */
function sampleLogOdds(b0: number, b1: number, covariance: number[][], x0: number, rng: Rng): number {
    const s00 = Math.sqrt(Math.max(covariance[0]![0]!, 0));
    const s10 = s00 > 1e-12 ? covariance[1]![0]! / s00 : 0;
    const s11 = Math.sqrt(Math.max(covariance[1]![1]! - s10 * s10, 0));
    const l0 = b0 + s00 * normalDraw(rng);
    const l1 = b1 + s10 * normalDraw(rng) + s11 * normalDraw(rng);
    return l0 + l1 * x0;
}

/** Platt scaling: fit p = σ(a·s + b) on scores `s` and labels `y` by Newton descent. */
export function fitPlattScaling(
    scores: readonly number[],
    y: readonly number[],
): { a: number; b: number; predict(score: number): number } {
    let a = 0;
    let b = Math.log((0.01 + sum(y)) / (0.01 + (y.length - sum(y))));
    const target = y.map((v) => (v === 1 ? 0.999 : 0.001));
    for (let iter = 0; iter < 100; iter++) {
        let ga = 0;
        let gb = 0;
        let haa = 1e-4;
        let hab = 0;
        let hbb = 1e-4;
        for (let i = 0; i < scores.length; i++) {
            const p = sigmoid(a * scores[i]! + b);
            const r = p - target[i]!;
            ga += r * scores[i]!;
            gb += r;
            const w = Math.max(p * (1 - p), 1e-9);
            haa += w * scores[i]! * scores[i]!;
            hab += w * scores[i]!;
            hbb += w;
        }
        const det = haa * hbb - hab * hab;
        const da = (hbb * -ga - hab * -gb) / det;
        const db = (-hab * -ga + haa * -gb) / det;
        a += da;
        b += db;
        if (Math.abs(da) < 1e-10 && Math.abs(db) < 1e-10) break;
    }
    return { a, b, predict: (score) => sigmoid(a * score + b) };
}

function sum(values: readonly number[]): number {
    let s = 0;
    for (const v of values) s += v;
    return s;
}

export interface IsotonicFit {
    /** Thresholds between fitted steps; `predict` returns fittedY[0] below x[0]. */
    thresholds: number[];
    values: number[];
    predict(x: number): number;
}

/** Isotonic regression by the pool-adjacent-violators algorithm. Deterministic. */
export function fitIsotonic(x: readonly number[], y: readonly number[]): IsotonicFit {
    const order = x.map((_, i) => i).sort((p, q) => x[p]! - x[q]!);
    const blocks: { weight: number; sum: number; value: number }[] = [];
    for (const i of order) {
        blocks.push({ weight: 1, sum: y[i]!, value: y[i]! });
        while (blocks.length >= 2 && blocks[blocks.length - 2]!.value > blocks[blocks.length - 1]!.value) {
            const top = blocks.pop()!;
            const below = blocks.pop()!;
            const merged = {
                weight: below.weight + top.weight,
                sum: below.sum + top.sum,
                value: (below.sum + top.sum) / (below.weight + top.weight),
            };
            blocks.push(merged);
        }
    }
    const thresholds: number[] = [];
    const values: number[] = [];
    for (let bi = 0; bi < blocks.length; bi++) {
        values.push(blocks[bi]!.value);
        if (bi < blocks.length - 1) {
            // Threshold at the midpoint of the x-range of this block.
            const startIdx = order.slice(
                blocks.slice(0, bi).reduce((acc, bl) => acc + bl.weight, 0),
            );
            const blockIdx = startIdx.slice(0, blocks[bi]!.weight);
            thresholds.push(x[blockIdx[blockIdx.length - 1]!]!);
        }
    }
    return {
        thresholds,
        values,
        predict: (x0) => {
            let v = values[0]!;
            for (let t = 0; t < thresholds.length; t++) {
                if (x0 >= thresholds[t]!) v = values[t + 1]!;
            }
            return v;
        },
    };
}
