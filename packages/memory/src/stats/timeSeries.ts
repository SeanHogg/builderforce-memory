/**
 * Time-series change detection for the stats module (PRD 41 Part D / D13): CUSUM
 * and Bayesian online change-point detection. Both are deterministic (fixed hazard,
 * bounded run length); EWMA itself is reused from ml/classical, not copied here.
 */

export interface CusumResult {
    /** First index where |cumulative sum| exceeds `threshold`, or -1. */
    alarmIndex: number;
    cumulative: number[];
}

/** Cumulative-sum detector on standardized deviations from `target`. */
export function cusum(
    values: readonly number[],
    target: number,
    options: { threshold?: number; sigma?: number; drift?: number } = {},
): CusumResult {
    const sigma = options.sigma ?? 1;
    const h = options.threshold ?? 5;
    const k = options.drift ?? 0.5;
    const cumulative: number[] = [];
    let cPlus = 0;
    let cMinus = 0;
    let alarmIndex = -1;
    for (let i = 0; i < values.length; i++) {
        const z = (values[i]! - target) / sigma;
        cPlus = Math.max(0, cPlus + z - k);
        cMinus = Math.min(0, cMinus + z + k);
        cumulative.push(cPlus + cMinus);
        if (alarmIndex === -1 && (cPlus > h || -cMinus > h)) alarmIndex = i;
    }
    return { alarmIndex, cumulative };
}

export interface ChangePointResult {
    /** MAP run length at the last observation; -1 when the series is empty. */
    currentRunLength: number;
    /** Posterior probability of a change point at each index (run length 0 mass). */
    changePointProbabilities: number[];
}

/**
 * Bayesian online change-point detection (Adams & MacKay, 2007), Gaussian with
 * unknown mean and known variance 1, conjugate N(0, 1) prior on the mean per run.
 * Each run length carries its own sufficient statistics (sum of observations), so a
 * level shift accumulates evidence for a young run. Run-length mass is truncated at
 * `maxRunLength` and renormalized; identical inputs give identical outputs.
 */
export function bayesianOnlineChangePoint(
    values: readonly number[],
    options: { hazard?: number; maxRunLength?: number } = {},
): ChangePointResult {
    const hazard = options.hazard ?? 1 / 250;
    const maxRunLength = options.maxRunLength ?? 500;
    let growth: number[] = [1];
    let sums: number[] = [0]; // sums[r] = Σ of observations inside a run of length r
    const changePointProbabilities: number[] = [];
    let currentRunLength = -1;
    for (let t = 0; t < values.length; t++) {
        const x = values[t]!;
        const predProbs = growth.map((g, r) => g * gaussianPredictive(x, sums[r]!, r));
        const evidence = predProbs.reduce((a, b) => a + b, 0) || 1e-300;
        const next = new Array<number>(maxRunLength).fill(0);
        const nextSums = new Array<number>(maxRunLength).fill(0);
        next[0] = hazard;
        nextSums[0] = 0;
        for (let r = 0; r < growth.length && r + 1 < maxRunLength; r++) {
            next[r + 1] = ((1 - hazard) * predProbs[r]!) / evidence;
            nextSums[r + 1] = sums[r]! + x;
        }
        growth = next;
        sums = nextSums;
        const norm = growth.reduce((a, b) => a + b, 0) || 1e-300;
        growth = growth.map((g) => g / norm);
        changePointProbabilities.push(growth[0]!);
        currentRunLength = growth.indexOf(Math.max(...growth));
    }
    return { currentRunLength, changePointProbabilities };
}

/**
 * Predictive N(x; sum/(r+1), 1 + 1/(r+1)) for a run of length r: the posterior mean
 * of the run's mean under the N(0, 1) prior is sum/(r+1), and the posterior variance
 * adds 1 to the observation variance 1.
 */
function gaussianPredictive(x: number, sum: number, runLength: number): number {
    const mean = sum / (runLength + 1);
    const variance = 1 + 1 / (runLength + 1);
    const z = (x - mean) / Math.sqrt(variance);
    return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI * variance);
}
