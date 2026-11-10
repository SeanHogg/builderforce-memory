/**
 * The binary logistic head for the stats module (PRD 41 Part D / D13): the ONE
 * numerically stable sigmoid, the probability of a stored head `σ(w·x + b)`, and one
 * online SGD step on the log-loss. Pure: every function returns new values and never
 * mutates its input.
 *
 * Batch fitting stays in memory-engine `ml/classical/logistic.ts` (gradient descent or
 * IRLS, L2, sample and class weights) and the Laplace/Platt fits in `fitting.ts`; this
 * is what a device needs to SCORE a fitted head and fold in one outcome at a time.
 */

/**
 * σ(z) = 1 / (1 + e^−z), evaluated so neither tail overflows. The engine's training
 * kernels keep their own (`memory-engine/src/training/cpu_ops.ts` `sigmoid`).
 */
export function sigmoid(z: number): number {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
}

/**
 * σ(b + Σᵢ wᵢ·xᵢ) over the features of `x` (accumulated in index order from the bias).
 * The caller owns the shape check: a head whose weights do not match its features has
 * no meaningful probability, and what to return then (0, ½, …) is the caller's policy.
 */
export function logisticProbability(weights: readonly number[], bias: number, x: readonly number[]): number {
    let z = bias;
    for (let i = 0; i < x.length; i++) z += weights[i]! * x[i]!;
    return sigmoid(z);
}

/**
 * One online SGD step on the log-loss for a single labelled example:
 * err = y − σ(w·x + b); wᵢ ← wᵢ + η·err·xᵢ; b ← b + η·err. Returns the new head.
 */
export function onlineLogisticStep(
    weights: readonly number[],
    bias: number,
    x: readonly number[],
    label: 0 | 1,
    learningRate: number,
): { weights: number[]; bias: number } {
    const err = label - logisticProbability(weights, bias, x);
    return {
        weights: weights.map((w, i) => w + learningRate * err * x[i]!),
        bias: bias + learningRate * err,
    };
}
