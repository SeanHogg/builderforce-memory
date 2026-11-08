/**
 * Scoring functions for the stats module (PRD 41 Part D / D13): Brier, expected
 * calibration error, PR-AUC, NDCG, Popt and effort-aware recall. All are pure,
 * deterministic orderings on their inputs.
 */

/** Brier score: mean squared error of probabilistic predictions. Lower is better. */
export function brierScore(probabilities: readonly number[], outcomes: readonly number[]): number {
    const n = Math.min(probabilities.length, outcomes.length);
    if (n === 0) return 0;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += (probabilities[i]! - outcomes[i]!) ** 2;
    return sum / n;
}

/**
 * Expected calibration error over `bins` equal-width probability bins: the weighted
 * mean of |accuracy − confidence| per bin.
 */
export function expectedCalibrationError(
    probabilities: readonly number[],
    outcomes: readonly number[],
    bins = 10,
): number {
    const n = Math.min(probabilities.length, outcomes.length);
    if (n === 0) return 0;
    let ece = 0;
    for (let b = 0; b < bins; b++) {
        const lo = b / bins;
        const hi = (b + 1) / bins;
        let count = 0;
        let confSum = 0;
        let accSum = 0;
        for (let i = 0; i < n; i++) {
            const p = probabilities[i]!;
            if (p >= lo && (p < hi || (b === bins - 1 && p === 1))) {
                count++;
                confSum += p;
                accSum += outcomes[i]!;
            }
        }
        if (count > 0) {
            ece += (count / n) * Math.abs(accSum / count - confSum / count);
        }
    }
    return ece;
}

/** Area under the precision–recall curve (trapezoid over the full recall range). */
export function prAuc(scores: readonly number[], labels: readonly number[]): number {
    const n = Math.min(scores.length, labels.length);
    if (n === 0) return 0;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b]! - scores[a]!);
    const totalPos = labels.reduce((a, _, i) => a + (i < n && labels[i] === 1 ? 1 : 0), 0);
    if (totalPos === 0) return 0;
    let tp = 0;
    let fp = 0;
    let prevRecall = 0;
    let prevPrecision = 1;
    let auc = 0;
    for (const i of order) {
        if (labels[i] === 1) tp++;
        else fp++;
        const recall = tp / totalPos;
        const precision = tp / (tp + fp);
        auc += (recall - prevRecall) * ((precision + prevPrecision) / 2);
        prevRecall = recall;
        prevPrecision = precision;
    }
    return auc;
}

/**
 * NDCG@k from graded relevance, with the log2 discount (position 1 = 1). The ideal
 * DCG orders by relevance descending.
 */
export function ndcgAtK(relevances: readonly number[], k: number): number {
    const take = relevances.slice(0, k);
    const dcg = take.reduce((acc, rel, i) => acc + rel / Math.log2(i + 2), 0);
    const ideal = [...relevances].sort((a, b) => b - a).slice(0, k)
        .reduce((acc, rel, i) => acc + rel / Math.log2(i + 2), 0);
    return ideal > 0 ? dcg / ideal : 0;
}

/**
 * Popt (performance optimum): the area under the effort–accuracy curve achieved in
 * the given order, normalized by the area of the optimal ordering (items sorted by
 * score descending). `efforts` are per-item costs, `scores` are per-item values.
 * Returns a value in (0, 1], 1 for the optimal order.
 */
export function popt(efforts: readonly number[], scores: readonly number[]): number {
    const n = Math.min(efforts.length, scores.length);
    if (n === 0) return 0;
    const area = (order: readonly number[]) => {
        let acc = 0;
        let cumEffort = 0;
        let cumScore = 0;
        let prevScore = 0;
        for (const i of order) {
            cumEffort += efforts[i]!;
            cumScore += scores[i]!;
            acc += efforts[i]! * (cumScore + prevScore) / 2;
            prevScore = cumScore;
        }
        void cumEffort;
        return acc;
    };
    const optimal = area(scores.map((_, i) => i).sort((a, b) => scores[b]! - scores[a]!));
    if (optimal <= 0) return 0;
    return area(scores.map((_, i) => i)) / optimal;
}

/**
 * Effort-aware recall: the share of the target found after spending `effortBudget`
 * fraction of the total effort, where items are examined in the given order.
 * `efforts` are per-item costs.
 */
export function effortAwareRecall(
    relevant: readonly boolean[],
    efforts: readonly number[],
    effortBudget: number,
): number {
    const totalRelevant = relevant.filter(Boolean).length;
    if (totalRelevant === 0) return 0;
    const totalEffort = efforts.reduce((a, b) => a + b, 0);
    const budget = totalEffort * effortBudget;
    let spent = 0;
    let found = 0;
    for (let i = 0; i < relevant.length; i++) {
        if (spent + efforts[i]! > budget) break;
        spent += efforts[i]!;
        if (relevant[i]) found++;
    }
    return found / totalRelevant;
}
