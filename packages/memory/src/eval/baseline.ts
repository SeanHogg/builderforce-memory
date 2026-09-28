/**
 * eval/baseline.ts — gating a candidate RELATIVE to a baseline.
 *
 * "Good enough to launch" is often not an absolute score but a fraction of what an
 * incumbent scores on the same suite: a small model may serve a task once it reaches
 * 90% of the frontier model it replaces. That comparison only means something when
 * both reports came from the SAME eval (same dataset, same cases), and when the
 * baseline scored above zero. These are the rules for it, used by
 * {@link evaluateGate}'s `minBaselineRatio` and by any caller that stores the two
 * scores and re-checks the bar later without the reports.
 */

/** The three facts of a report that a baseline comparison reads. */
export interface EvalReportSummary {
    dataset: string;
    meanScore: number;
    /** The cases array, or just its length. */
    cases: number | readonly unknown[];
}

/** Float tolerance for a ratio bar, so an exact 90% (0.81 / 0.9) meets a 0.9 bar. */
export const BASELINE_RATIO_EPSILON = 1e-9;

/**
 * `score ÷ baselineScore`, or null when the pair cannot be compared — a non-finite or
 * negative score, or a baseline at or below zero, proves nothing.
 */
export function baselineRatio(score: number, baselineScore: number): number | null {
    if (!Number.isFinite(score) || !Number.isFinite(baselineScore) || score < 0 || baselineScore <= 0) return null;
    return score / baselineScore;
}

/** True when `ratio` reaches `minRatio` (within {@link BASELINE_RATIO_EPSILON}). A null ratio never does. */
export function meetsBaselineRatio(ratio: number | null, minRatio: number): boolean {
    return ratio !== null && ratio + BASELINE_RATIO_EPSILON >= minRatio;
}

/**
 * Why two reports are not the same eval, or null when they are. "The same eval" is
 * enforced, not assumed: a ratio of scores from different suites compares nothing.
 */
export function sameEvalProblem(candidate: EvalReportSummary, baseline: EvalReportSummary): string | null {
    const count = (c: EvalReportSummary['cases']) => (typeof c === 'number' ? c : c.length);
    if (candidate.dataset !== baseline.dataset) {
        return `reports are from different evals ('${candidate.dataset}' vs '${baseline.dataset}')`;
    }
    if (count(candidate.cases) !== count(baseline.cases)) {
        return `reports scored different case counts (${count(candidate.cases)} vs ${count(baseline.cases)})`;
    }
    return null;
}
