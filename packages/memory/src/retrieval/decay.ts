/**
 * Recency decay — exponential half-life down-weighting of an older result's score.
 *
 * `score · e^(−λ·age)` with `λ = ln2 / halfLife`, so a result `halfLife` days old
 * keeps half its score. The maths is here; deciding how old a result IS (a dated
 * file name, an mtime, a `created_at` column) and which results never decay stays
 * with the caller, because that is a property of its corpus.
 *
 * Pure and zero-dependency.
 */

/** The decay rate for a half-life in days; 0 (no decay) for a non-positive or non-finite half-life. */
export function toDecayLambda(halfLifeDays: number): number {
    if (!Number.isFinite(halfLifeDays) || halfLifeDays <= 0) return 0;
    return Math.LN2 / halfLifeDays;
}

/** The factor in (0,1] a score is multiplied by at `ageInDays` (negative ages count as 0). */
export function calculateTemporalDecayMultiplier(params: { ageInDays: number; halfLifeDays: number }): number {
    const lambda = toDecayLambda(params.halfLifeDays);
    const clampedAge = Math.max(0, params.ageInDays);
    if (lambda <= 0 || !Number.isFinite(clampedAge)) return 1;
    return Math.exp(-lambda * clampedAge);
}

/** `score` decayed for its age. */
export function applyTemporalDecayToScore(params: { score: number; ageInDays: number; halfLifeDays: number }): number {
    return params.score * calculateTemporalDecayMultiplier(params);
}
