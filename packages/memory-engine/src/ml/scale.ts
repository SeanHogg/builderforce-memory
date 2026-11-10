/**
 * ml/scale.ts — the fixed feature scales every task row shares (re-exported by `features.ts`). A leaf: no imports, so a
 * light entry (the `./ml/tasks/check-scheduler` subpath) can use them without the feature extractor's embedder types.
 */

/**
 * A count or size on a fixed log scale: log1p(max(v, 0)) / log1p(ref). `ref` is
 * the value that maps to 1 (e.g. 100k tokens), so the scale is the same everywhere.
 */
export function logScale(v: number, ref: number): number {
  return Math.log1p(Math.max(0, v)) / Math.log1p(Math.max(ref, 1));
}

/** A ratio clamped to [0, max] — fill fractions, success rates. */
export function clampRatio(numerator: number, denominator: number, max = 4): number {
  if (!(denominator > 0)) return 0;
  return Math.max(0, Math.min(max, numerator / denominator));
}
