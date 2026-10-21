/**
 * ml/classical/math.ts — the small vector helpers every classical algorithm shares.
 * Pure, allocation-light, deterministic.
 */
import { SeededRng } from "../../utils/rng.js";

/** The engine's ONE sigmoid / softmax (`training/cpu_ops.ts`). */
export { sigmoid, softmax } from "../../training/cpu_ops.js";

export type Vec = ArrayLike<number>;

export function dot(a: Vec, b: Vec): number {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i]! * b[i]!;
  return s;
}

export function norm(a: Vec): number {
  return Math.sqrt(dot(a, a));
}

/** Cosine similarity in [-1, 1]; 0 when either vector is zero. */
export function cosine(a: Vec, b: Vec): number {
  const na = norm(a);
  const nb = norm(b);
  return na > 0 && nb > 0 ? dot(a, b) / (na * nb) : 0;
}

export function squaredDistance(a: Vec, b: Vec): number {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = a[i]! - b[i]!;
    s += d * d;
  }
  return s;
}

/** Median of a numeric list (copy-sorted). NaN for an empty list. */
export function median(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

/** Index of the largest value (first on ties). */
export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i]! > values[best]!) best = i;
  return best;
}

/** Number of distinct classes implied by integer labels 0..k-1. */
export function classCount(labels: readonly number[]): number {
  let k = 0;
  for (const y of labels) {
    if (!Number.isInteger(y) || y < 0) throw new Error(`class label must be a non-negative integer, got ${y}`);
    k = Math.max(k, y + 1);
  }
  return Math.max(k, 2);
}

/** Feature dimension of a non-empty sample matrix; throws on ragged rows. */
export function featureDim(X: readonly Vec[]): number {
  if (X.length === 0) throw new Error("no samples");
  const d = X[0]!.length;
  for (const x of X) if (x.length !== d) throw new Error(`ragged features: expected ${d}, got ${x.length}`);
  return d;
}

/** Confidence of a probability vector: its top probability. */
export function topProbability(p: readonly number[]): number {
  return p.length ? Math.max(...p) : 0;
}

/** A deterministic [0,1) source for algorithms that sample — the engine's ONE seeded RNG. */
export function seededUniform(seed: number): () => number {
  const rng = new SeededRng(seed);
  return () => rng.next();
}

/**
 * Average path length of an unsuccessful BST search over n points — the
 * isolation-forest normaliser c(n).
 */
export function averagePathLength(n: number): number {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  const harmonic = Math.log(n - 1) + 0.5772156649;
  return 2 * harmonic - (2 * (n - 1)) / n;
}
