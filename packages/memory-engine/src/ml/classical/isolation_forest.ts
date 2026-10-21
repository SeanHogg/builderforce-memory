/**
 * ml/classical/isolation_forest.ts — Isolation Forest (Liu, Ting & Zhou 2008).
 *
 * Anomalies are few and different, so random axis-aligned splits isolate them in
 * fewer steps. Score s(x) = 2^(−E[h(x)] / c(ψ)): near 1 = anomalous, ≤ 0.5 =
 * normal. Prediction: `scores[0] = s / threshold` — ≥ 1 is anomalous.
 */
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { averagePathLength, featureDim, seededUniform, type Vec } from "./math.js";
import { anomalyConfidence } from "./robust_z.js";

export interface IsolationForestOptions {
  trees?: number;
  /** Subsample size ψ per tree. Default min(256, n). */
  sampleSize?: number;
  /** Raw score at which a point is anomalous. Default 0.6. */
  threshold?: number;
  seed?: number;
}

type ITree = { size: number } | { feature: number; split: number; left: ITree; right: ITree };

function buildTree(X: Vec[], depth: number, limit: number, rnd: () => number): ITree {
  if (depth >= limit || X.length <= 1) return { size: X.length };
  const d = X[0]!.length;
  // Only features that still vary can split; none left = an isolated leaf.
  const varying: { f: number; lo: number; hi: number }[] = [];
  for (let f = 0; f < d; f++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const x of X) {
      lo = Math.min(lo, x[f]!);
      hi = Math.max(hi, x[f]!);
    }
    if (hi > lo) varying.push({ f, lo, hi });
  }
  if (varying.length === 0) return { size: X.length };
  const { f, lo, hi } = varying[Math.floor(rnd() * varying.length)]!;
  const split = lo + rnd() * (hi - lo);
  const left = X.filter((x) => x[f]! < split);
  const right = X.filter((x) => x[f]! >= split);
  return { feature: f, split, left: buildTree(left, depth + 1, limit, rnd), right: buildTree(right, depth + 1, limit, rnd) };
}

function pathLength(tree: ITree, x: Vec, depth = 0): number {
  if ("size" in tree) return depth + averagePathLength(tree.size);
  return pathLength(x[tree.feature]! < tree.split ? tree.left : tree.right, x, depth + 1);
}

export class IsolationForestModel implements MlPredictor {
  constructor(
    readonly trees: readonly ITree[],
    readonly sampleSize: number,
    readonly threshold: number,
  ) {}

  /** The raw isolation score in (0, 1]. */
  score(x: Vec): number {
    const c = averagePathLength(this.sampleSize);
    if (c === 0) return 0.5;
    const mean = this.trees.reduce((s, t) => s + pathLength(t, x), 0) / this.trees.length;
    return Math.pow(2, -mean / c);
  }

  predict(x: Float32Array): MlPrediction {
    const s = this.score(x) / this.threshold;
    return { scores: [s], confidence: anomalyConfidence(s) };
  }
}

export function fitIsolationForest(samples: readonly MlSample[], opts: IsolationForestOptions = {}): IsolationForestModel {
  const X = samples.map((s) => s.x);
  featureDim(X);
  const rnd = seededUniform(opts.seed ?? 0x15f0);
  const psi = Math.min(opts.sampleSize ?? 256, X.length);
  const limit = Math.ceil(Math.log2(Math.max(psi, 2)));
  const trees: ITree[] = [];
  for (let t = 0; t < (opts.trees ?? 100); t++) {
    // Sample ψ points without replacement (partial Fisher–Yates over indices).
    const idx = X.map((_, i) => i);
    for (let i = 0; i < psi; i++) {
      const j = i + Math.floor(rnd() * (idx.length - i));
      [idx[i], idx[j]] = [idx[j]!, idx[i]!];
    }
    trees.push(buildTree(idx.slice(0, psi).map((i) => X[i]!), 0, limit, rnd));
  }
  return new IsolationForestModel(trees, psi, opts.threshold ?? 0.6);
}
