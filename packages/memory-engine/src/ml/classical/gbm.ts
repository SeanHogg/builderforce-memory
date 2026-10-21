/**
 * ml/classical/gbm.ts — a small gradient-boosted tree ensemble.
 *
 * Depth-limited regression trees fitted to the loss gradient, one after another,
 * shrunk by a learning rate. Objectives: `binary` (log-loss, Newton leaf values)
 * and `regression` (squared error). Exhaustive split search over midpoints —
 * fine for the catalog's small numeric feature sets, not for wide embeddings.
 */
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { featureDim, sigmoid, type Vec } from "./math.js";

export type GbmObjective = "binary" | "regression";

export interface GbmOptions {
  objective?: GbmObjective;
  trees?: number;
  maxDepth?: number;
  learningRate?: number;
  /** Minimum summed sample weight in a leaf. Default 1. */
  minLeafWeight?: number;
  /** L2 on leaf values (Newton denominator). Default 1. */
  lambda?: number;
}

type RTree = { value: number } | { feature: number; split: number; left: RTree; right: RTree };

interface Row {
  x: Vec;
  g: number; // gradient
  h: number; // hessian
  w: number;
}

function leafValue(rows: readonly Row[], lambda: number): number {
  let g = 0;
  let h = 0;
  for (const r of rows) {
    g += r.w * r.g;
    h += r.w * r.h;
  }
  return -g / (h + lambda);
}

/** XGBoost-style structure score G²/(H+λ) of a node. */
function gain(g: number, h: number, lambda: number): number {
  return (g * g) / (h + lambda);
}

function buildTree(rows: Row[], depth: number, opts: Required<Omit<GbmOptions, "objective" | "trees" | "learningRate">>): RTree {
  if (depth >= opts.maxDepth || rows.length < 2) return { value: leafValue(rows, opts.lambda) };
  let G = 0;
  let H = 0;
  let W = 0;
  for (const r of rows) {
    G += r.w * r.g;
    H += r.w * r.h;
    W += r.w;
  }
  const parent = gain(G, H, opts.lambda);
  let best: { feature: number; split: number; score: number } | null = null;
  const d = rows[0]!.x.length;
  for (let f = 0; f < d; f++) {
    const sorted = [...rows].sort((a, b) => a.x[f]! - b.x[f]!);
    let gl = 0;
    let hl = 0;
    let wl = 0;
    for (let i = 0; i < sorted.length - 1; i++) {
      const r = sorted[i]!;
      gl += r.w * r.g;
      hl += r.w * r.h;
      wl += r.w;
      const a = r.x[f]!;
      const b = sorted[i + 1]!.x[f]!;
      if (a === b) continue;
      if (wl < opts.minLeafWeight || W - wl < opts.minLeafWeight) continue;
      const score = gain(gl, hl, opts.lambda) + gain(G - gl, H - hl, opts.lambda) - parent;
      if (score > 1e-12 && (!best || score > best.score)) best = { feature: f, split: (a + b) / 2, score };
    }
  }
  if (!best) return { value: leafValue(rows, opts.lambda) };
  const left = rows.filter((r) => r.x[best!.feature]! < best!.split);
  const right = rows.filter((r) => r.x[best!.feature]! >= best!.split);
  return { feature: best.feature, split: best.split, left: buildTree(left, depth + 1, opts), right: buildTree(right, depth + 1, opts) };
}

function treeValue(t: RTree, x: Vec): number {
  while (!("value" in t)) t = x[t.feature]! < t.split ? t.left : t.right;
  return t.value;
}

export class GbmModel implements MlPredictor {
  constructor(
    readonly objective: GbmObjective,
    readonly base: number,
    readonly trees: readonly RTree[],
    readonly learningRate: number,
  ) {}

  /** The raw additive score (log-odds for `binary`). */
  margin(x: Vec): number {
    let s = this.base;
    for (const t of this.trees) s += this.learningRate * treeValue(t, x);
    return s;
  }

  predict(x: Float32Array): MlPrediction {
    const m = this.margin(x);
    if (this.objective === "regression") return { scores: [m], confidence: 1 };
    const p = sigmoid(m);
    return { scores: [1 - p, p], confidence: Math.max(p, 1 - p) };
  }
}

export function fitGbm(samples: readonly MlSample[], opts: GbmOptions = {}): GbmModel {
  featureDim(samples.map((s) => s.x));
  const objective = opts.objective ?? "binary";
  const lr = opts.learningRate ?? 0.3;
  const treeOpts = { maxDepth: opts.maxDepth ?? 2, minLeafWeight: opts.minLeafWeight ?? 1, lambda: opts.lambda ?? 1 };

  let wSum = 0;
  let ySum = 0;
  for (const s of samples) {
    wSum += s.weight ?? 1;
    ySum += (s.weight ?? 1) * s.y;
  }
  const yMean = ySum / wSum;
  const base = objective === "binary" ? Math.log(Math.max(yMean, 1e-6) / Math.max(1 - yMean, 1e-6)) : yMean;
  const margins = samples.map(() => base);
  const trees: RTree[] = [];
  for (let t = 0; t < (opts.trees ?? 50); t++) {
    const rows: Row[] = samples.map((s, i) => {
      if (objective === "binary") {
        const p = sigmoid(margins[i]!);
        return { x: s.x, g: p - s.y, h: Math.max(p * (1 - p), 1e-6), w: s.weight ?? 1 };
      }
      return { x: s.x, g: margins[i]! - s.y, h: 1, w: s.weight ?? 1 };
    });
    const tree = buildTree(rows, 0, treeOpts);
    trees.push(tree);
    samples.forEach((s, i) => (margins[i] = margins[i]! + lr * treeValue(tree, s.x)));
  }
  return new GbmModel(objective, base, trees, lr);
}
