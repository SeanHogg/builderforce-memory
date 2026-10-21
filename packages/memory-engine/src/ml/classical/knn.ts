/**
 * ml/classical/knn.ts — k-nearest-neighbours classification over feature vectors.
 *
 * Votes are similarity-weighted (cosine similarity, or 1/(1+distance) for the
 * euclidean metric), so a close neighbour counts for more than a far one. The
 * same index answers "what is my nearest neighbour?" — the de-duplication query.
 */
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { classCount, cosine, featureDim, squaredDistance, topProbability, type Vec } from "./math.js";

export type KnnMetric = "cosine" | "euclidean";

export interface KnnOptions {
  /**
   * Neighbours that vote. Default min(5, ⌊√n⌋), at least 1: with a small set a fixed
   * 5 makes EVERY point vote, and the answer collapses to the class prior.
   */
  k?: number;
  metric?: KnnMetric;
  /** Number of classes (default: inferred from the labels, at least 2). */
  classes?: number;
}

export interface Neighbour {
  index: number;
  /** Higher = closer: cosine similarity, or 1/(1+distance) for euclidean. */
  similarity: number;
}

export class KnnModel implements MlPredictor {
  private readonly _x: Float32Array[] = [];
  private readonly _y: number[] = [];
  private readonly _w: number[] = [];

  constructor(
    readonly k: number,
    readonly metric: KnnMetric,
    readonly classes: number,
  ) {
    if (!(k >= 1)) throw new Error(`KnnModel: k must be ≥ 1 (got ${k})`);
  }

  /** How many points the index holds. */
  get size(): number {
    return this._x.length;
  }

  /** Add one labelled point (online — the index never needs a refit). */
  add(x: Vec, y = 0, weight = 1): void {
    this._x.push(Float32Array.from(x));
    this._y.push(y);
    this._w.push(weight);
  }

  similarity(a: Vec, b: Vec): number {
    return this.metric === "cosine" ? cosine(a, b) : 1 / (1 + Math.sqrt(squaredDistance(a, b)));
  }

  /** The `k` closest points, closest first — never the point at `exclude` (leave-one-out). */
  neighbours(x: Vec, k: number = this.k, exclude?: number): Neighbour[] {
    const all: Neighbour[] = [];
    for (let index = 0; index < this._x.length; index++) {
      if (index !== exclude) all.push({ index, similarity: this.similarity(x, this._x[index]!) });
    }
    all.sort((a, b) => b.similarity - a.similarity || a.index - b.index);
    return all.slice(0, Math.min(k, all.length));
  }

  /** The single closest point, or null when the index is empty. */
  nearest(x: Vec): Neighbour | null {
    return this.neighbours(x, 1)[0] ?? null;
  }

  /** The label of an indexed point. */
  labelOf(index: number): number {
    return this._y[index]!;
  }

  /**
   * Class probabilities from similarity-weighted votes (uniform when empty). With
   * `exclude`, the point at that index does not vote — the leave-one-out score of a
   * point that is itself in the index, which would otherwise vote for itself at
   * similarity 1.
   */
  proba(x: Vec, exclude?: number): number[] {
    const votes = new Array<number>(this.classes).fill(0);
    for (const n of this.neighbours(x, this.k, exclude)) {
      // Shift cosine into (0, 2] so an orthogonal neighbour still casts a small vote.
      const w = (this.metric === "cosine" ? n.similarity + 1 : n.similarity) * this._w[n.index]! + 1e-9;
      votes[this._y[n.index]!]! += w;
    }
    const sum = votes.reduce((a, b) => a + b, 0);
    return sum > 0 ? votes.map((v) => v / sum) : votes.map(() => 1 / this.classes);
  }

  predict(x: Float32Array): MlPrediction {
    const scores = this.proba(x);
    return { scores, confidence: topProbability(scores) };
  }

  /** This index as a predictor that leaves the point at `index` out of every vote. */
  excluding(index: number): MlPredictor {
    if (!(index >= 0 && index < this._x.length)) throw new Error(`KnnModel.excluding: no point ${index} (size ${this._x.length})`);
    return {
      predict: (x: Float32Array): MlPrediction => {
        const scores = this.proba(x, index);
        return { scores, confidence: topProbability(scores) };
      },
    };
  }
}

/** Index `samples` for kNN classification. */
export function fitKnn(samples: readonly MlSample[], opts: KnnOptions = {}): KnnModel {
  featureDim(samples.map((s) => s.x));
  const classes = opts.classes ?? classCount(samples.map((s) => s.y));
  const k = opts.k ?? Math.min(5, Math.max(1, Math.floor(Math.sqrt(samples.length))));
  const model = new KnnModel(Math.min(k, samples.length), opts.metric ?? "cosine", classes);
  for (const s of samples) model.add(s.x, s.y, s.weight ?? 1);
  return model;
}
