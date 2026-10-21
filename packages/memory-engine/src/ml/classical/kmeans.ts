/**
 * ml/classical/kmeans.ts — k-means clustering (k-means++ seeding, Lloyd iterations).
 *
 * Deterministic for a given seed. A prediction is the nearest centroid; its
 * confidence is how much closer that centroid is than the runner-up.
 */
import type { MlPrediction, MlPredictor, MlSample } from "../types.js";
import { featureDim, seededUniform, squaredDistance, type Vec } from "./math.js";

export interface KMeansOptions {
  /** Clusters. Default round(√(n/2)), at least 1. */
  k?: number;
  maxIterations?: number;
  seed?: number;
}

export class KMeansModel implements MlPredictor {
  constructor(
    readonly centroids: Float32Array[],
    /** Sum of squared distances of the fitted points to their centroid. */
    readonly inertia: number,
  ) {}

  /** Nearest centroid index and the two smallest squared distances. */
  assign(x: Vec): { cluster: number; d1: number; d2: number } {
    let cluster = 0;
    let d1 = Infinity;
    let d2 = Infinity;
    this.centroids.forEach((c, i) => {
      const d = squaredDistance(x, c);
      if (d < d1) {
        d2 = d1;
        d1 = d;
        cluster = i;
      } else if (d < d2) d2 = d;
    });
    return { cluster, d1, d2 };
  }

  predict(x: Float32Array): MlPrediction {
    const { cluster, d1, d2 } = this.assign(x);
    const confidence = Number.isFinite(d2) && d2 > 0 ? 1 - Math.sqrt(d1) / Math.sqrt(d2) : 1;
    return { scores: [cluster], confidence: Math.max(0, Math.min(1, confidence)) };
  }
}

export function fitKMeans(samples: readonly MlSample[], opts: KMeansOptions = {}): KMeansModel {
  const X = samples.map((s) => s.x);
  const d = featureDim(X);
  const k = Math.min(X.length, Math.max(1, opts.k ?? Math.round(Math.sqrt(X.length / 2))));
  const rnd = seededUniform(opts.seed ?? 0x6b6d);

  // k-means++: first centroid uniformly, the rest ∝ squared distance to the nearest chosen one.
  const centroids: Float32Array[] = [Float32Array.from(X[Math.floor(rnd() * X.length)]!)];
  const nearest = X.map((x) => squaredDistance(x, centroids[0]!));
  while (centroids.length < k) {
    const total = nearest.reduce((a, b) => a + b, 0);
    let pick = 0;
    if (total > 0) {
      // Walk only points with weight: a zero-distance point (already a centroid) can
      // never be picked, even when rnd() returns 0 and r starts at 0.
      let r = rnd() * total;
      let last = -1;
      for (pick = 0; pick < X.length; pick++) {
        if (!(nearest[pick]! > 0)) continue;
        last = pick;
        r -= nearest[pick]!;
        if (r <= 0) break;
      }
      if (pick >= X.length) pick = last; // float round-off left r > 0: take the last weighted point
    } else {
      pick = centroids.length; // every point coincides with a centroid; take the next index
    }
    const c = Float32Array.from(X[pick]!);
    centroids.push(c);
    X.forEach((x, i) => (nearest[i] = Math.min(nearest[i]!, squaredDistance(x, c))));
  }

  const assign = new Int32Array(X.length).fill(-1);
  /** Assign every point to its nearest current centroid; true when any assignment moved. */
  const reassign = (): boolean => {
    let changed = false;
    const model = new KMeansModel(centroids, 0);
    X.forEach((x, i) => {
      const c = model.assign(x).cluster;
      if (assign[i] !== c) {
        assign[i] = c;
        changed = true;
      }
    });
    return changed;
  };
  const maxIt = opts.maxIterations ?? 100;
  let converged = false;
  for (let it = 0; it < maxIt; it++) {
    if (!reassign()) {
      converged = true;
      break;
    }
    const sums = centroids.map(() => new Float64Array(d));
    const counts = new Array<number>(k).fill(0);
    X.forEach((x, i) => {
      const c = assign[i]!;
      counts[c] = counts[c]! + 1;
      for (let j = 0; j < d; j++) sums[c]![j] = sums[c]![j]! + x[j]!;
    });
    // An emptied cluster keeps its previous centroid.
    centroids.forEach((c, ci) => {
      if (counts[ci]! > 0) for (let j = 0; j < d; j++) c[j] = sums[ci]![j]! / counts[ci]!;
    });
  }

  // Stopped on maxIterations: the centroids moved after the last assignment, so
  // assign once more — inertia (and the stored model) must describe the same state.
  if (!converged) reassign();

  let inertia = 0;
  X.forEach((x, i) => (inertia += squaredDistance(x, centroids[assign[i]!]!)));
  return new KMeansModel(centroids, inertia);
}
