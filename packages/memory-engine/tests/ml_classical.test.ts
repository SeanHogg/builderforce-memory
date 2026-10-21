/**
 * tests/ml_classical.test.ts — known-answer cases for every classical algorithm in
 * the ML task catalog (`src/ml/classical/`).
 */
import {
  fitClassical,
  fitKnn,
  fitNaiveBayes,
  fitLogistic,
  fitLinear,
  solveLinearSystem,
  fitKMeans,
  fitRobustZ,
  fitEwma,
  fitIsolationForest,
  fitGbm,
} from "../src/ml/classical/index.js";
import { averagePathLength, cosine, median } from "../src/ml/classical/math.js";
import type { ClassicalAlgorithm, MlSample } from "../src/ml/types.js";

const s = (x: number[], y = 0, weight?: number): MlSample => ({ x: Float32Array.from(x), y, ...(weight != null ? { weight } : {}) });
const v = (...x: number[]) => Float32Array.from(x);

/** Deterministic [0,1). */
function rnd(seed: number): () => number {
  let st = seed >>> 0 || 1;
  return () => {
    st = (Math.imul(1664525, st) + 1013904223) >>> 0;
    return st / 0x1_0000_0000;
  };
}

describe("math helpers", () => {
  test("cosine, median, c(n)", () => {
    expect(cosine([1, 0], [0, 1])).toBe(0);
    expect(cosine([1, 2], [2, 4])).toBeCloseTo(1, 6);
    expect(cosine([0, 0], [1, 1])).toBe(0);
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
    expect(averagePathLength(1)).toBe(0);
    expect(averagePathLength(2)).toBe(1);
    expect(averagePathLength(256)).toBeCloseTo(10.24, 1);
  });
});

describe("knn", () => {
  const data = [s([1, 0], 0), s([0.9, 0.1], 0), s([0.95, 0.05], 0), s([0, 1], 1), s([0.1, 0.9], 1), s([0.05, 0.95], 1)];

  test("classifies by similarity-weighted neighbour votes", () => {
    const m = fitKnn(data, { k: 3 });
    expect(m.predict(v(1, 0.02)).scores[0]).toBeGreaterThan(0.9);
    expect(m.predict(v(0.02, 1)).scores[1]).toBeGreaterThan(0.9);
    const p = m.predict(v(1, 1)).scores;
    expect(p[0]! + p[1]!).toBeCloseTo(1, 6);
  });

  test("nearest neighbour answers the dedupe query", () => {
    const m = fitKnn(data, { k: 1 });
    const n = m.nearest(v(2, 4))!;
    expect(m.labelOf(n.index)).toBe(1);
    expect(m.nearest(v(0.9, 0.1))!.similarity).toBeCloseTo(1, 6);
  });

  test("euclidean metric", () => {
    const m = fitKnn([s([0], 0), s([1], 0), s([10], 1), s([11], 1)], { k: 2, metric: "euclidean" });
    expect(m.predict(v(0.5)).scores[0]).toBeGreaterThan(0.9);
    expect(m.predict(v(10.5)).scores[1]).toBeGreaterThan(0.9);
  });
});

describe("naive-bayes (gaussian)", () => {
  test("separates two 1-D gaussians and honours priors", () => {
    const data = [0, 0.5, -0.5, 1, -1].map((x) => s([x], 0)).concat([10, 10.5, 9.5, 11, 9].map((x) => s([x], 1)));
    const m = fitNaiveBayes(data);
    expect(m.predict(v(0.2)).scores[0]).toBeGreaterThan(0.99);
    expect(m.predict(v(9.8)).scores[1]).toBeGreaterThan(0.99);
    // Midpoint: equal priors and variances → ~50/50.
    expect(m.predict(v(5)).scores[0]).toBeCloseTo(0.5, 1);
  });

  test("a constant feature does not divide by zero", () => {
    const m = fitNaiveBayes([s([1, 0], 0), s([1, 1], 0), s([1, 5], 1), s([1, 6], 1)]);
    const p = m.predict(v(1, 0.5)).scores;
    expect(p.every(Number.isFinite)).toBe(true);
    expect(p[0]).toBeGreaterThan(0.9);
  });
});

describe("logistic", () => {
  const data = [-3, -2, -1, -0.5, 0.5, 1, 2, 3].map((x) => s([x], x > 0 ? 1 : 0));

  test("fits a separable boundary", () => {
    const m = fitLogistic(data, { epochs: 500 });
    expect(m.predict(v(3)).scores[1]).toBeGreaterThan(0.9);
    expect(m.predict(v(-3)).scores[0]).toBeGreaterThan(0.9);
    expect(m.predict(v(0)).scores[1]).toBeCloseTo(0.5, 1);
    expect(m.loss(data)).toBeLessThan(0.3);
  });

  test("three classes", () => {
    const d3 = [s([0, 0], 0), s([0.2, 0.1], 0), s([5, 0], 1), s([5.2, 0.1], 1), s([0, 5], 2), s([0.1, 5.2], 2)];
    const m = fitLogistic(d3, { epochs: 800 });
    expect(m.predict(v(5, 0)).scores.indexOf(Math.max(...m.predict(v(5, 0)).scores))).toBe(1);
    expect(m.predict(v(0, 5)).scores.indexOf(Math.max(...m.predict(v(0, 5)).scores))).toBe(2);
  });

  test("params round-trip (the federation unit) and sample weights count", () => {
    const m = fitLogistic(data);
    const copy = fitLogistic(data, { epochs: 0 });
    copy.importParams(m.exportParams());
    expect(copy.predict(v(1.3)).scores).toEqual(m.predict(v(1.3)).scores);
    expect(() => copy.importParams([1, 2, 3])).toThrow(/expected/);
    // Up-weighting one mislabelled point drags the boundary toward it.
    const noisy = [...data, s([2.5], 0, 20)];
    expect(fitLogistic(noisy).predict(v(2.5)).scores[1]).toBeLessThan(m.predict(v(2.5)).scores[1]!);
  });
});

describe("linear (ridge, closed form)", () => {
  test("solves a known system", () => {
    const x = solveLinearSystem(Float64Array.of(2, 1, 1, 3), Float64Array.of(3, 5), 2);
    expect(x[0]).toBeCloseTo(0.8, 9);
    expect(x[1]).toBeCloseTo(1.4, 9);
    expect(() => solveLinearSystem(Float64Array.of(1, 2, 2, 4), Float64Array.of(1, 2), 2)).toThrow(/singular/);
  });

  test("recovers y = 2a − 3b + 5 exactly", () => {
    const r = rnd(3);
    const data = Array.from({ length: 30 }, () => {
      const a = r() * 10;
      const b = r() * 10;
      return s([a, b], 2 * a - 3 * b + 5);
    });
    const m = fitLinear(data);
    expect(m.weights[0]).toBeCloseTo(2, 3);
    expect(m.weights[1]).toBeCloseTo(-3, 3);
    expect(m.intercept).toBeCloseTo(5, 2);
    expect(m.r2).toBeCloseTo(1, 6);
    expect(m.predict(v(1, 1)).scores[0]).toBeCloseTo(4, 2);
    const back = fitLinear(data);
    back.importParams(m.exportParams());
    expect(back.value([2, 2])).toBeCloseTo(m.value([2, 2]), 6);
  });
});

describe("kmeans", () => {
  test("finds two well-separated blobs", () => {
    const r = rnd(9);
    const pts = [
      ...Array.from({ length: 20 }, () => s([r(), r()])),
      ...Array.from({ length: 20 }, () => s([10 + r(), 10 + r()])),
    ];
    const m = fitKMeans(pts, { k: 2 });
    const a = m.predict(v(0.5, 0.5));
    const b = m.predict(v(10.5, 10.5));
    expect(a.scores[0]).not.toBe(b.scores[0]);
    expect(a.confidence).toBeGreaterThan(0.9);
    const centres = m.centroids.map((c) => Math.round(c[0]! + c[1]!)).sort((x, y) => x - y);
    expect(centres).toEqual([1, 21]);
    // Within-blob spread only: 40 points with ≤ 0.5 squared distance each.
    expect(m.inertia).toBeLessThan(20);
  });

  test("is deterministic for a seed", () => {
    const pts = [0, 1, 2, 10, 11, 12].map((x) => s([x]));
    expect(fitKMeans(pts, { k: 2, seed: 5 }).centroids).toEqual(fitKMeans(pts, { k: 2, seed: 5 }).centroids);
  });
});

describe("robust-z", () => {
  test("flags a far point and is not dragged by an outlier in the history", () => {
    const hist = [10, 11, 9, 10, 12, 10, 11, 500].map((x) => s([x]));
    const m = fitRobustZ(hist);
    expect(m.center[0]).toBe(10.5);
    expect(m.predict(v(10)).scores[0]).toBeLessThan(1);
    expect(m.predict(v(100)).scores[0]).toBeGreaterThan(1);
  });

  test("a constant history still scores a change (MAD = 0 floor)", () => {
    const m = fitRobustZ([5, 5, 5, 5].map((x) => s([x])));
    expect(m.predict(v(5)).scores[0]).toBe(0);
    expect(m.predict(v(6)).scores[0]).toBeGreaterThan(1);
  });

  test("max over features", () => {
    const m = fitRobustZ([s([1, 100]), s([2, 101]), s([3, 99]), s([2, 100])]);
    expect(m.zScores([2, 100])).toEqual([0, 0]);
    expect(m.predict(v(2, 200)).scores[0]).toBeGreaterThan(1);
  });
});

describe("ewma", () => {
  test("known-answer update and z-score", () => {
    const m = fitEwma([s([0]), s([2])], { alpha: 0.5, minScaleFraction: 0 });
    expect(m.mean[0]).toBeCloseTo(1, 9);
    expect(m.variance[0]).toBeCloseTo(1, 9);
    expect(m.zScores([3])[0]).toBeCloseTo(2, 6);
    expect(m.predict(v(3)).scores[0]).toBeCloseTo(2 / 3, 6);
  });

  test("tracks drift: a level shift stops being anomalous once absorbed", () => {
    const series = [...Array.from({ length: 20 }, (_, i) => 10 + (i % 2)), ...Array.from({ length: 30 }, (_, i) => 20 + (i % 2))];
    const fresh = fitEwma(series.slice(0, 20).map((x) => s([x])));
    expect(fresh.predict(v(20)).scores[0]).toBeGreaterThan(1);
    const absorbed = fitEwma(series.map((x) => s([x])));
    expect(absorbed.predict(v(20)).scores[0]).toBeLessThan(1);
    expect(() => fitEwma([s([1])], { alpha: 0 })).toThrow(/alpha/);
  });
});

describe("isolation-forest", () => {
  test("isolates an outlier faster than inliers", () => {
    const r = rnd(21);
    const pts = Array.from({ length: 128 }, () => s([r(), r()]));
    const m = fitIsolationForest([...pts, s([8, 8])], { trees: 100 });
    const outlier = m.predict(v(8, 8)).scores[0]!;
    expect(outlier).toBeGreaterThan(1);
    const inliers = pts.slice(0, 40).map((p) => m.score(p.x)).sort((a, b) => a - b);
    expect(inliers[20]!).toBeLessThan(0.6);
    expect(m.score([8, 8])).toBeGreaterThan(inliers[39]!);
  });
});

describe("gbm-small", () => {
  const grid: MlSample[] = [];
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) grid.push(s([i / 7, j / 7], i / 7 > 0.5 && j / 7 > 0.5 ? 1 : 0));

  test("learns an AND of two thresholds (needs depth 2)", () => {
    const m = fitGbm(grid, { trees: 40, maxDepth: 2 });
    const acc = grid.filter((g) => (m.predict(g.x).scores[1]! >= 0.5 ? 1 : 0) === g.y).length / grid.length;
    expect(acc).toBe(1);
    expect(m.predict(v(0.9, 0.9)).scores[1]).toBeGreaterThan(0.8);
    expect(m.predict(v(0.1, 0.9)).scores[1]).toBeLessThan(0.2);
  });

  test("regression objective", () => {
    const data = [0, 0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9].map((x) => s([x], x > 0.5 ? 3 : -1));
    const m = fitGbm(data, { objective: "regression", trees: 60 });
    expect(m.predict(v(0.85)).scores[0]).toBeCloseTo(3, 1);
    expect(m.predict(v(0.15)).scores[0]).toBeCloseTo(-1, 1);
  });
});

describe("fitClassical dispatch", () => {
  const algos: ClassicalAlgorithm[] = ["knn", "naive-bayes", "logistic", "linear", "kmeans", "robust-z", "ewma", "isolation-forest", "gbm-small"];
  const data = [s([0, 0], 0), s([0.1, 0.2], 0), s([1, 1], 1), s([0.9, 1.1], 1), s([0.2, 0.1], 0), s([1.1, 0.9], 1)];

  test.each(algos)("%s fits and predicts finite scores", (algo) => {
    const p = fitClassical(algo, data).predict(v(0.5, 0.5));
    expect(p.scores.length).toBeGreaterThan(0);
    expect(p.scores.every(Number.isFinite)).toBe(true);
    expect(p.confidence).toBeGreaterThanOrEqual(0);
    expect(p.confidence).toBeLessThanOrEqual(1);
  });

  test("refuses an empty sample set", () => {
    expect(() => fitClassical("knn", [])).toThrow(/no samples/);
  });
});
