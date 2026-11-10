/**
 * tests/ml_classical_extensions.test.ts — the PRD 41 D13 extensions of `ml/classical`:
 * logistic class weights and the binary IRLS solver, the opt-in Mahalanobis term of the
 * anomaly composite (§D2.5), and the covariance helpers it fits on.
 */
import {
  choleskyInverse,
  fitAnomalyComposite,
  fitLogistic,
  fitMahalanobis,
  leaveOneOutMahalanobisThreshold,
  ledoitWolfCovariance,
  mahalanobisMinSamples,
} from "../src/ml/classical/index.js";
import type { MlSample } from "../src/ml/types.js";

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

/** Overlapping 1-D classes, so the optimum is finite. */
function noisyData(): MlSample[] {
  const r = rnd(17);
  const out: MlSample[] = [];
  for (let i = 0; i < 80; i++) {
    const x = r() * 6 - 3;
    const p = 1 / (1 + Math.exp(-(1.5 * x - 0.5)));
    out.push(s([x], r() < p ? 1 : 0));
  }
  return out;
}

describe("logistic: IRLS solver", () => {
  test("reaches the same optimum as long gradient descent on the same objective", () => {
    const data = noisyData();
    const irls = fitLogistic(data, { solver: "irls", l2: 1e-2 });
    const gd = fitLogistic(data, { l2: 1e-2, epochs: 3_000, learningRate: 2 });
    for (const x of [-2, -0.5, 0, 0.7, 2]) {
      expect(irls.predict(v(x)).scores[1]!).toBeCloseTo(gd.predict(v(x)).scores[1]!, 3);
    }
  });

  test("is a two-class softmax whose class-0 row is zero, and deterministic", () => {
    const data = noisyData();
    const a = fitLogistic(data, { solver: "irls" });
    const b = fitLogistic(data, { solver: "irls" });
    expect(Array.from(a.exportParams())).toEqual(Array.from(b.exportParams()));
    expect(a.weights[0]).toBe(0);
    expect(a.bias[0]).toBe(0);
    const p = a.predict(v(1)).scores;
    expect(p[0]! + p[1]!).toBeCloseTo(1, 6);
  });

  test("stays finite on separable data", () => {
    const data = [-3, -2, -1, 1, 2, 3].map((x) => s([x], x > 0 ? 1 : 0));
    const m = fitLogistic(data, { solver: "irls" });
    expect(Array.from(m.exportParams()).every(Number.isFinite)).toBe(true);
    expect(m.predict(v(3)).scores[1]).toBeGreaterThan(0.9);
  });

  test("refuses more than two classes", () => {
    expect(() => fitLogistic([s([0], 0), s([1], 1), s([2], 2)], { solver: "irls" })).toThrow(/two classes/);
  });
});

describe("logistic: class weights", () => {
  test("up-weighting a class pulls the boundary toward the other (both solvers)", () => {
    const data = noisyData();
    for (const solver of ["gradient-descent", "irls"] as const) {
      const plain = fitLogistic(data, { solver }).predict(v(0)).scores[1]!;
      const weighted = fitLogistic(data, { solver, classWeights: [1, 4] }).predict(v(0)).scores[1]!;
      expect(weighted).toBeGreaterThan(plain);
    }
  });

  test("a weight of 1 for every class is the unweighted fit", () => {
    const data = noisyData();
    expect(Array.from(fitLogistic(data, { classWeights: [1, 1] }).exportParams()))
      .toEqual(Array.from(fitLogistic(data).exportParams()));
  });
});

describe("anomaly-composite: Mahalanobis term", () => {
  /** x₂ tracks x₁: each feature alone spans [-3, 3], but x₁ − x₂ is tiny. */
  function correlated(): MlSample[] {
    const r = rnd(5);
    return Array.from({ length: 200 }, () => {
      const a = r() * 6 - 3;
      return s([a, a + (r() - 0.5) * 0.2]);
    });
  }

  test("flags an unusual COMBINATION that no per-feature detector sees", () => {
    const history = correlated();
    const off = v(1.5, -1.5);
    const plain = fitAnomalyComposite(history);
    expect(plain.mahalanobisScore(off)).toBeNull();
    expect(plain.predict(off).scores[0]).toBeLessThan(1);
    const shaped = fitAnomalyComposite(history, { mahalanobis: {} });
    expect(shaped.mahalanobisScore(off)!).toBeGreaterThan(1);
    expect(shaped.predict(off).scores[0]).toBeGreaterThan(1);
    // A point on the ridge is not anomalous to it.
    expect(shaped.mahalanobisScore(v(1.5, 1.5))!).toBeLessThan(1);
  });

  test("absent = the old two-detector score, unchanged", () => {
    const history = correlated();
    const a = fitAnomalyComposite(history);
    const b = fitAnomalyComposite(history, { mahalanobis: undefined });
    for (const x of [v(0, 0), v(2, 2), v(1.5, -1.5)]) expect(b.predict(x).scores).toEqual(a.predict(x).scores);
  });

  test("a constant history stays invertible", () => {
    const d = fitMahalanobis(Array.from({ length: 30 }, () => s([1, 1])))!;
    expect(d.distance(v(1, 1))).toBe(0);
    expect(Number.isFinite(d.distance(v(2, 1)))).toBe(true);
    expect(d.distance(v(2, 1))).toBeGreaterThan(d.threshold);
  });

  test("below the minimum history there is no term (default 30, never below d + 2)", () => {
    const history = correlated();
    // One run of history used to flag a 0.01 deviation; now it fits nothing.
    expect(fitMahalanobis(history.slice(0, 1))).toBeNull();
    expect(fitMahalanobis(history.slice(0, 29))).toBeNull();
    expect(fitMahalanobis(history.slice(0, 30))).not.toBeNull();
    const shortModel = fitAnomalyComposite(history.slice(0, 29), { mahalanobis: {} });
    expect(shortModel.mahalanobis).toBeNull();
    expect(shortModel.mahalanobisScore(v(1.5, -1.5))).toBeNull();
    // A smaller minSamples is honoured, but never below d + 2 (= 4 here).
    expect(mahalanobisMinSamples(2, { minSamples: 1 })).toBe(4);
    expect(fitMahalanobis(history.slice(0, 3), { minSamples: 1 })).toBeNull();
    expect(fitMahalanobis(history.slice(0, 4), { minSamples: 1 })).not.toBeNull();
  });

  test("a small-scale feature is not masked by a large-scale one (features are standardized)", () => {
    const r = rnd(23);
    const history = Array.from({ length: 200 }, () => s([r() * 1e4, r() * 0.01]));
    const m = fitMahalanobis(history)!;
    // Feature 2 five times past its whole range: anomalous; inside its range: not.
    expect(m.score(v(5000, 0.05))).toBeGreaterThan(1);
    expect(m.score(v(5000, 0.005))).toBeLessThan(1);
    // Symmetric: the large-scale feature is judged on its own scale too.
    expect(m.score(v(5e4, 0.005))).toBeGreaterThan(1);
  });

  test("the threshold is the empirical leave-one-out quantile, and deterministic", () => {
    const history = correlated();
    const rows = history.map((h) => h.x);
    const t99 = leaveOneOutMahalanobisThreshold(rows);
    expect(leaveOneOutMahalanobisThreshold(rows)).toBe(t99);
    const fitted = fitMahalanobis(history)!;
    expect(fitted.threshold).toBe(t99);
    expect(leaveOneOutMahalanobisThreshold(rows, { quantile: 0.5 })).toBeLessThan(t99);
    // A new draw from the same process rarely exceeds it (≈ 1% expected; well under 5% allowed).
    const r = rnd(99);
    const freshRows = Array.from({ length: 200 }, () => {
      const a = r() * 6 - 3;
      return v(a, a + (r() - 0.5) * 0.2);
    });
    const fresh = freshRows.map((x) => fitted.score(x));
    expect(fresh.filter((x) => x > 1).length).toBeLessThanOrEqual(10);
    // An explicit threshold wins.
    expect(fitMahalanobis(history, { threshold: 7 })!.threshold).toBe(7);
  });
});

describe("covariance", () => {
  test("choleskyInverse inverts an SPD matrix and refuses a non-PD one", () => {
    const a = [4, 2, 0.6, 2, 2, 0.5, 0.6, 0.5, 3];
    const inv = choleskyInverse(a, 3)!;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let sum = 0;
        for (let k = 0; k < 3; k++) sum += a[i * 3 + k]! * inv[k * 3 + j]!;
        expect(sum).toBeCloseTo(i === j ? 1 : 0, 10);
      }
    }
    expect(choleskyInverse([1, 2, 2, 1], 2)).toBeNull();
  });

  test("ledoitWolfCovariance is the engine's (memory ./stats re-exports it)", () => {
    const { covariance, shrinkage } = ledoitWolfCovariance([[0, 1], [1, 0], [1, 1], [0, 0]]);
    expect(shrinkage).toBeGreaterThanOrEqual(0);
    expect(shrinkage).toBeLessThanOrEqual(1);
    expect(covariance[0]![1]).toBeCloseTo(covariance[1]![0]!, 15);
  });
});

describe("logistic: class weight validation", () => {
  test("a non-finite or negative class weight is refused", () => {
    const data = noisyData();
    for (const bad of [[1, -1], [1, Number.NaN], [Number.POSITIVE_INFINITY, 1]]) {
      expect(() => fitLogistic(data, { classWeights: bad })).toThrow(/classWeights/);
      expect(() => fitLogistic(data, { solver: "irls", classWeights: bad })).toThrow(/classWeights/);
    }
  });
});
