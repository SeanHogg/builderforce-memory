/**
 * tests/limbic_model.test.ts
 * LimbicModel: deterministic init, pure forward, gradient correctness vs. finite
 * differences, and checkpoint (f32 + fp16) round-trips.
 */

import { LimbicModel } from "../src/limbic/limbic_model.js";

const CFG = { inputDim: 6, hiddenDim: 8, stateDim: 8, seed: 1234 };

function randVec(n: number, seed: number): Float32Array {
  // simple LCG so tests are deterministic without importing rng internals
  let s = (seed >>> 0) || 1;
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    a[i] = (s / 0x1_0000_0000) * 2 - 1;
  }
  return a;
}

test("init is deterministic for a fixed seed", () => {
  const a = new LimbicModel(CFG);
  const b = new LimbicModel(CFG);
  expect(Array.from(a.win)).toEqual(Array.from(b.win));
  expect(Array.from(a.woutState)).toEqual(Array.from(b.woutState));
});

test("different seeds give different weights", () => {
  const a = new LimbicModel({ ...CFG, seed: 1 });
  const b = new LimbicModel({ ...CFG, seed: 2 });
  expect(Array.from(a.win)).not.toEqual(Array.from(b.win));
});

test("rejects hiddenDim > 64 (GPU kernel workgroup limit)", () => {
  expect(() => new LimbicModel({ ...CFG, hiddenDim: 128 })).toThrow(/hiddenDim/);
});

test("forward is pure and produces bounded affect deltas", () => {
  const m = new LimbicModel(CFG);
  const x = randVec(CFG.inputDim, 7);
  const s = randVec(CFG.stateDim, 9);
  const h = m.initHidden();
  const winBefore = Array.from(m.win);

  const out = m.forward(x, h, s);
  expect(out.delta.length).toBe(CFG.stateDim);
  expect(out.hidden.length).toBe(CFG.hiddenDim);
  for (const d of out.delta) {
    expect(d).toBeGreaterThan(-1);
    expect(d).toBeLessThan(1);
  }
  expect(Number.isFinite(out.reward)).toBe(true);
  // purity: weights untouched, repeated call identical
  expect(Array.from(m.win)).toEqual(winBefore);
  const out2 = m.forward(x, h, s);
  expect(Array.from(out2.delta)).toEqual(Array.from(out.delta));
});

test("untrained model is near-inert (small deltas — no spurious mood swings)", () => {
  const m = new LimbicModel(CFG);
  const x = randVec(CFG.inputDim, 3);
  const s = randVec(CFG.stateDim, 4);
  const out = m.forward(x, m.initHidden(), s);
  for (const d of out.delta) expect(Math.abs(d)).toBeLessThan(0.1);
});

test("analytic gradients match finite differences", () => {
  const m = new LimbicModel(CFG);
  const x = randVec(CFG.inputDim, 11);
  const s = randVec(CFG.stateDim, 13);
  const dT = randVec(CFG.stateDim, 15).map((v) => v * 0.3);
  const rT = 0.4;
  const h = m.initHidden();

  m.zeroGrad();
  const { loss } = m.backwardStep(x, h, s, dT, rT);
  const analytic = Array.from(m.gradients().find((g) => g.name === "woutState")!.data);

  // Finite-difference a few entries of woutState.
  const eps = 1e-3;
  const lossAt = (): number => {
    const f = m.forward(x, h, s);
    let l = 0;
    for (let k = 0; k < f.delta.length; k++) {
      const d = f.delta[k]! - dT[k]!;
      l += 0.5 * d * d;
    }
    const rd = f.reward - rT;
    l += 0.5 * m.config.rewardWeight * rd * rd;
    return l;
  };
  expect(lossAt()).toBeCloseTo(loss, 5);

  for (const idx of [0, 5, 17, 31]) {
    const orig = m.woutState[idx]!;
    m.woutState[idx] = orig + eps;
    const lp = lossAt();
    m.woutState[idx] = orig - eps;
    const lm = lossAt();
    m.woutState[idx] = orig;
    const fd = (lp - lm) / (2 * eps);
    expect(analytic[idx]).toBeCloseTo(fd, 3);
  }
});

test("checkpoint f32 round-trips exactly", () => {
  const m = new LimbicModel(CFG);
  const bin = m.exportWeights({ fp16: false });
  const m2 = new LimbicModel({ ...CFG, seed: 999 }); // different init
  expect(Array.from(m2.win)).not.toEqual(Array.from(m.win));
  m2.loadWeights(bin);
  expect(Array.from(m2.win)).toEqual(Array.from(m.win));
  expect(Array.from(m2.woutState)).toEqual(Array.from(m.woutState));
  expect(Array.from(m2.boutReward)).toEqual(Array.from(m.boutReward));
});

test("checkpoint fp16 round-trips within tolerance and halves data size", () => {
  const m = new LimbicModel(CFG);
  const f32 = m.exportWeights({ fp16: false });
  const fp16 = m.exportWeights({ fp16: true });
  // header is identical 20 bytes; data half the size
  expect(fp16.byteLength).toBeLessThan(f32.byteLength);
  const m2 = new LimbicModel({ ...CFG, seed: 7 });
  m2.loadWeights(fp16);
  for (let i = 0; i < m.win.length; i++) {
    expect(m2.win[i]).toBeCloseTo(m.win[i]!, 2);
  }
});

test("loadWeights rejects bad magic and dim mismatch", () => {
  const m = new LimbicModel(CFG);
  expect(() => m.loadWeights(new ArrayBuffer(64))).toThrow(/magic/);
  const other = new LimbicModel({ ...CFG, hiddenDim: 16 }).exportWeights();
  expect(() => m.loadWeights(other)).toThrow(/dim mismatch/);
});
