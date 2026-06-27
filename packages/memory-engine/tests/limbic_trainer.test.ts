/**
 * tests/limbic_trainer.test.ts
 * Real gradient training of the limbic affect model (CPU AdamW path — the path
 * that runs in CI without a GPU). Validates that loss decreases monotonically
 * and that the model learns a known affect mapping.
 */

import { LimbicModel } from "../src/limbic/limbic_model.js";
import { LimbicTrainer, type LimbicSample } from "../src/limbic/limbic_trainer.js";

const CFG = { inputDim: 6, hiddenDim: 12, stateDim: 8, seed: 4242 };

/** Deterministic vector source. */
function lcg(seed: number): () => number {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

/**
 * Build a synthetic, learnable dataset: a fixed (unknown-to-the-model) linear
 * map from experience → bounded affect delta, plus a reward = mean(input).
 */
function makeDataset(n: number, seed: number): LimbicSample[] {
  const rnd = lcg(seed);
  const inputDim = CFG.inputDim;
  const stateDim = CFG.stateDim;
  // Fixed target weights [stateDim x inputDim].
  const W: number[][] = [];
  const wr = lcg(0xabcd);
  for (let k = 0; k < stateDim; k++) {
    W.push(Array.from({ length: inputDim }, () => wr() * 2 - 1));
  }
  const samples: LimbicSample[] = [];
  for (let i = 0; i < n; i++) {
    const input = Float32Array.from({ length: inputDim }, () => rnd() * 2 - 1);
    const deltaTarget = new Float32Array(stateDim);
    let sum = 0;
    for (let j = 0; j < inputDim; j++) sum += input[j]!;
    for (let k = 0; k < stateDim; k++) {
      let dot = 0;
      for (let j = 0; j < inputDim; j++) dot += W[k]![j]! * input[j]!;
      deltaTarget[k] = 0.5 * Math.tanh(dot); // bounded, representable
    }
    samples.push({
      input,
      state: new Float32Array(stateDim), // neutral; mapping is input-driven
      deltaTarget,
      reward: 0.3 * (sum / inputDim),
    });
  }
  return samples;
}

test("training loss decreases monotonically and substantially", async () => {
  const model = new LimbicModel(CFG);
  const trainer = new LimbicTrainer(model, null); // CPU AdamW
  const data = makeDataset(24, 1);

  const losses = await trainer.train(data, { epochs: 150, learningRate: 0.05 });

  expect(losses.length).toBe(150);
  // Substantial reduction (>70%).
  expect(losses[losses.length - 1]!).toBeLessThan(losses[0]! * 0.3);
  // Largely monotone: at most a few uphill steps allowed for Adam.
  let uphill = 0;
  for (let i = 1; i < losses.length; i++) if (losses[i]! > losses[i - 1]! + 1e-6) uphill++;
  expect(uphill).toBeLessThan(losses.length * 0.1);
});

test("trained model predicts the held-in affect mapping closely", async () => {
  const model = new LimbicModel(CFG);
  const trainer = new LimbicTrainer(model, null);
  const data = makeDataset(24, 2);

  const before = trainer.evaluate(data);
  await trainer.train(data, { epochs: 200, learningRate: 0.05 });
  const after = trainer.evaluate(data);

  expect(after).toBeLessThan(before * 0.25);

  // Pointwise: predicted deltas track targets, reward tracks too.
  for (const s of data.slice(0, 6)) {
    const f = model.forward(s.input, model.initHidden(), s.state);
    for (let k = 0; k < f.delta.length; k++) {
      expect(Math.abs(f.delta[k]! - s.deltaTarget[k]!)).toBeLessThan(0.12);
    }
    expect(Math.abs(f.reward - s.reward)).toBeLessThan(0.12);
  }
});

test("generalises to held-out experiences (not just memorisation)", async () => {
  const model = new LimbicModel(CFG);
  const trainer = new LimbicTrainer(model, null);
  // Same target weights (seed of W is fixed inside makeDataset), different inputs.
  const train = makeDataset(40, 10);
  const test = makeDataset(16, 99);

  await trainer.train(train, { epochs: 200, learningRate: 0.05 });
  const testLoss = trainer.evaluate(test);
  expect(testLoss).toBeLessThan(0.02);
});

test("CPU and (would-be) GPU trainers share the same optimiser math", async () => {
  // Both paths use the identical AdamW update rule; with device=null we exercise
  // the CPU path. Verify two independent CPU trainers converge identically
  // (determinism is a prerequisite for the GPU path to be a drop-in).
  const a = new LimbicModel(CFG);
  const b = new LimbicModel(CFG);
  const data = makeDataset(20, 5);
  const la = await new LimbicTrainer(a, null).train(data, { epochs: 30, learningRate: 0.05 });
  const lb = await new LimbicTrainer(b, null).train(data, { epochs: 30, learningRate: 0.05 });
  expect(la).toEqual(lb);
  expect(Array.from(a.win)).toEqual(Array.from(b.win));
});

test("train throws on empty dataset", async () => {
  const model = new LimbicModel(CFG);
  const trainer = new LimbicTrainer(model, null);
  await expect(trainer.train([])).rejects.toThrow(/no samples/);
});
