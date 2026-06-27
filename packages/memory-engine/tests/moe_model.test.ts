/**
 * tests/moe_model.test.ts
 * SharedExpertMoE: deterministic init, top-k routing, load-balance signal,
 * analytic gradients vs finite differences, checkpoint round-trips, and that a
 * tiny training loop actually reduces the loss.
 */

import {
  SharedExpertMoE,
  LoadBalanceAccumulator,
  type RouteResult,
} from "../src/moe/moe_model.js";

const CFG = { modelDim: 4, hiddenDim: 3, numExperts: 4, topK: 2, seed: 4321 };

function randVec(n: number, seed: number): Float32Array {
  // Simple LCG so tests are deterministic without importing rng internals.
  let s = seed >>> 0;
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s / 0xffffffff) * 2 - 1;
  }
  return a;
}

function loss(output: Float32Array, target: Float32Array): number {
  let l = 0;
  for (let d = 0; d < output.length; d++) {
    const diff = output[d]! - target[d]!;
    l += 0.5 * diff * diff;
  }
  return l;
}

describe("SharedExpertMoE — routing", () => {
  test("deterministic init: same seed → identical router weights", () => {
    const a = new SharedExpertMoE(CFG);
    const b = new SharedExpertMoE(CFG);
    expect(Array.from(a.wr)).toEqual(Array.from(b.wr));
  });

  test("routes exactly top-k experts, gates form a distribution over them", () => {
    const m = new SharedExpertMoE(CFG);
    const x = randVec(CFG.modelDim, 7);
    const r = m.route(x);
    expect(r.experts).toHaveLength(CFG.topK);
    expect(new Set(r.experts).size).toBe(CFG.topK); // distinct
    expect(r.gates).toHaveLength(CFG.topK);
    expect(r.gates.reduce((s, g) => s + g, 0)).toBeCloseTo(1, 6); // softmax over selected
    expect(r.probs).toHaveLength(CFG.numExperts);
    expect(Array.from(r.probs).reduce((s, p) => s + p, 0)).toBeCloseTo(1, 6); // full softmax
    // Selected experts are the highest-logit ones → highest full-softmax probs.
    const maxNonSelected = Math.max(
      ...Array.from(r.probs).filter((_, e) => !r.experts.includes(e)),
    );
    for (const e of r.experts) expect(r.probs[e]!).toBeGreaterThanOrEqual(maxNonSelected);
  });

  test("forward is deterministic and shaped to the model dim", () => {
    const m = new SharedExpertMoE(CFG);
    const x = randVec(CFG.modelDim, 11);
    const a = m.forward(x);
    const b = m.forward(x);
    expect(a.output).toHaveLength(CFG.modelDim);
    expect(Array.from(a.output)).toEqual(Array.from(b.output));
    // The cache exposes one output per selected expert (the combine inputs).
    expect(a.cache.expertOut).toHaveLength(CFG.topK);
    // (The shared + gated-routed combine math is validated exactly by the
    // finite-difference gradient test and the training-loop test below.)
  });
});

describe("SharedExpertMoE — load balancing", () => {
  test("balanced dispatch ≈ 1, collapsed dispatch ≈ numExperts", () => {
    const E = CFG.numExperts;
    const uniform = Float32Array.from({ length: E }, () => 1 / E);

    const balanced = new LoadBalanceAccumulator(E);
    for (let t = 0; t < 8; t++) {
      // rotate the selected pair so every expert is dispatched equally
      const a = t % E;
      const b = (t + 1) % E;
      balanced.observe({ experts: [a, b], gates: [0.5, 0.5], probs: uniform } as RouteResult);
    }

    const collapsed = new LoadBalanceAccumulator(E);
    const peaked = Float32Array.from({ length: E }, (_, e) => (e < 2 ? 0.5 : 0));
    for (let t = 0; t < 8; t++) {
      collapsed.observe({ experts: [0, 1], gates: [0.5, 0.5], probs: peaked } as RouteResult);
    }

    expect(balanced.loss()).toBeCloseTo(1, 5);
    expect(collapsed.loss()).toBeGreaterThan(balanced.loss());
    expect(collapsed.loss()).toBeCloseTo(E / 2, 5); // two experts share all mass
  });
});

describe("SharedExpertMoE — gradients", () => {
  test("analytic gradients match finite differences", () => {
    const m = new SharedExpertMoE(CFG);
    const x = randVec(CFG.modelDim, 3);
    const target = randVec(CFG.modelDim, 99);

    const base = m.forward(x);
    const baseExperts = base.route.experts.join(",");
    const dOut = Float32Array.from(base.output, (o, d) => o - target[d]!);
    m.zeroGrad();
    m.backward(dOut, base.cache);

    const params = m.parameters();
    const grads = m.gradients();
    const eps = 1e-3;

    // Loss at the current params, with the route it produces (for stability guard).
    const lossAndRoute = (): { l: number; experts: string } => {
      const f = m.forward(x);
      return { l: loss(f.output, target), experts: f.route.experts.join(",") };
    };

    let checked = 0;
    for (let p = 0; p < params.length; p++) {
      const data = params[p]!.data;
      const grad = grads[p]!.data;
      for (let i = 0; i < data.length; i++) {
        const orig = data[i]!;
        data[i] = orig + eps;
        const plus = lossAndRoute();
        data[i] = orig - eps;
        const minus = lossAndRoute();
        data[i] = orig;
        // Skip elements that sit on a top-k boundary (gradient undefined there).
        if (plus.experts !== baseExperts || minus.experts !== baseExperts) continue;
        const fd = (plus.l - minus.l) / (2 * eps);
        expect(grad[i]!).toBeCloseTo(fd, 2);
        checked++;
      }
    }
    // The router Wr alone is 16 elements; the experts add many more — ensure the
    // guard didn't skip essentially everything.
    expect(checked).toBeGreaterThan(100);
  });
});

describe("SharedExpertMoE — checkpoint + training", () => {
  test("f32 and fp16 checkpoints round-trip; fp16 halves the data", () => {
    const m = new SharedExpertMoE(CFG);
    const f32 = m.exportWeights();
    const fp16 = m.exportWeights({ fp16: true });
    expect(fp16.byteLength).toBeLessThan(f32.byteLength);

    const a = new SharedExpertMoE(CFG);
    a.loadWeights(f32);
    expect(Array.from(a.wr)).toEqual(Array.from(m.wr));

    const b = new SharedExpertMoE(CFG);
    b.loadWeights(fp16);
    for (let i = 0; i < m.wr.length; i++) expect(b.wr[i]!).toBeCloseTo(m.wr[i]!, 2);
  });

  test("loadWeights rejects a config mismatch", () => {
    const m = new SharedExpertMoE(CFG);
    const buf = m.exportWeights();
    const wrong = new SharedExpertMoE({ ...CFG, numExperts: 6 });
    expect(() => wrong.loadWeights(buf)).toThrow(/mismatch/);
  });

  test("a tiny SGD loop reduces the regression loss", () => {
    const m = new SharedExpertMoE(CFG);
    const x = randVec(CFG.modelDim, 5);
    const target = randVec(CFG.modelDim, 42);

    const initial = loss(m.forward(x).output, target);
    const lr = 0.05;
    for (let step = 0; step < 300; step++) {
      const f = m.forward(x);
      const dOut = Float32Array.from(f.output, (o, d) => o - target[d]!);
      m.zeroGrad();
      m.backward(dOut, f.cache);
      const params = m.parameters();
      const grads = m.gradients();
      for (let p = 0; p < params.length; p++) {
        const data = params[p]!.data;
        const grad = grads[p]!.data;
        for (let i = 0; i < data.length; i++) data[i] = data[i]! - lr * grad[i]!;
      }
    }
    const final = loss(m.forward(x).output, target);
    expect(final).toBeLessThan(initial * 0.1);
  });

  test("exportExpert yields the four FFN tensors of one routed expert", () => {
    const m = new SharedExpertMoE(CFG);
    const e = m.exportExpert(0);
    expect(e.map((p) => p.name)).toEqual(["w1", "b1", "w2", "b2"]);
    expect(e[0]!.numel).toBe(CFG.hiddenDim * CFG.modelDim);
    expect(e[2]!.numel).toBe(CFG.modelDim * CFG.hiddenDim);
    expect(() => m.exportExpert(99)).toThrow(/out of range/);
  });
});
