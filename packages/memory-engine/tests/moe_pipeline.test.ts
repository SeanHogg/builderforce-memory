/**
 * tests/moe_pipeline.test.ts
 * The train → publish pipeline: MoETrainer (AdamW + load-balance aux) and
 * EvermindModelPackage (the portable .evermind artifact).
 */

import { SharedExpertMoE } from "../src/moe/moe_model.js";
import { MoETrainer, type MoESample } from "../src/moe/moe_trainer.js";
import { EvermindModelPackage } from "../src/moe/moe_package.js";

const CFG = { modelDim: 4, hiddenDim: 6, numExperts: 4, topK: 2, seed: 2024 };

function randVec(n: number, seed: number): Float32Array {
  let s = seed >>> 0;
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s / 0xffffffff) * 2 - 1;
  }
  return a;
}

function dataset(n: number): MoESample[] {
  return Array.from({ length: n }, (_, i) => ({
    input: randVec(CFG.modelDim, 100 + i),
    target: randVec(CFG.modelDim, 900 + i),
  }));
}

describe("MoETrainer", () => {
  test("fit reduces the task loss and keeps the aux loss bounded", () => {
    const model = new SharedExpertMoE(CFG);
    const data = dataset(8);
    const trainer = new MoETrainer(model, { lr: 0.02, epochs: 60, auxWeight: 0.01 });
    const history = trainer.fit(data);

    expect(history).toHaveLength(60);
    expect(history.at(-1)!.loss).toBeLessThan(history[0]!.loss * 0.5);
    for (const h of history) {
      expect(h.auxLoss).toBeGreaterThan(0);
      expect(h.auxLoss).toBeLessThanOrEqual(CFG.numExperts + 1e-6); // ≤ E always
    }
  });

  test("training is deterministic for a fixed seed + data", () => {
    const data = dataset(6);
    const run = () => {
      const m = new SharedExpertMoE(CFG);
      new MoETrainer(m, { lr: 0.02, epochs: 10 }).fit(data);
      return Array.from(m.wr);
    };
    expect(run()).toEqual(run());
  });

  test("aux gradient pushes load away from over-used experts", () => {
    const m = new SharedExpertMoE(CFG);
    m.zeroGrad();
    const x = Float32Array.from({ length: CFG.modelDim }, () => 1); // all-positive
    const uniform = Float32Array.from({ length: CFG.numExperts }, () => 1 / CFG.numExperts);
    const f = Float32Array.from([0.7, 0.1, 0.1, 0.1]); // expert 0 over-dispatched
    m.auxGradStep(x, uniform, f, 1);

    const g = m.gradients()[0]!.data; // gWr, numExperts × modelDim
    // Over-used expert 0 → positive grad (SGD lowers its logit); under-used → negative.
    expect(g[0]!).toBeGreaterThan(0);
    expect(g[1 * CFG.modelDim]!).toBeLessThan(0);
  });
});

describe("EvermindModelPackage", () => {
  function trained(): SharedExpertMoE {
    const m = new SharedExpertMoE(CFG);
    new MoETrainer(m, { lr: 0.02, epochs: 20 }).fit(dataset(6));
    return m;
  }

  test("fromModel → toBlob → fromBlob round-trips manifest + checkpoint", () => {
    const m = trained();
    const pkg = EvermindModelPackage.fromModel(m, {
      name: "my-evermind",
      version: "1.0.0",
      card: { description: "A test model", license: "MIT", tags: ["demo"] },
      createdAt: "2026-06-27T00:00:00Z",
    });
    expect(pkg.manifest.name).toBe("my-evermind");
    expect(pkg.manifest.modelType).toBe("shared-expert-moe");
    expect(pkg.manifest.paramCount).toBeGreaterThan(0);
    expect(pkg.manifest.config.numExperts).toBe(CFG.numExperts);

    const back = EvermindModelPackage.fromBlob(pkg.toBlob());
    expect(back.manifest).toEqual(pkg.manifest);
    expect(back.checkpoint.byteLength).toBe(pkg.checkpoint.byteLength);
    expect(back.validate().ok).toBe(true);
  });

  test("a buyer's loadModel reproduces the creator's forward exactly", () => {
    const m = trained();
    const x = randVec(CFG.modelDim, 7);
    const before = m.forward(x).output;

    const blob = EvermindModelPackage.fromModel(m, {
      name: "x",
      version: "1",
      card: { description: "d" },
    }).toBlob();
    const loaded = EvermindModelPackage.fromBlob(blob).loadModel();
    const after = loaded.forward(x).output;

    for (let d = 0; d < CFG.modelDim; d++) expect(after[d]!).toBeCloseTo(before[d]!, 6);
  });

  test("validate catches a tampered checkpoint and fromBlob rejects bad magic", () => {
    const pkg = EvermindModelPackage.fromModel(new SharedExpertMoE(CFG), {
      name: "x",
      version: "1",
      card: { description: "d" },
    });
    // Tamper with the checkpoint bytes → checksum mismatch.
    const bad = new EvermindModelPackage(pkg.manifest, (() => {
      const buf = pkg.checkpoint.slice(0);
      new Uint8Array(buf)[40] ^= 0xff;
      return buf;
    })());
    const v = bad.validate();
    expect(v.ok).toBe(false);
    expect(v.errors.join(" ")).toMatch(/checksum/);
    expect(() => bad.loadModel()).toThrow(/checksum/);

    expect(() => EvermindModelPackage.fromBlob(new ArrayBuffer(8))).toThrow();
  });
});
