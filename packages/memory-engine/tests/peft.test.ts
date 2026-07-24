/**
 * tests/peft.test.ts — parameter-efficient / efficient-training toolkit:
 * LoRA + QLoRA, mixed-precision (dynamic loss scaling), activation checkpointing,
 * gradient accumulation, and optimizer-state sharding. These close the
 * llama-cookbook best-practice gaps on the CPU reference training path.
 */

import { EvermindLM, EvermindLMTrainer } from "../src/lm/evermind_lm.js";
import { LoRAAdapter, EvermindLMLoRA } from "../src/training/lora.js";
import { DynamicLossScaler, roundFp16 } from "../src/training/mixed_precision.js";
import { AdamW } from "../src/optim/adamw.js";
import { crossEntropyLoss } from "../src/training/autograd.js";

// topK === numExperts ⇒ every expert active ⇒ no discrete routing boundary, so
// finite-difference gradient checks are smooth.
const CFG = {
  vocabSize: 7,
  dModel: 8,
  numLayers: 2,
  convKernel: 3,
  hiddenDim: 12,
  numExperts: 2,
  topK: 2,
  seed: 123,
};

const SEQ = [1, 4, 2, 5, 3, 0];

function pureLoss(m: EvermindLM, tokens: number[]): number {
  const { logits } = m.forward(tokens);
  let L = 0;
  const inv = 1 / (tokens.length - 1);
  for (let t = 0; t < tokens.length - 1; t++) L += crossEntropyLoss(logits[t]!, tokens[t + 1]!) * inv;
  return L;
}

// ── Gap 1: LoRA ────────────────────────────────────────────────────────────────

describe("LoRA adapter", () => {
  test("ΔW starts at zero (B=0), so the adapted model equals the base", () => {
    const adapter = new LoRAAdapter(7, 8, { rank: 4 });
    const delta = adapter.delta();
    expect(delta.every((x) => x === 0)).toBe(true);
  });

  test("adapter gradient matches finite differences of the loss", () => {
    const model = new EvermindLM(CFG);
    const lora = new EvermindLMLoRA(model, { rank: 4, alpha: 4, seed: 9 });
    // B starts at 0 (⇒ gA would be 0); nudge it so both A and B gradients are live.
    for (let i = 0; i < lora.adapter.B.length; i++) lora.adapter.B[i] = 0.03 * Math.sin(i);

    // Analytic adapter gradient via the full-emb gradient projected onto (A,B).
    const saved = model.emb;
    model.emb = lora.mergedEmb();
    model.zeroGrad();
    model.lossAndBackward(SEQ);
    const gW = Float32Array.from(model.gradients()[0]!.data);
    model.emb = saved;
    lora.adapter.zeroGrad();
    lora.adapter.accumulateGradient(gW);

    const eps = 1e-3;
    const check = (buf: Float32Array, grad: Float32Array) => {
      const stride = Math.max(1, Math.floor(buf.length / 5));
      let checked = 0;
      for (let i = 0; i < buf.length; i += stride) {
        expect(grad[i]!).toBeCloseTo(fdAt(lora, buf, i, eps), 2);
        checked++;
      }
      expect(checked).toBeGreaterThan(2);
    };
    check(lora.adapter.A, lora.adapter.gradients()[1]!.data); // gradients()[1] = gA
    check(lora.adapter.B, lora.adapter.gradients()[0]!.data); // gradients()[0] = gB
  });

  test("training the adapter lowers loss while the frozen base is untouched", () => {
    const model = new EvermindLM(CFG);
    const baseBefore = Float32Array.from(model.emb);
    const lora = new EvermindLMLoRA(model, { rank: 4, seed: 3 });
    const before = pureLossMerged(lora, SEQ);
    const hist = lora.fit([SEQ], { epochs: 40, lr: 0.05 });
    const after = pureLossMerged(lora, SEQ);
    expect(after).toBeLessThan(before);
    expect(hist[hist.length - 1]!).toBeLessThan(hist[0]!);
    // Base embedding never moved.
    for (let i = 0; i < baseBefore.length; i++) expect(model.emb[i]!).toBeCloseTo(baseBefore[i]!, 10);
  });

  test("adapter serialises round-trip and is smaller than the full matrix at real vocab", () => {
    // Realistic vocab so the low-rank saving is visible: full = 100·8 = 800 params,
    // rank-4 adapter = 4·(100+8) = 432.
    const big = { ...CFG, vocabSize: 100 };
    const model = new EvermindLM(big);
    const lora = new EvermindLMLoRA(model, { rank: 4, seed: 3 });
    lora.fit([SEQ], { epochs: 5, lr: 0.05 });
    const blob = lora.serializeAdapter();
    const restored = EvermindLMLoRA.loadAdapter(new EvermindLM(big), blob);
    expect(pureLossMerged(restored, SEQ)).toBeCloseTo(pureLossMerged(lora, SEQ), 6);
    const fp = lora.footprint();
    expect(fp.trainableParams).toBeLessThan(fp.baseParams); // 432 < 800
    expect(fp.adapterBytes).toBeLessThan(fp.baseBytes);
  });
});

// finite-diff of the merged-model loss wrt a raw adapter buffer entry
function fdAt(lora: EvermindLMLoRA, buf: Float32Array, i: number, eps: number): number {
  const orig = buf[i]!;
  buf[i] = orig + eps;
  const plus = pureLossMerged(lora, SEQ);
  buf[i] = orig - eps;
  const minus = pureLossMerged(lora, SEQ);
  buf[i] = orig;
  return (plus - minus) / (2 * eps);
}

function pureLossMerged(lora: EvermindLMLoRA, seq: number[]): number {
  const model = lora.baseModel;
  const saved = model.emb;
  model.emb = lora.mergedEmb();
  const L = pureLoss(model, seq);
  model.emb = saved;
  return L;
}

// ── Gap 2: QLoRA ────────────────────────────────────────────────────────────────

describe("QLoRA (quantized frozen base + trainable adapter)", () => {
  test("int8 base costs ~1/4 of fp32 and the adapter still trains", () => {
    const model = new EvermindLM(CFG);
    const lora = new EvermindLMLoRA(model, { rank: 4, seed: 5, baseQuant: "int8" });
    const fp32Bytes = model.config.vocabSize * model.config.dModel * 4;
    const fp = lora.footprint();
    expect(fp.baseBytes).toBeLessThan(fp32Bytes / 3); // int8 ≈ 1 byte/param
    const before = pureLossMerged(lora, SEQ);
    lora.fit([SEQ], { epochs: 40, lr: 0.05 });
    const after = pureLossMerged(lora, SEQ);
    expect(after).toBeLessThan(before);
  });

  test("fp16 base halves the resident bytes", () => {
    const model = new EvermindLM(CFG);
    const lora = new EvermindLMLoRA(model, { rank: 4, baseQuant: "fp16" });
    const fp32Bytes = model.config.vocabSize * model.config.dModel * 4;
    expect(lora.footprint().baseBytes).toBe(fp32Bytes / 2);
  });
});

// ── Gap 3: Mixed precision ──────────────────────────────────────────────────────

describe("mixed-precision training", () => {
  test("loss scaling rescues a gradient that would underflow in fp16", () => {
    const tiny = 3e-8; // below the smallest fp16 subnormal (~5.96e-8)
    expect(roundFp16(tiny)).toBe(0); // underflows without scaling
    expect(roundFp16(tiny * 65536)).not.toBe(0); // survives once scaled up
  });

  test("dynamic scaler detects overflow, skips the step, and backs off", () => {
    const scaler = new DynamicLossScaler({ initScale: 1024, growthInterval: 2 });
    const bad = [{ data: new Float32Array([Infinity, 1]) }];
    expect(scaler.check(bad)).toBe(true);
    const proceed = scaler.update(true);
    expect(proceed).toBe(false);
    expect(scaler.scale).toBe(512); // halved
    expect(scaler.skipped).toBe(1);
  });

  test("scaler grows the scale after a clean streak", () => {
    const scaler = new DynamicLossScaler({ initScale: 8, growthInterval: 2 });
    expect(scaler.update(false)).toBe(true);
    expect(scaler.scale).toBe(8);
    expect(scaler.update(false)).toBe(true);
    expect(scaler.scale).toBe(16); // grew after 2 clean steps
  });

  test("EvermindLMTrainer trains under mixed precision", () => {
    const model = new EvermindLM(CFG);
    const trainer = new EvermindLMTrainer(model, { epochs: 30, lr: 0.05, mixedPrecision: { initScale: 1024 } });
    const hist = trainer.fit([SEQ]);
    expect(hist[hist.length - 1]!).toBeLessThan(hist[0]!);
    expect(trainer.lossScaler).not.toBeNull();
  });
});

// ── Gap 4: Activation checkpointing + gradient accumulation ──────────────────────

describe("activation checkpointing", () => {
  test("checkpointed backward yields identical gradients and loss", () => {
    const mFull = new EvermindLM(CFG);
    const mCkpt = new EvermindLM(CFG); // same seed ⇒ identical weights
    mFull.zeroGrad();
    mCkpt.zeroGrad();
    const lFull = mFull.lossAndBackward(SEQ);
    const lCkpt = mCkpt.lossAndBackwardCheckpointed(SEQ);
    expect(lCkpt).toBeCloseTo(lFull, 10);
    const gA = mFull.gradients();
    const gB = mCkpt.gradients();
    for (let p = 0; p < gA.length; p++) {
      for (let i = 0; i < gA[p]!.data.length; i++) {
        expect(gB[p]!.data[i]!).toBeCloseTo(gA[p]!.data[i]!, 9);
      }
    }
  });
});

describe("gradient accumulation", () => {
  test("accumulating k identical micro-batches equals one step on their mean", () => {
    const lr = 0.03;
    const mSingle = new EvermindLM(CFG);
    mSingle.zeroGrad();
    mSingle.lossAndBackward(SEQ);
    new AdamW(mSingle, { lr }).step();

    const mAccum = new EvermindLM(CFG);
    new EvermindLMTrainer(mAccum, { epochs: 1, lr, accumSteps: 3 }).fit([SEQ, SEQ, SEQ]);

    const a = mSingle.parameters();
    const b = mAccum.parameters();
    for (let p = 0; p < a.length; p++) {
      for (let i = 0; i < a[p]!.data.length; i++) {
        expect(b[p]!.data[i]!).toBeCloseTo(a[p]!.data[i]!, 6);
      }
    }
  });
});

// ── Gap 5: Optimizer-state sharding ─────────────────────────────────────────────

describe("optimizer-state sharding (ZeRO-1 analog)", () => {
  test("two shards over the same model equal one full step, at half the state each", () => {
    const mFull = new EvermindLM(CFG);
    mFull.zeroGrad();
    mFull.lossAndBackward(SEQ);
    const mShard = new EvermindLM(CFG); // identical init
    mShard.zeroGrad();
    mShard.lossAndBackward(SEQ); // deterministic ⇒ identical grads

    const full = new AdamW(mFull, { lr: 0.02 });
    full.step();

    const s0 = new AdamW(mShard, { lr: 0.02, shard: { index: 0, count: 2 } });
    const s1 = new AdamW(mShard, { lr: 0.02, shard: { index: 1, count: 2 } });
    s0.step();
    s1.step();

    const a = mFull.parameters();
    const b = mShard.parameters();
    for (let p = 0; p < a.length; p++) {
      for (let i = 0; i < a[p]!.data.length; i++) {
        expect(b[p]!.data[i]!).toBeCloseTo(a[p]!.data[i]!, 10);
      }
    }
    // Each shard holds only part of the optimizer state; together they equal the full state.
    expect(s0.stateBytes()).toBeLessThan(full.stateBytes());
    expect(s1.stateBytes()).toBeLessThan(full.stateBytes());
    expect(s0.stateBytes() + s1.stateBytes()).toBe(full.stateBytes());
  });
});
