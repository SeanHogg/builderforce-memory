/**
 * tests/evermind_lm.test.ts
 * EvermindLM: exact backward (finite differences), overfit→generate, training,
 * and checkpoint round-trip — the generative model behind the .evermind artifact.
 */

import { EvermindLM, EvermindLMTrainer, type TextCodec } from "../src/lm/evermind_lm.js";
import { EvermindModelPackage } from "../src/moe/moe_package.js";
import { BPETokenizer } from "../src/tokenizer/bpe.js";
import { crossEntropyLoss } from "../src/training/autograd.js";

const CFG = {
  vocabSize: 6,
  dModel: 8,
  numLayers: 2,
  convKernel: 3,
  hiddenDim: 12,
  numExperts: 4,
  topK: 2,
  seed: 77,
};

/** Pure next-token loss + a route signature (for finite-diff stability). */
function lossAndSig(m: EvermindLM, tokens: number[]): { loss: number; sig: string } {
  const { logits, cache } = m.forward(tokens);
  let loss = 0;
  const inv = 1 / (tokens.length - 1);
  for (let t = 0; t < tokens.length - 1; t++) loss += crossEntropyLoss(logits[t]!, tokens[t + 1]!) * inv;
  const parts: string[] = [];
  for (const lc of cache.layers) for (const mc of lc.moeCache) parts.push(mc.route.experts.join("."));
  return { loss, sig: parts.join("|") };
}

describe("EvermindLM — gradients", () => {
  test("analytic gradients match finite differences (incl. tied embeddings)", () => {
    const m = new EvermindLM(CFG);
    const tokens = [1, 3, 2, 4, 0];
    const base = lossAndSig(m, tokens);

    m.zeroGrad();
    const loss = m.lossAndBackward(tokens);
    expect(loss).toBeCloseTo(base.loss, 5);

    const params = m.parameters();
    const grads = m.gradients();
    const eps = 1e-3;
    let checked = 0;
    for (let p = 0; p < params.length; p++) {
      const data = params[p]!.data;
      const grad = grads[p]!.data;
      // Stride so the test stays fast but covers every parameter array.
      const stride = Math.max(1, Math.floor(data.length / 6));
      for (let i = 0; i < data.length; i += stride) {
        const orig = data[i]!;
        data[i] = orig + eps;
        const plus = lossAndSig(m, tokens);
        data[i] = orig - eps;
        const minus = lossAndSig(m, tokens);
        data[i] = orig;
        if (plus.sig !== base.sig || minus.sig !== base.sig) continue; // routing boundary
        const fd = (plus.loss - minus.loss) / (2 * eps);
        expect(grad[i]!).toBeCloseTo(fd, 2);
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(30);
  });
});

describe("EvermindLM — train + generate", () => {
  test("overfits a short sequence and generates it back (greedy)", () => {
    const m = new EvermindLM(CFG);
    const seq = [1, 2, 3, 4, 5];
    const trainer = new EvermindLMTrainer(m, { lr: 0.03, epochs: 500 });
    const history = trainer.fit([seq]);

    expect(history.at(-1)!).toBeLessThan(history[0]! * 0.2);
    // From the first token, greedy decoding should reproduce the rest.
    const out = m.generate([1], { maxNewTokens: 4, temperature: 0 });
    expect(out).toEqual([2, 3, 4, 5]);
  });

  test("training is deterministic for a fixed seed", () => {
    const run = () => {
      const m = new EvermindLM(CFG);
      new EvermindLMTrainer(m, { lr: 0.03, epochs: 20 }).fit([[1, 2, 3, 4, 5]]);
      return Array.from(m.emb);
    };
    expect(run()).toEqual(run());
  });
});

describe("EvermindLM — checkpoint", () => {
  test("f32/fp16 checkpoints round-trip and a reloaded model generates identically", () => {
    const m = new EvermindLM(CFG);
    new EvermindLMTrainer(m, { lr: 0.03, epochs: 100 }).fit([[1, 2, 3, 4, 5]]);
    const promptOut = m.generate([1], { maxNewTokens: 4, temperature: 0 });

    const f32 = m.exportWeights();
    const fp16 = m.exportWeights({ fp16: true });
    expect(fp16.byteLength).toBeLessThan(f32.byteLength);

    const reloaded = new EvermindLM(CFG);
    reloaded.loadWeights(f32);
    expect(reloaded.generate([1], { maxNewTokens: 4, temperature: 0 })).toEqual(promptOut);

    const wrong = new EvermindLM({ ...CFG, numLayers: 3 });
    expect(() => wrong.loadWeights(f32)).toThrow(/mismatch/);
    expect(() => reloaded.loadWeights(new ArrayBuffer(32))).toThrow(/magic/);
  });
});

describe("EvermindLM — publish → buy → run", () => {
  test("a trained LM packages as .evermind, and a buyer loads it and generates identically", () => {
    const m = new EvermindLM(CFG);
    new EvermindLMTrainer(m, { lr: 0.03, epochs: 200 }).fit([[1, 2, 3, 4, 5]]);
    const creatorOut = m.generate([1], { maxNewTokens: 4, temperature: 0 });

    // Creator publishes.
    const pkg = EvermindModelPackage.fromLM(m, {
      name: "tiny-storyteller",
      version: "1.0.0",
      card: { description: "Generates 2,3,4,5 from 1", license: "MIT", tags: ["demo"] },
      createdAt: "2026-06-27T00:00:00Z",
    });
    expect(pkg.manifest.modelType).toBe("evermind-lm");
    expect(pkg.manifest.paramCount).toBeGreaterThan(0);

    // Ship one blob; buyer reconstitutes, validates, and runs it.
    const blob = pkg.toBlob();
    const bought = EvermindModelPackage.fromBlob(blob);
    expect(bought.validate().ok).toBe(true);
    const buyerOut = bought.loadLM().generate([1], { maxNewTokens: 4, temperature: 0 });
    expect(buyerOut).toEqual(creatorOut);

    // Wrong loader is rejected.
    expect(() => bought.loadModel()).toThrow(/loadLM/);
  });
});

describe("EvermindLM — text I/O", () => {
  /** A tiny char-level codec so the model overfits text quickly. */
  const ALPHABET = "abcde";
  const codec: TextCodec = {
    encode: (text) => [...text].map((ch) => ALPHABET.indexOf(ch)).filter((i) => i >= 0),
    decode: (ids) => ids.map((i) => ALPHABET[i] ?? "").join(""),
  };

  test("generateText consumes and emits real text", () => {
    const m = new EvermindLM({ ...CFG, vocabSize: ALPHABET.length });
    new EvermindLMTrainer(m, { lr: 0.03, epochs: 500 }).fit([codec.encode("abcde")]);
    const out = m.generateText("a", codec, { maxNewTokens: 4, temperature: 0 });
    expect(out).toBe("bcde");
  });

  test("the engine BPETokenizer is usable as the LM's TextCodec", () => {
    // Structural contract (this assignment compiling is the integration point):
    // BPETokenizer.encode/decode match what generateText needs. A *loaded* vocab
    // (tok.load(...)) round-trips real text; here we assert the shape of the seam.
    const tok: TextCodec = new BPETokenizer();
    expect(Array.isArray(tok.encode("x"))).toBe(true);
    expect(typeof tok.decode([])).toBe("string");
  });
});
