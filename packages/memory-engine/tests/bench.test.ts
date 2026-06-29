/**
 * tests/bench.test.ts — the Evermind benchmarking harness.
 */

import {
  benchmarkModel,
  benchmarkModelAsync,
  benchmarkText,
  compareReports,
  compareModels,
  corpusToSequences,
  trainAndBenchmark,
  topKIndices,
  argmax,
  perplexity,
  bitsPerToken,
} from "../src/bench/index";
import type { LogitsModel } from "../src/bench/index";
import { EvermindLM } from "../src/lm/evermind_lm";
import { BPETokenizer } from "../src/tokenizer/bpe";

const CORPUS =
  "BuilderForce orchestrates many agents through a planning loop. " +
  "The memory layer stores facts as embeddings. " +
  "Deployment runs on Cloudflare Workers and Durable Objects. " +
  "Tools are gated by a capability registry. " +
  "Agents recall facts and act on them. " +
  "The planning loop retrieves context before generating.";

/** A model that returns fixed logits per position regardless of input. */
function constModel(logits: Float32Array): LogitsModel {
  return { forward: (tokens) => ({ logits: tokens.map(() => logits) }) };
}

// ── metrics ────────────────────────────────────────────────────────────────────

describe("metrics", () => {
  test("argmax / topKIndices", () => {
    const lg = new Float32Array([0.1, 5.0, 0.2, 3.0]);
    expect(argmax(lg)).toBe(1);
    expect(topKIndices(lg, 2)).toEqual([1, 3]);
    // k larger than vocab clamps to vocab size.
    expect(topKIndices(lg, 99)).toHaveLength(4);
  });

  test("perplexity and bits-per-token are monotonic in cross-entropy", () => {
    expect(perplexity(0)).toBeCloseTo(1, 6);
    expect(perplexity(1)).toBeCloseTo(Math.E, 6);
    expect(bitsPerToken(Math.LN2)).toBeCloseTo(1, 6);
  });
});

// ── benchmarkModel ───────────────────────────────────────────────────────────

describe("benchmarkModel", () => {
  test("a model that always predicts the right next token scores perfectly", () => {
    // Vocab of 3; logits put all mass on token 2, and every sequence's next
    // token is 2 → top-1 accuracy 1.0 and low perplexity.
    const logits = new Float32Array([-10, -10, 20]);
    const seqs = [
      [0, 2],
      [1, 2],
      [0, 2],
    ];
    const r = benchmarkModel(constModel(logits), seqs, { now: () => 0 });
    expect(r.sequences).toBe(3);
    expect(r.tokens).toBe(3);
    expect(r.top1Accuracy).toBe(1);
    expect(r.topKAccuracy).toBe(1);
    expect(r.perplexity).toBeGreaterThanOrEqual(1);
    expect(r.perplexity).toBeLessThan(1.001);
  });

  test("skips sequences shorter than 2 tokens and reports zeros for empty input", () => {
    const r = benchmarkModel(constModel(new Float32Array([0, 0])), [[1], []], { now: () => 0 });
    expect(r.sequences).toBe(0);
    expect(r.tokens).toBe(0);
    expect(r.top1Accuracy).toBe(0);
  });

  test("uniform logits give perplexity ≈ vocab size", () => {
    const V = 4;
    const r = benchmarkModel(constModel(new Float32Array(V).fill(0)), [[0, 1, 2, 3]], { now: () => 0 });
    expect(r.perplexity).toBeCloseTo(V, 4);
    expect(r.bitsPerToken).toBeCloseTo(Math.log2(V), 4);
  });

  test("throughput is reported when latency is measured", () => {
    let t = 0;
    const r = benchmarkModel(constModel(new Float32Array([0, 0])), [[0, 1, 0, 1]], {
      measureLatency: true,
      now: () => (t += 5), // each now() advances 5ms
    });
    expect(r.elapsedMs).toBeGreaterThan(0);
    expect(r.tokensPerSecond).toBeGreaterThan(0);
  });
});

// ── async ────────────────────────────────────────────────────────────────────

test("benchmarkModelAsync matches the sync result", async () => {
  const logits = new Float32Array([-10, -10, 20]);
  const seqs = [[0, 2], [1, 2]];
  const sync = benchmarkModel(constModel(logits), seqs, { now: () => 0 });
  const asyncModel = { forward: async (t: number[]) => ({ logits: t.map(() => logits) }) };
  const out = await benchmarkModelAsync(asyncModel, seqs, { now: () => 0 });
  expect(out.top1Accuracy).toBe(sync.top1Accuracy);
  expect(out.perplexity).toBeCloseTo(sync.perplexity, 6);
});

// ── compare ──────────────────────────────────────────────────────────────────

describe("compareModels / compareReports", () => {
  test("the model with lower perplexity wins", () => {
    const seqs = [[0, 2], [1, 2]];
    const good = constModel(new Float32Array([-10, -10, 20])); // predicts 2
    const bad = constModel(new Float32Array([20, 20, -10])); // never predicts 2
    const cmp = compareModels(good, bad, seqs, { now: () => 0 });
    expect(cmp.winner).toBe("candidate");
    expect(cmp.perplexityDelta).toBeLessThan(0);
    expect(cmp.perplexityRatio).toBeLessThan(1);
    expect(cmp.summary).toContain("Candidate wins");
  });

  test("identical reports tie", () => {
    const seqs = [[0, 2], [1, 2]];
    const m = constModel(new Float32Array([-10, -10, 20]));
    const a = benchmarkModel(m, seqs, { now: () => 0 });
    const b = benchmarkModel(m, seqs, { now: () => 0 });
    expect(compareReports(a, b).winner).toBe("tie");
  });
});

// ── text + real model ──────────────────────────────────────────────────────────

describe("benchmarkText + trainAndBenchmark (EvermindLM)", () => {
  test("corpusToSequences splits sentences and drops 1-token fragments", () => {
    const tok = new BPETokenizer();
    tok.train(CORPUS, { numMerges: 60 });
    const seqs = corpusToSequences(CORPUS, tok);
    expect(seqs.length).toBeGreaterThanOrEqual(4);
    expect(seqs.every((s) => s.length >= 2)).toBe(true);
  });

  test("benchmarkText scores a trained model on raw text", () => {
    const tok = new BPETokenizer();
    tok.train(CORPUS, { numMerges: 60 });
    const model = new EvermindLM({ vocabSize: tok.vocabSize, dModel: 16, numLayers: 2, hiddenDim: 24, seed: 7 });
    const r = benchmarkText(model, tok, CORPUS, { now: () => 0 });
    expect(r.tokens).toBeGreaterThan(0);
    expect(Number.isFinite(r.perplexity)).toBe(true);
    expect(r.perplexity).toBeGreaterThan(1);
  });

  test("trainAndBenchmark holds out an eval split, learns, and scores it", () => {
    const r = trainAndBenchmark(CORPUS, { epochs: 40, numMerges: 80, seed: 7 });
    // Held-out split is non-empty and disjoint in size from train.
    expect(r.evalSequences).toBeGreaterThanOrEqual(1);
    expect(r.trainSequences).toBeGreaterThanOrEqual(1);
    expect(r.evalSequences + r.trainSequences).toBeGreaterThanOrEqual(4);
    // Training reduced loss (the model learned something).
    expect(r.finalTrainLoss).toBeLessThan(r.initialTrainLoss);
    // Produces a real scorecard + a non-empty sample.
    expect(Number.isFinite(r.perplexity)).toBe(true);
    expect(r.perplexity).toBeGreaterThan(1);
    expect(typeof r.sample).toBe("string");
    expect(r.vocabSize).toBeGreaterThan(0);
  });

  test("trainAndBenchmark is deterministic for a fixed seed", () => {
    const a = trainAndBenchmark(CORPUS, { epochs: 20, numMerges: 80, seed: 11 });
    const b = trainAndBenchmark(CORPUS, { epochs: 20, numMerges: 80, seed: 11 });
    expect(b.perplexity).toBeCloseTo(a.perplexity, 6);
    expect(b.top1Accuracy).toBe(a.top1Accuracy);
    expect(b.sample).toBe(a.sample);
  });

  test("rejects a corpus too small to hold out an eval split", () => {
    expect(() => trainAndBenchmark("Only one sentence here.", {})).toThrow(/at least 2/);
  });
});
