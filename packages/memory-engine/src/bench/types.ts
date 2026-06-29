/**
 * bench/types.ts — public types for the Evermind benchmarking harness.
 *
 * Benchmarking answers the question the marketplace (and the "beats a frozen
 * LLM" thesis) actually rests on: *how good is this model?* The harness measures
 * a trained model on held-out data with the standard language-model yardsticks —
 * perplexity, bits-per-token, and next-token accuracy — plus generation
 * throughput, and can A/B two models (e.g. a quantized vs full-precision build,
 * or a freshly-adapted vs prior checkpoint).
 *
 * The metrics operate on any object that can produce per-position logits from a
 * token sequence — the engine's `EvermindLM` satisfies {@link LogitsModel}
 * directly — so the harness is model-agnostic and dependency-free.
 */

/** The minimal surface a model must expose to be benchmarked: per-position logits. */
export interface LogitsModel {
  /** Run the model over a token sequence; returns one logit vector per position. */
  forward(tokens: number[]): { logits: Float32Array[] };
}

/** A model whose forward pass is asynchronous (e.g. a WebGPU backend). */
export interface AsyncLogitsModel {
  forward(tokens: number[]): Promise<{ logits: Float32Array[] }>;
}

/** Knobs for a benchmark run. */
export interface BenchmarkOptions {
  /** k for top-k next-token accuracy. Default 5. */
  topK?: number;
  /** Measure forward throughput (tokens/sec). Default true. */
  measureLatency?: boolean;
  /**
   * Monotonic clock in milliseconds, injectable for deterministic tests.
   * Defaults to `performance.now()` when available, else `Date.now()`.
   */
  now?: () => number;
}

/** The scorecard a benchmark run produces. */
export interface BenchmarkReport {
  /** Number of evaluated sequences. */
  sequences: number;
  /** Number of predicted positions (next-token targets) scored. */
  tokens: number;
  /** Mean next-token cross-entropy, in nats. Lower is better. */
  crossEntropy: number;
  /** Perplexity = exp(crossEntropy). Lower is better; 1.0 is perfect. */
  perplexity: number;
  /** Bits per token = crossEntropy / ln(2). Lower is better. */
  bitsPerToken: number;
  /** Fraction of positions where the argmax prediction was correct (0..1). */
  top1Accuracy: number;
  /** Fraction of positions where the true token was in the top-k (0..1). */
  topKAccuracy: number;
  /** The k used for {@link topKAccuracy}. */
  topK: number;
  /** Forward throughput in tokens/sec, when {@link BenchmarkOptions.measureLatency}. */
  tokensPerSecond?: number;
  /** Wall-clock spent in the forward passes (ms), when measured. */
  elapsedMs?: number;
}

/** The result of A/B-ing two models on the same eval set. */
export interface ComparisonReport {
  candidate: BenchmarkReport;
  baseline: BenchmarkReport;
  /** candidate.perplexity − baseline.perplexity (negative = candidate better). */
  perplexityDelta: number;
  /** candidate.perplexity / baseline.perplexity (<1 = candidate better). */
  perplexityRatio: number;
  /** candidate.top1Accuracy − baseline.top1Accuracy (positive = candidate better). */
  top1Delta: number;
  /** Which model won on perplexity (the primary metric). */
  winner: "candidate" | "baseline" | "tie";
  /** Human-readable one-liner. */
  summary: string;
}

/** Options for {@link trainAndBenchmark}: train a fresh EvermindLM, then score it. */
export interface TrainAndBenchmarkOptions {
  /** BPE merges to learn for the tokenizer. Default 100. */
  numMerges?: number;
  /** Model channel dimension. Default 32. */
  dModel?: number;
  /** Number of (conv + MoE) blocks. Default 2. */
  numLayers?: number;
  /** MoE expert FFN hidden width. Default 48. */
  hiddenDim?: number;
  /** Training epochs. Default 30. */
  epochs?: number;
  /** AdamW learning rate. Default 0.03. */
  lr?: number;
  /** Deterministic seed (model init + held-out split). Default 7. */
  seed?: number;
  /**
   * Fraction of sequences reserved for evaluation (never trained on). Default
   * 0.25. A real benchmark must score held-out data, so this is enforced > 0.
   */
  heldOutRatio?: number;
  /** k for top-k accuracy. Default 5. */
  topK?: number;
  /** Prompt used to capture a qualitative generation sample. Default "The". */
  prompt?: string;
}

/** {@link trainAndBenchmark} result: the held-out scorecard plus training context. */
export interface TrainAndBenchmarkResult extends BenchmarkReport {
  /** Sequences used for training. */
  trainSequences: number;
  /** Held-out sequences used for evaluation. */
  evalSequences: number;
  /** Final-epoch mean training loss (nats). */
  finalTrainLoss: number;
  /** First-epoch mean training loss (nats) — pairs with final to show the drop. */
  initialTrainLoss: number;
  /** Learned tokenizer vocabulary size. */
  vocabSize: number;
  /** A short greedy generation from {@link TrainAndBenchmarkOptions.prompt}. */
  sample: string;
}
