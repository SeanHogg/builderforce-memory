/**
 * bench — the Evermind benchmarking harness.
 *
 * Measures a trained model on held-out data (perplexity, bits-per-token,
 * next-token accuracy, throughput), A/Bs two models, and offers a one-call
 * train-and-score path the Studio drives in the browser.
 */

export {
  benchmarkModel,
  benchmarkModelAsync,
  benchmarkText,
  compareModels,
  compareReports,
  corpusToSequences,
  trainAndBenchmark,
} from "./harness.js";

export {
  argmax,
  topKIndices,
  perplexity,
  bitsPerToken,
  newAccumulator,
  scoreInto,
  LN2,
} from "./metrics.js";

export type {
  LogitsModel,
  AsyncLogitsModel,
  BenchmarkOptions,
  BenchmarkReport,
  ComparisonReport,
  TrainAndBenchmarkOptions,
  TrainAndBenchmarkResult,
} from "./types.js";
