/**
 * import/foreign/vocab.ts — fit a ported checkpoint to a different vocabulary size.
 *
 * A ported model keeps its source tokenizer's vocabulary. To pair it with another
 * tokenizer the embedding table (tied to the LM head) must have that tokenizer's row
 * count: surplus rows are dropped, missing rows are drawn small and random so the
 * new tokens start near the embedding cloud's centre. The new rows are recorded on
 * the port as synthesised, never passed off as ported.
 */

import { SeededRng, gaussianArray } from "../../utils/rng.js";
import type { PortedCheckpoint } from "./registry.js";

export interface ResizeVocabOptions {
  /** Standard deviation of new rows. Default 0.02, the usual embedding init. */
  std?: number;
  /** Seed for the new rows, so the same resize produces the same checkpoint. Default 1. */
  seed?: number;
}

/** `ported` with an embedding of `vocabSize` rows and a model config to match. */
export function resizePortedVocab(ported: PortedCheckpoint, vocabSize: number, opts: ResizeVocabOptions = {}): PortedCheckpoint {
  if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
    throw new Error(`import/foreign: vocabulary size must be a positive integer, got ${vocabSize}`);
  }
  const from = ported.modelConfig.vocabSize;
  if (vocabSize === from) return ported;

  const emb = ported.weights.get("embedding");
  if (!emb) throw new Error('import/foreign: the port has no "embedding" to resize');
  const dModel = ported.modelConfig.dModel;

  const resized = new Float32Array(vocabSize * dModel);
  resized.set(emb.subarray(0, Math.min(from, vocabSize) * dModel));
  const synthesisedTargets = [...ported.synthesisedTargets];
  if (vocabSize > from) {
    const std = opts.std ?? 0.02;
    const rng = new SeededRng(opts.seed ?? 1);
    resized.set(gaussianArray((vocabSize - from) * dModel, std, () => rng.next()), from * dModel);
    synthesisedTargets.push({
      target: "embedding",
      value: std,
      reason: `rows ${from}..${vocabSize - 1} drawn from N(0, ${std}²) to reach a ${vocabSize}-token vocabulary`,
    });
  }

  const weights = new Map(ported.weights);
  weights.set("embedding", resized);
  return { ...ported, weights, synthesisedTargets, modelConfig: { ...ported.modelConfig, vocabSize } };
}
