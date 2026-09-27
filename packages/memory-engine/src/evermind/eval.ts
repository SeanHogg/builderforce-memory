/**
 * evermind/eval.ts — the pre/post-merge regression check.
 *
 * Each merge produces a NEW version from the PREVIOUS one. To tell whether it helped
 * or quietly regressed (catastrophic forgetting), both models are scored on a small
 * held-out set of previously-learned examples and their mean next-token loss compared.
 *
 * Forward-only: the loss is the same mean cross-entropy the trainer reports, computed
 * from the logits alone, so scoring never pays for a backward pass or touches grads.
 */
import type { LogitsModel } from "../bench/types.js";
import { crossEntropyLoss } from "../training/autograd.js";

/** One held-out example: the learned text, optionally with the task prompt it answered. */
export interface EvalExample {
  prompt?: string;
  text: string;
}

/** Max tokens scored per example — bounds the forward cost of eval. */
export const EVAL_MAX_TOKENS = 128;

/** The tokenizer surface eval needs (the engine's `BPETokenizer` satisfies it). */
export interface EvalTokenizer {
  encode(text: string): number[];
}

/** Mean next-token cross-entropy over one token sequence, or null when it has no target. */
export function sequenceLoss(model: LogitsModel, ids: number[]): number | null {
  if (ids.length < 2) return null;
  const { logits } = model.forward(ids);
  const n = Math.min(logits.length, ids.length) - 1;
  if (n < 1) return null;
  let sum = 0;
  for (let t = 0; t < n; t++) sum += crossEntropyLoss(logits[t]!, ids[t + 1]!);
  const loss = sum / n;
  return Number.isFinite(loss) && loss >= 0 ? loss : null;
}

/**
 * Mean loss of `model` across `examples` (unscorable ones skipped), each example
 * weighted equally. Null when nothing was scorable, so a caller can decline to
 * record a meaningless eval point.
 */
export function meanEvalLoss(
  model: LogitsModel,
  tokenizer: EvalTokenizer,
  examples: readonly EvalExample[],
  maxTokens: number = EVAL_MAX_TOKENS,
): number | null {
  let sum = 0;
  let n = 0;
  for (const ex of examples) {
    const text = (ex.prompt ? `${ex.prompt}\n` : "") + ex.text;
    const loss = sequenceLoss(model, tokenizer.encode(text).slice(0, maxTokens));
    if (loss != null) {
      sum += loss;
      n++;
    }
  }
  return n > 0 ? sum / n : null;
}
