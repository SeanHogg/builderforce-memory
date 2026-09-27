/**
 * evermind/adapt.ts — ONE adaptation recipe: fit a private copy of a model version
 * on some text, then diff it against that version.
 *
 * Every learner that contributes to a shared Evermind (an on-prem host pushing a
 * delta, or the single writer fitting queued text itself) must run the SAME recipe,
 * so a locally-produced delta is the same size of update the writer would have
 * produced from the same text. That recipe lives here and nowhere else.
 */
import type { EvermindModelPackage } from "./package.js";
import { EvermindLMTrainer } from "../lm/evermind_lm.js";
import { applyCheckpointDiff, diffCheckpoints } from "../utils/delta.js";
import { tokenWindows } from "../bench/adaptation.js";

/** Chars of text fed to one adaptation pass. */
export const EVERMIND_ADAPT_MAX_CHARS = 4000;
/** Token window length of the adaptation training sequences. */
export const EVERMIND_ADAPT_WINDOW_TOKENS = 64;

/** The tokenizer surface adaptation needs (the engine's `BPETokenizer` satisfies it). */
export interface AdaptTokenizer {
  encode(text: string): number[];
}

/** The outcome of one adaptation. */
export interface AdaptResult {
  /** Sparse serialized RowDelta of adapted vs base. */
  diff: ArrayBuffer;
  /** The trainer's final-epoch mean loss. */
  loss: number;
  /** How many training windows the text produced. */
  sequences: number;
}

export interface AdaptOptions {
  maxChars?: number;
  windowTokens?: number;
  epochs?: number;
}

/**
 * Adapt a freshly-loaded copy of `pkg` on `text` and return the diff, or null when
 * the text yields no trainable window. The package's own checkpoint is never
 * mutated, so the diff is exactly "this text's update", never an accumulation.
 * Throws when `pkg` is not an `evermind-lm` (as `loadLM` does).
 */
export function adaptAndDiff(
  pkg: EvermindModelPackage,
  tokenizer: AdaptTokenizer,
  text: string,
  opts: AdaptOptions = {},
): AdaptResult | null {
  const maxChars = opts.maxChars ?? EVERMIND_ADAPT_MAX_CHARS;
  const windowTokens = opts.windowTokens ?? EVERMIND_ADAPT_WINDOW_TOKENS;
  const seqs = tokenWindows(tokenizer.encode(text.slice(0, maxChars)), windowTokens);
  if (seqs.length === 0) return null;
  const lm = pkg.loadLM();
  const history = new EvermindLMTrainer(lm, { epochs: opts.epochs ?? 1 }).fit(seqs);
  const loss = history.length > 0 ? history[history.length - 1]! : 0;
  return { diff: diffCheckpoints(pkg.checkpoint, lm.exportWeights()), loss, sequences: seqs.length };
}

/** The outcome of {@link adaptPackage}: the adapted package itself. */
export interface AdaptedPackage {
  pkg: EvermindModelPackage;
  loss: number;
  sequences: number;
}

/**
 * The same recipe for a PRIVATE Evermind — a single local writer that keeps its own
 * update instead of shipping it. The update is the exact diff {@link adaptAndDiff}
 * would send, applied to the version it was fitted on, so a person's local model moves
 * by the same step a shared writer would have taken from the same text.
 */
export function adaptPackage(
  pkg: EvermindModelPackage,
  tokenizer: AdaptTokenizer,
  text: string,
  version: string,
  opts: AdaptOptions = {},
): AdaptedPackage | null {
  const r = adaptAndDiff(pkg, tokenizer, text, opts);
  if (!r) return null;
  return { pkg: pkg.withCheckpoint(applyCheckpointDiff(pkg.checkpoint, r.diff), version), loss: r.loss, sequences: r.sequences };
}
