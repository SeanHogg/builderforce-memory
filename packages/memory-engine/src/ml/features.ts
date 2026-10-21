/**
 * ml/features.ts — how an input becomes a feature vector.
 *
 * Text goes through the existing CPU embedder seam ({@link TextEmbedder}, which
 * `EvermindTextEmbedder` implements: the person's own Evermind turns text into
 * an L2-normalised vector with no GPU and no network). Numeric fields are
 * appended as the task supplies them — on a FIXED scale (see {@link logScale}),
 * never one fitted to local data, so a federated weight means the same thing on
 * every machine.
 */
import type { TextEmbedder } from "../lm/text_embedder.js";
import type { FeatureSpec } from "./types.js";

export type { TextEmbedder } from "../lm/text_embedder.js";

/** Feature vector of one input: `[embed(text) …, numeric …]`. */
export async function extractFeatures<I>(spec: FeatureSpec<I>, input: I, embedder?: TextEmbedder): Promise<Float32Array> {
  const parts: ArrayLike<number>[] = [];
  if (spec.text) {
    if (!embedder) throw new Error("extractFeatures: this task embeds text but no embedder was supplied");
    parts.push(await embedder.embed(spec.text(input)));
  }
  if (spec.numeric) {
    const nums = spec.numeric(input);
    for (const v of nums) if (!Number.isFinite(v)) throw new Error(`extractFeatures: non-finite numeric feature ${v}`);
    parts.push(nums);
  }
  if (parts.length === 0) throw new Error("extractFeatures: the feature spec has neither text nor numeric fields");
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

/** Feature vectors of many inputs, in order. */
export async function extractFeatureMatrix<I>(
  spec: FeatureSpec<I>,
  inputs: readonly I[],
  embedder?: TextEmbedder,
): Promise<Float32Array[]> {
  const out: Float32Array[] = [];
  for (const i of inputs) out.push(await extractFeatures(spec, i, embedder));
  return out;
}

/**
 * A count or size on a fixed log scale: log1p(max(v, 0)) / log1p(ref). `ref` is
 * the value that maps to 1 (e.g. 100k tokens), so the scale is the same everywhere.
 */
export function logScale(v: number, ref: number): number {
  return Math.log1p(Math.max(0, v)) / Math.log1p(Math.max(ref, 1));
}

/** A ratio clamped to [0, max] — fill fractions, success rates. */
export function clampRatio(numerator: number, denominator: number, max = 4): number {
  if (!(denominator > 0)) return 0;
  return Math.max(0, Math.min(max, numerator / denominator));
}
