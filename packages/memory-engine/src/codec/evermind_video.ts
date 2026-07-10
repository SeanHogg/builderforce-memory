/**
 * evermind_video.ts — thin bridge that makes the EXISTING EvermindLM generate
 * video. No new model, no new generator: the codec turns frames into tokens, the
 * unqualified `EvermindLM.generate` autoregresses over them, and the codec turns
 * the emitted tokens back into frames. Deliberately DRY — everything the text
 * path already has (sampling, checkpoints, delta export, training) is reused.
 */

import type { EvermindLM, LMGenerateOptions } from "../lm/evermind_lm.js";
import type { VideoRVQCodec, Video, Frame } from "./video_rvq.js";
import type { ImageRVQCodec } from "./image_rvq.js";

/**
 * Build a unified training sequence `text… <vid> frames… </vid>` for
 * `EvermindLMTrainer.fit`. `textTokens` are ids in the codec's text region (from
 * a BPE tokenizer whose vocab size equals `codec.vocab.textVocabSize`). Training
 * the LM on many such sequences is what teaches text→video.
 */
export function buildVideoSequence(codec: VideoRVQCodec, textTokens: number[], video: Video): number[] {
  return [...textTokens, ...codec.encode(video)];
}

/**
 * Generate video from a prompt of already-tokenised context (text ids and/or a
 * partial video stream). The prompt should normally end just before or at the
 * `<vid>` marker; generation stops at `</vid>` or after `maxNewTokens`.
 *
 * Returns the decoded clip plus the raw produced token ids (useful for chaining
 * or continued generation).
 */
export function generateVideo(
  lm: EvermindLM,
  codec: VideoRVQCodec,
  promptTokens: number[],
  opts: LMGenerateOptions,
): { video: Video; tokens: number[] } {
  if (lm.config.vocabSize !== codec.vocabSize) {
    throw new Error(
      `generateVideo: EvermindLM vocabSize (${lm.config.vocabSize}) must equal codec.vocabSize (${codec.vocabSize})`,
    );
  }
  const tokens = lm.generate(promptTokens, { ...opts, stopToken: opts.stopToken ?? codec.vocab.eosVideo });
  return { video: codec.decode(tokens), tokens };
}

/** Generate a single image — the still-image case of {@link generateVideo}. */
export function generateImage(
  lm: EvermindLM,
  codec: ImageRVQCodec,
  promptTokens: number[],
  opts: LMGenerateOptions,
): { image: Frame; tokens: number[] } {
  const { video, tokens } = generateVideo(lm, codec.video, promptTokens, opts);
  return { image: video[0] ?? new Float32Array(codec.frameSize), tokens };
}
