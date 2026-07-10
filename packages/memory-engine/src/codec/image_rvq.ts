/**
 * image_rvq.ts — ImageRVQCodec: still-image generation as the single-frame case
 * of {@link VideoRVQCodec}. An image has no temporal axis, so every frame is a
 * keyframe (intra) — this wrapper just fixes `keyframeInterval = 1` and speaks in
 * single frames instead of clips. Deliberately thin: it reuses the video codec's
 * patchify + residual-VQ + learned codebooks + tolerant decode wholesale, so
 * there is exactly one implementation to maintain.
 */

import { VideoRVQCodec, type VideoRVQConfig, type Frame } from "./video_rvq.js";

export type { Frame } from "./video_rvq.js";

export class ImageRVQCodec {
  /** The underlying video codec (pass to `generateVideo`/`generateImage`). */
  readonly video: VideoRVQCodec;

  constructor(config: Omit<VideoRVQConfig, "keyframeInterval">) {
    this.video = new VideoRVQCodec({ ...config, keyframeInterval: 1 });
  }

  get vocab() {
    return this.video.vocab;
  }
  get vocabSize(): number {
    return this.video.vocabSize;
  }
  get frameSize(): number {
    return this.video.height * this.video.width * this.video.channels;
  }

  /** Encode one image to a `<vid> <key> codes… </vid>` token stream. */
  encode(image: Frame): number[] {
    return this.video.encode([image]);
  }

  /** Decode a token stream to a single image (zeros if the stream yields no frame). */
  decode(tokens: number[]): Frame {
    const frames = this.video.decode(tokens);
    return frames[0] ?? new Float32Array(this.frameSize);
  }

  /** Learn codebooks from a set of images (each is a one-frame clip). Returns recon MSE. */
  fit(images: Frame[], opts?: { iterations?: number; seed?: number }): number {
    return this.video.fit(
      images.map((im) => [im]),
      opts,
    );
  }
}
