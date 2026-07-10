/**
 * video_rvq.ts — VideoRVQCodec: the pixels ⇄ tokens bottleneck that lets Evermind
 * generate video without changing the generator.
 *
 * Evermind is a discrete-token autoregressive SSM. To make it a video model you
 * do NOT touch the model — you give it a codec that maps a clip to a stream of
 * discrete tokens and back. This is the visual analogue of the BPE tokenizer:
 * where BPE turns text ⇄ ids, VideoRVQCodec turns frames ⇄ ids.
 *
 * Design — TEMPORAL residual vector quantization (the "video-from-the-start"
 * choice):
 *
 *   • Each frame is cut into non-overlapping p×p patches; each patch is a vector
 *     of length p·p·C (the "latent").
 *   • KEYFRAMES (frame 0 and every `keyframeInterval`-th frame) are quantized
 *     directly against the INTRA codebook bank.
 *   • Every other frame is INTER: we quantize the *residual against the previous
 *     RECONSTRUCTED frame* against the INTER bank. Video is mostly temporally
 *     redundant, so these deltas are small and cheap — this is exactly why an
 *     SSM (linear-time in sequence length) is the right generator for the long
 *     token streams video produces.
 *   • Quantization at each bank is RESIDUAL VQ over `levels` codebooks: level 0
 *     picks the nearest entry, level 1 quantizes what level 0 missed, and so on.
 *     More levels ⇒ finer reconstruction ⇒ more tokens per patch.
 *
 * The encoder runs closed-loop (it references its own reconstruction, never the
 * ground-truth previous frame) so the decoder — which only ever has
 * reconstructions — stays exactly in sync.
 *
 * Codebooks start random (a cold codec is lossy, like any untrained neural
 * codec) and are learned by {@link VideoRVQCodec.fit} (greedy per-level k-means).
 * Reaching photoreal fidelity needs training on a real corpus with real compute
 * — that is the one genuine blocker on shipping generated video, not the wiring.
 *
 * Pure CPU, zero deps, deterministic under a seed — same conventions as the rest
 * of the engine. A WGSL/WebGPU acceleration is a future drop-in with these shapes.
 */

import { SeededRng } from "../utils/rng.js";
import { MultimodalVocab, VIDEO_BANK_INTRA, VIDEO_BANK_INTER } from "./multimodal_vocab.js";

/** A single frame: length `height·width·channels`, layout `((y·W)+x)·C + ch`, values in [0,1]. */
export type Frame = Float32Array;
/** A clip: T frames, all the same shape. */
export type Video = Frame[];

export interface VideoRVQConfig {
  height: number;
  width: number;
  /** Colour channels. Default 3 (RGB). */
  channels?: number;
  /** Square patch size; `height` and `width` must be divisible by it. Default 4. */
  patch?: number;
  /** Residual-VQ depth (codes per patch). Default 2. */
  levels?: number;
  /** Codebook entries per level per bank. Default 16. */
  codebookSize?: number;
  /** Emit a keyframe every N frames (frame 0 is always a keyframe). Default 12. */
  keyframeInterval?: number;
  /** Size of the text region this codec's tokens sit above; 0 ⇒ pure-video vocab. Default 0. */
  textVocabSize?: number;
  /** Deterministic seed for codebook init. */
  seed?: number;
}

const DEFAULTS = {
  channels: 3,
  patch: 4,
  levels: 2,
  codebookSize: 16,
  keyframeInterval: 12,
  textVocabSize: 0,
  seed: 0x56524651, // "VRFQ"
} as const;

export class VideoRVQCodec {
  readonly height: number;
  readonly width: number;
  readonly channels: number;
  readonly patch: number;
  readonly levels: number;
  readonly codebookSize: number;
  readonly keyframeInterval: number;

  /** Patch latent dimension = patch·patch·channels. */
  readonly latentDim: number;
  /** Patches per frame = (H/patch)·(W/patch). */
  readonly patchesPerFrame: number;
  /** Code tokens emitted per frame = patchesPerFrame·levels. */
  readonly tokensPerFrame: number;

  /** The unified text+video vocabulary; feed `.vocab.size` to EvermindLM. */
  readonly vocab: MultimodalVocab;

  /** Two banks × levels codebooks; each codebook is `codebookSize × latentDim` row-major. */
  private readonly banks: [Float32Array[], Float32Array[]];

  constructor(config: VideoRVQConfig) {
    const channels = config.channels ?? DEFAULTS.channels;
    const patch = config.patch ?? DEFAULTS.patch;
    if (config.height <= 0 || config.width <= 0) throw new Error("VideoRVQCodec: height/width must be > 0");
    if (config.height % patch !== 0 || config.width % patch !== 0) {
      throw new Error(`VideoRVQCodec: height/width must be divisible by patch (${patch})`);
    }
    this.height = config.height;
    this.width = config.width;
    this.channels = channels;
    this.patch = patch;
    this.levels = config.levels ?? DEFAULTS.levels;
    this.codebookSize = config.codebookSize ?? DEFAULTS.codebookSize;
    this.keyframeInterval = Math.max(1, config.keyframeInterval ?? DEFAULTS.keyframeInterval);
    this.latentDim = patch * patch * channels;
    this.patchesPerFrame = (this.height / patch) * (this.width / patch);
    this.tokensPerFrame = this.patchesPerFrame * this.levels;

    this.vocab = new MultimodalVocab({
      textVocabSize: config.textVocabSize ?? DEFAULTS.textVocabSize,
      levels: this.levels,
      codebookSize: this.codebookSize,
    });

    const rng = new SeededRng((config.seed ?? DEFAULTS.seed) >>> 0 || 1);
    const mkBank = (): Float32Array[] =>
      Array.from({ length: this.levels }, () => {
        const cb = new Float32Array(this.codebookSize * this.latentDim);
        // Small-magnitude init: intra entries hover near mid-grey, inter (delta) near 0.
        for (let i = 0; i < cb.length; i++) cb[i] = (rng.next() - 0.5) * 0.1;
        return cb;
      });
    this.banks = [mkBank(), mkBank()];
    // Bias the intra bank toward the [0,1] pixel range so a cold codec is not black.
    for (const cb of this.banks[VIDEO_BANK_INTRA]) for (let i = 0; i < cb.length; i++) cb[i] += 0.5;
  }

  /** Total vocabulary size for this codec's model. */
  get vocabSize(): number {
    return this.vocab.size;
  }

  // ── Encode ───────────────────────────────────────────────────────────────────

  /** Encode a clip to a self-delimiting token stream: `<vid> (marker codes…)… </vid>`. */
  encode(video: Video): number[] {
    this.assertShape(video);
    const out: number[] = [this.vocab.bosVideo];
    let prevRecon: Float32Array[] | null = null;

    for (let t = 0; t < video.length; t++) {
      const patches = this.toPatches(video[t]!);
      const isKey = prevRecon === null || t % this.keyframeInterval === 0;
      const bankId = isKey ? VIDEO_BANK_INTRA : VIDEO_BANK_INTER;
      out.push(isKey ? this.vocab.frameKey : this.vocab.frameDelta);

      const reconPatches: Float32Array[] = [];
      for (let p = 0; p < this.patchesPerFrame; p++) {
        const target = isKey ? patches[p]! : sub(patches[p]!, prevRecon![p]!);
        const { codes, recon } = this.rvqEncode(bankId, target);
        for (let l = 0; l < this.levels; l++) out.push(this.vocab.codeToken(bankId, l, codes[l]!));
        reconPatches.push(isKey ? recon : add(prevRecon![p]!, recon));
      }
      prevRecon = reconPatches;
    }

    out.push(this.vocab.eosVideo);
    return out;
  }

  // ── Decode ───────────────────────────────────────────────────────────────────

  /**
   * Decode a token stream back to frames. Tolerant by design: it skips text /
   * stray tokens, treats a frame whose codes run short as zero-padded, and stops
   * at `</vid>` or end-of-stream — so a stream sampled from an under-trained
   * generator still yields a valid (if noisy) clip instead of throwing.
   */
  decode(tokens: number[]): Video {
    const frames: Video = [];
    let prevRecon: Float32Array[] | null = null;
    let i = 0;

    // Advance to the first frame marker (tolerate a leading prompt / <vid>).
    while (i < tokens.length && !this.vocab.isVideoMarker(tokens[i]!)) {
      if (tokens[i] === this.vocab.eosVideo) return frames;
      i++;
    }

    while (i < tokens.length) {
      const marker = tokens[i]!;
      if (!this.vocab.isVideoMarker(marker)) break; // </vid>, text, or garbage → end
      i++;
      const isKey = marker === this.vocab.frameKey || prevRecon === null;
      const bankId = isKey ? VIDEO_BANK_INTRA : VIDEO_BANK_INTER;

      // Pull up to tokensPerFrame code tokens; stop early on the next marker/EOS.
      const codes: number[] = [];
      while (codes.length < this.tokensPerFrame && i < tokens.length && this.vocab.isCode(tokens[i]!)) {
        codes.push(tokens[i]!);
        i++;
      }

      const reconPatches: Float32Array[] = [];
      for (let p = 0; p < this.patchesPerFrame; p++) {
        const recon = new Float32Array(this.latentDim);
        for (let l = 0; l < this.levels; l++) {
          const slot = p * this.levels + l;
          const tok = slot < codes.length ? codes[slot]! : -1;
          if (tok < 0) continue; // missing → contribute zero (graceful degradation)
          const { level, code } = this.vocab.decodeCode(tok);
          addRowInto(recon, this.banks[bankId]![level]!, code, this.latentDim);
        }
        reconPatches.push(isKey ? recon : add(prevRecon![p]!, recon));
      }

      frames.push(this.fromPatches(reconPatches));
      prevRecon = reconPatches;
    }

    return frames;
  }

  // ── Learn codebooks ────────────────────────────────────────────────────────────

  /**
   * Learn both codebook banks from a set of clips (greedy per-level k-means over
   * residuals — the standard way to train residual VQ). Returns the mean
   * reconstruction MSE over the training clips after fitting. This is the codec's
   * "training"; the generator is trained separately on the resulting token streams.
   */
  fit(videos: Video[], opts: { iterations?: number; seed?: number } = {}): number {
    const iterations = opts.iterations ?? 8;
    const rng = new SeededRng((opts.seed ?? 0x66697421) >>> 0 || 1); // "fit!"
    for (const v of videos) this.assertShape(v);

    // Gather intra patches and inter deltas (open-loop deltas are a fine proxy for fitting).
    const intra: Float32Array[] = [];
    const inter: Float32Array[] = [];
    for (const video of videos) {
      let prev: Float32Array[] | null = null;
      for (let t = 0; t < video.length; t++) {
        const patches = this.toPatches(video[t]!);
        const isKey = prev === null || t % this.keyframeInterval === 0;
        if (isKey) for (const p of patches) intra.push(p);
        else for (let p = 0; p < patches.length; p++) inter.push(sub(patches[p]!, prev![p]!));
        prev = patches;
      }
    }

    this.fitBank(VIDEO_BANK_INTRA, intra, iterations, rng);
    this.fitBank(VIDEO_BANK_INTER, inter, iterations, rng);

    // Report closed-loop reconstruction error over the corpus.
    let se = 0;
    let n = 0;
    for (const video of videos) {
      const recon = this.decode(this.encode(video));
      for (let t = 0; t < video.length; t++) {
        const a = video[t]!;
        const b = recon[t]!;
        for (let k = 0; k < a.length; k++) {
          const d = a[k]! - b[k]!;
          se += d * d;
        }
        n += a.length;
      }
    }
    return n > 0 ? se / n : 0;
  }

  /** Fit one bank's `levels` codebooks greedily over residuals. */
  private fitBank(bankId: number, vectors: Float32Array[], iterations: number, rng: SeededRng): void {
    if (vectors.length === 0) return;
    let residuals = vectors.map((v) => Float32Array.from(v));
    for (let l = 0; l < this.levels; l++) {
      const centroids = kmeans(residuals, this.codebookSize, this.latentDim, iterations, rng);
      this.banks[bankId]![l] = centroids;
      residuals = residuals.map((r) => {
        const idx = nearestRow(centroids, r, this.codebookSize, this.latentDim);
        const out = Float32Array.from(r);
        const off = idx * this.latentDim;
        for (let d = 0; d < this.latentDim; d++) out[d] = out[d]! - centroids[off + d]!;
        return out;
      });
    }
  }

  // ── Residual VQ over one bank ───────────────────────────────────────────────────

  private rvqEncode(bankId: number, vector: Float32Array): { codes: number[]; recon: Float32Array } {
    const residual = Float32Array.from(vector);
    const recon = new Float32Array(this.latentDim);
    const codes: number[] = [];
    for (let l = 0; l < this.levels; l++) {
      const cb = this.banks[bankId]![l]!;
      const idx = nearestRow(cb, residual, this.codebookSize, this.latentDim);
      codes.push(idx);
      const off = idx * this.latentDim;
      for (let d = 0; d < this.latentDim; d++) {
        recon[d] = recon[d]! + cb[off + d]!;
        residual[d] = residual[d]! - cb[off + d]!;
      }
    }
    return { codes, recon };
  }

  // ── Patch <-> frame ─────────────────────────────────────────────────────────────

  private toPatches(frame: Frame): Float32Array[] {
    const { patch, channels, width } = this;
    const pw = this.width / patch;
    const ph = this.height / patch;
    const out: Float32Array[] = [];
    for (let py = 0; py < ph; py++) {
      for (let px = 0; px < pw; px++) {
        const vec = new Float32Array(this.latentDim);
        for (let dy = 0; dy < patch; dy++) {
          for (let dx = 0; dx < patch; dx++) {
            const y = py * patch + dy;
            const x = px * patch + dx;
            const src = ((y * width) + x) * channels;
            const dst = ((dy * patch) + dx) * channels;
            for (let c = 0; c < channels; c++) vec[dst + c] = frame[src + c]!;
          }
        }
        out.push(vec);
      }
    }
    return out;
  }

  private fromPatches(patches: Float32Array[]): Frame {
    const { patch, channels, width } = this;
    const pw = this.width / patch;
    const frame = new Float32Array(this.height * this.width * channels);
    for (let p = 0; p < patches.length; p++) {
      const py = Math.floor(p / pw);
      const px = p - py * pw;
      const vec = patches[p]!;
      for (let dy = 0; dy < patch; dy++) {
        for (let dx = 0; dx < patch; dx++) {
          const y = py * patch + dy;
          const x = px * patch + dx;
          const dst = ((y * width) + x) * channels;
          const src = ((dy * patch) + dx) * channels;
          for (let c = 0; c < channels; c++) frame[dst + c] = clamp01(vec[src + c]!);
        }
      }
    }
    return frame;
  }

  private assertShape(video: Video): void {
    const expect = this.height * this.width * this.channels;
    for (let t = 0; t < video.length; t++) {
      if (video[t]!.length !== expect) {
        throw new Error(`VideoRVQCodec: frame ${t} has length ${video[t]!.length}, expected ${expect}`);
      }
    }
  }
}

// ── free helpers ─────────────────────────────────────────────────────────────────

function sub(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! - b[i]!;
  return out;
}

function add(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! + b[i]!;
  return out;
}

function addRowInto(acc: Float32Array, codebook: Float32Array, row: number, dim: number): void {
  const off = row * dim;
  for (let d = 0; d < dim; d++) acc[d] = acc[d]! + codebook[off + d]!;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Nearest codebook row to `v` by squared L2. `codebook` is `count × dim` row-major. */
function nearestRow(codebook: Float32Array, v: Float32Array, count: number, dim: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let r = 0; r < count; r++) {
    const off = r * dim;
    let d = 0;
    for (let k = 0; k < dim; k++) {
      const diff = v[k]! - codebook[off + k]!;
      d += diff * diff;
      if (d >= bestD) break;
    }
    if (d < bestD) {
      bestD = d;
      best = r;
    }
  }
  return best;
}

/** Lloyd's k-means → `count × dim` centroids. Empty clusters reseed to a random sample. */
function kmeans(vectors: Float32Array[], count: number, dim: number, iterations: number, rng: SeededRng): Float32Array {
  const centroids = new Float32Array(count * dim);
  // Init: distinct random samples.
  for (let r = 0; r < count; r++) {
    const src = vectors[Math.floor(rng.next() * vectors.length) % vectors.length]!;
    centroids.set(src, r * dim);
  }
  const sums = new Float32Array(count * dim);
  const counts = new Int32Array(count);
  for (let it = 0; it < iterations; it++) {
    sums.fill(0);
    counts.fill(0);
    for (const v of vectors) {
      const r = nearestRow(centroids, v, count, dim);
      counts[r] = counts[r]! + 1;
      const off = r * dim;
      for (let d = 0; d < dim; d++) sums[off + d] = sums[off + d]! + v[d]!;
    }
    for (let r = 0; r < count; r++) {
      const off = r * dim;
      if (counts[r]! === 0) {
        // Reseed the dead centroid onto a random sample so it can win points next round.
        const src = vectors[Math.floor(rng.next() * vectors.length) % vectors.length]!;
        centroids.set(src, off);
        continue;
      }
      const inv = 1 / counts[r]!;
      for (let d = 0; d < dim; d++) centroids[off + d] = sums[off + d]! * inv;
    }
  }
  return centroids;
}
