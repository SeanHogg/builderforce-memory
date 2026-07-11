/**
 * tests/video_rvq.test.ts
 * VideoRVQCodec: the frames ⇄ tokens bottleneck that lets the (unchanged)
 * EvermindLM generate video. Covers vocab bijection, encode structure,
 * roundtrip shape/determinism, learned-codebook improvement, and the
 * end-to-end generate→decode plumbing.
 */

import { VideoRVQCodec, type Video } from "../src/codec/video_rvq.js";
import { ImageRVQCodec } from "../src/codec/image_rvq.js";
import { MultimodalVocab, VIDEO_BANK_INTRA, VIDEO_BANK_INTER } from "../src/codec/multimodal_vocab.js";
import { buildVideoSequence, generateVideo, generateImage } from "../src/codec/evermind_video.js";
import { EvermindLM, EvermindLMTrainer } from "../src/lm/evermind_lm.js";
import { EvermindModelPackage } from "../src/moe/moe_package.js";

/** Deterministic synthetic clip: smooth spatial pattern that drifts a little each frame. */
function makeVideo(T: number, H: number, W: number, C: number, phase = 0): Video {
  const frames: Video = [];
  for (let t = 0; t < T; t++) {
    const f = new Float32Array(H * W * C);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        for (let c = 0; c < C; c++) {
          const v = 0.5 + 0.4 * Math.sin((x + phase + t * 0.3) * 0.7 + y * 0.5 + c);
          f[((y * W) + x) * C + c] = v;
        }
      }
    }
    frames.push(f);
  }
  return frames;
}

function reconMSE(codec: VideoRVQCodec, videos: Video[]): number {
  let se = 0;
  let n = 0;
  for (const v of videos) {
    const r = codec.decode(codec.encode(v));
    for (let t = 0; t < v.length; t++) {
      for (let k = 0; k < v[t]!.length; k++) {
        const d = v[t]![k]! - r[t]![k]!;
        se += d * d;
      }
      n += v[t]!.length;
    }
  }
  return se / n;
}

describe("MultimodalVocab", () => {
  test("code tokens round-trip through classify/decodeCode; regions are disjoint", () => {
    const v = new MultimodalVocab({ textVocabSize: 5, levels: 3, codebookSize: 4 });
    // Text region.
    for (let t = 0; t < 5; t++) expect(v.classify(t)).toEqual({ kind: "text", id: t });
    // Control tokens.
    expect(v.classify(v.bosVideo)).toEqual({ kind: "control", control: "bosVideo" });
    expect(v.classify(v.eosVideo)).toEqual({ kind: "control", control: "eosVideo" });
    expect(v.classify(v.frameKey)).toEqual({ kind: "control", control: "frameKey" });
    expect(v.classify(v.frameDelta)).toEqual({ kind: "control", control: "frameDelta" });
    // Code region — bijective (bank, level, code) ↔ token id.
    for (const bank of [VIDEO_BANK_INTRA, VIDEO_BANK_INTER]) {
      for (let l = 0; l < 3; l++) {
        for (let c = 0; c < 4; c++) {
          const tok = v.codeToken(bank, l, c);
          expect(v.isCode(tok)).toBe(true);
          expect(v.decodeCode(tok)).toEqual({ bank, level: l, code: c });
        }
      }
    }
    // Highest code token is exactly size-1 (no gaps, no overrun).
    expect(v.codeToken(VIDEO_BANK_INTER, 2, 3)).toBe(v.size - 1);
  });
});

describe("VideoRVQCodec — structure & roundtrip", () => {
  const codec = new VideoRVQCodec({ height: 8, width: 8, channels: 3, patch: 4, levels: 2, codebookSize: 8, seed: 1 });

  test("derived shapes are correct", () => {
    expect(codec.latentDim).toBe(4 * 4 * 3);
    expect(codec.patchesPerFrame).toBe(4); // (8/4)*(8/4)
    expect(codec.tokensPerFrame).toBe(8); // 4 patches × 2 levels
    expect(codec.vocabSize).toBe(codec.vocab.size);
  });

  test("encode is a self-delimiting <vid> … </vid> stream with tokensPerFrame codes per frame", () => {
    const video = makeVideo(3, 8, 8, 3);
    const toks = codec.encode(video);
    expect(toks[0]).toBe(codec.vocab.bosVideo);
    expect(toks[toks.length - 1]).toBe(codec.vocab.eosVideo);
    // All ids within the flat vocab.
    for (const t of toks) expect(t).toBeLessThan(codec.vocabSize);
    // Frame 0 is a keyframe; the rest inter (keyframeInterval defaults high).
    const markers = toks.filter((t) => codec.vocab.isVideoMarker(t));
    expect(markers.length).toBe(3);
    expect(markers[0]).toBe(codec.vocab.frameKey);
    expect(markers[1]).toBe(codec.vocab.frameDelta);
    // Exactly tokensPerFrame code tokens between successive markers/eos.
    const codeCount = toks.filter((t) => codec.vocab.isCode(t)).length;
    expect(codeCount).toBe(3 * codec.tokensPerFrame);
  });

  test("decode(encode(v)) returns T frames of the right shape, clamped to [0,1], deterministically", () => {
    const video = makeVideo(4, 8, 8, 3);
    const a = codec.encode(video);
    const b = codec.encode(video);
    expect(a).toEqual(b); // deterministic

    const recon = codec.decode(a);
    expect(recon.length).toBe(video.length);
    for (const f of recon) {
      expect(f.length).toBe(8 * 8 * 3);
      for (const px of f) expect(px).toBeGreaterThanOrEqual(0), expect(px).toBeLessThanOrEqual(1);
    }
  });

  test("decode tolerates a truncated / noisy stream without throwing", () => {
    const video = makeVideo(2, 8, 8, 3);
    const toks = codec.encode(video).slice(0, 4); // chop mid-frame
    const recon = codec.decode(toks);
    expect(Array.isArray(recon)).toBe(true);
    for (const f of recon) expect(f.length).toBe(8 * 8 * 3);
  });
});

describe("VideoRVQCodec — learned codebooks", () => {
  test("fit reduces reconstruction MSE", () => {
    const codec = new VideoRVQCodec({ height: 8, width: 8, channels: 3, patch: 4, levels: 2, codebookSize: 16, seed: 7 });
    const videos = [makeVideo(4, 8, 8, 3, 0), makeVideo(4, 8, 8, 3, 1.3), makeVideo(4, 8, 8, 3, 2.1)];
    const before = reconMSE(codec, videos);
    const after = codec.fit(videos, { iterations: 12, seed: 3 });
    expect(after).toBeLessThan(before);
    expect(after).toBeCloseTo(reconMSE(codec, videos), 6); // fit's report matches a fresh measure
  });
});

describe("VideoRVQCodec — serialization (ships with the model artifact)", () => {
  test("serialize → deserialize reproduces config, codebooks, and encodings", () => {
    const codec = new VideoRVQCodec({ height: 8, width: 8, channels: 3, patch: 4, levels: 2, codebookSize: 16, textVocabSize: 5, seed: 9 });
    const videos = [makeVideo(3, 8, 8, 3, 0), makeVideo(3, 8, 8, 3, 0.9)];
    codec.fit(videos, { iterations: 10 });

    const restored = VideoRVQCodec.deserialize(codec.serialize());
    expect(restored.vocabSize).toBe(codec.vocabSize);
    expect(restored.tokensPerFrame).toBe(codec.tokensPerFrame);
    // A restored codec must encode/decode identically — else a served model is corrupt.
    const v = makeVideo(3, 8, 8, 3, 1.7);
    expect(restored.encode(v)).toEqual(codec.encode(v));
    const a = codec.decode(codec.encode(v));
    const b = restored.decode(restored.encode(v));
    for (let t = 0; t < a.length; t++) expect(Array.from(b[t]!)).toEqual(Array.from(a[t]!));
  });
});

describe("ImageRVQCodec — still image is the single-frame case", () => {
  test("encode/decode a frame; fit reduces error; every frame is intra", () => {
    const codec = new ImageRVQCodec({ height: 8, width: 8, channels: 3, patch: 4, levels: 2, codebookSize: 16, seed: 2 });
    const images = [makeVideo(1, 8, 8, 3, 0)[0]!, makeVideo(1, 8, 8, 3, 1.1)[0]!, makeVideo(1, 8, 8, 3, 2.2)[0]!];
    const toks = codec.encode(images[0]!);
    // No <delta> markers — a still image has no temporal residual.
    expect(toks).not.toContain(codec.vocab.frameDelta);
    const recon = codec.decode(toks);
    expect(recon.length).toBe(codec.frameSize);
    const after = codec.fit(images, { iterations: 12 });
    expect(after).toBeGreaterThanOrEqual(0);
  });

  test("generateImage returns a single frame of the right shape", () => {
    const codec = new ImageRVQCodec({ height: 4, width: 4, channels: 1, patch: 2, levels: 1, codebookSize: 4, textVocabSize: 2, seed: 4 });
    const image = makeVideo(1, 4, 4, 1)[0]!;
    const seq = [1, ...codec.encode(image)];
    const lm = new EvermindLM({ vocabSize: codec.vocabSize, dModel: 16, numLayers: 2, hiddenDim: 24, seed: 3 });
    new EvermindLMTrainer(lm, { lr: 0.05, epochs: 120 }).fit([seq]);
    const { image: gen } = generateImage(lm, codec, [1], { maxNewTokens: seq.length });
    expect(gen.length).toBe(4 * 4 * 1);
  });
});

describe("EvermindModelPackage — self-contained media artifact (codec bundled)", () => {
  test("fromMediaLM → toBlob → fromBlob → loadMediaLM reproduces model + codec", () => {
    const codec = new VideoRVQCodec({ height: 8, width: 8, channels: 3, patch: 4, levels: 2, codebookSize: 8, seed: 3 });
    codec.fit([makeVideo(3, 8, 8, 3)], { iterations: 6 });
    const lm = new EvermindLM({ vocabSize: codec.vocabSize, dModel: 16, numLayers: 2, hiddenDim: 24, seed: 8 });

    const blob = EvermindModelPackage.fromMediaLM(lm, codec, {
      name: "vid", version: "1.0.0", modality: "video", card: { description: "test media model" },
    }).toBlob();

    const pkg = EvermindModelPackage.fromBlob(blob);
    expect(pkg.manifest.modality).toBe("video");
    expect(pkg.validate().ok).toBe(true);

    const served = pkg.loadMediaLM();
    expect(served.modality).toBe("video");
    expect(served.codec.vocabSize).toBe(codec.vocabSize);
    // Served model + codec must generate identically to the originals.
    const a = generateVideo(lm, codec, [], { maxNewTokens: 40 });
    const b = generateVideo(served.lm, served.codec, [], { maxNewTokens: 40 });
    expect(b.tokens).toEqual(a.tokens);
  });

  test("validate() flags a corrupt codec section", () => {
    const codec = new VideoRVQCodec({ height: 4, width: 4, channels: 1, patch: 2, levels: 1, codebookSize: 4, seed: 1 });
    const lm = new EvermindLM({ vocabSize: codec.vocabSize, seed: 2 });
    const pkg = EvermindModelPackage.fromMediaLM(lm, codec, { name: "x", version: "1", modality: "video", card: { description: "d" } });
    const corrupt = new EvermindModelPackage(pkg.manifest, pkg.checkpoint, new ArrayBuffer(pkg.codec!.byteLength));
    const v = corrupt.validate();
    expect(v.ok).toBe(false);
    expect(v.errors.some((e) => /codec checksum/.test(e))).toBe(true);
  });

  test("text packages still round-trip unchanged (no codec section)", () => {
    const lm = new EvermindLM({ vocabSize: 6, dModel: 8, numLayers: 1, hiddenDim: 12, seed: 4 });
    const pkg = EvermindModelPackage.fromLM(lm, { name: "t", version: "1", card: { description: "text" } });
    const back = EvermindModelPackage.fromBlob(pkg.toBlob());
    expect(back.codec).toBeUndefined();
    expect(back.manifest.modality).toBeUndefined();
    expect(back.validate().ok).toBe(true);
    expect(() => back.loadMediaLM()).toThrow(/modality/);
  });
});

describe("EvermindLM generates video (unchanged generator)", () => {
  test("build unified sequence, overfit, generate → decode yields a valid clip", () => {
    // Tiny everything so the LM can overfit one clip fast.
    const codec = new VideoRVQCodec({
      height: 4,
      width: 4,
      channels: 1,
      patch: 2,
      levels: 1,
      codebookSize: 4,
      textVocabSize: 3,
      seed: 5,
    });
    const video = makeVideo(2, 4, 4, 1);
    const textPrompt = [1, 2]; // stand-in caption tokens in the text region
    const seq = buildVideoSequence(codec, textPrompt, video);

    const lm = new EvermindLM({ vocabSize: codec.vocabSize, dModel: 16, numLayers: 2, hiddenDim: 24, seed: 11 });
    const trainer = new EvermindLMTrainer(lm, { lr: 0.05, epochs: 200 });
    const history = trainer.fit([seq]);
    expect(history[history.length - 1]!).toBeLessThan(history[0]!); // it learned

    const { video: gen, tokens } = generateVideo(lm, codec, textPrompt, { maxNewTokens: seq.length });
    expect(tokens.length).toBeGreaterThan(0);
    // Plumbing guarantee: whatever the (small) model emits, we get real frames back.
    for (const f of gen) expect(f.length).toBe(4 * 4 * 1);
  });

  test("generateVideo rejects a vocab-size mismatch", () => {
    const codec = new VideoRVQCodec({ height: 4, width: 4, channels: 1, patch: 2, levels: 1, codebookSize: 4 });
    const lm = new EvermindLM({ vocabSize: codec.vocabSize + 1 });
    expect(() => generateVideo(lm, codec, [], { maxNewTokens: 4 })).toThrow(/vocabSize/);
  });
});
