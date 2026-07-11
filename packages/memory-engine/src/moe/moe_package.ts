/**
 * moe_package.ts — EvermindModelPackage: the portable, publishable AI artifact.
 *
 * This is the unit a creator publishes to the marketplace and a buyer downloads
 * and runs: a self-describing manifest (name, version, config, model card,
 * integrity checksum) bundled with the trained checkpoint into one `.evermind`
 * blob. It is the contract every downstream consumer (marketplace listing,
 * purchase entitlement, workflow generator) reads — define it once, here.
 *
 * Zero-dep and isomorphic: serialises via TextEncoder/TextDecoder, runs in the
 * browser (where models are trained) and in Node/Workers (where they execute).
 */

import { SharedExpertMoE, type MoEConfig } from "./moe_model.js";
import { EvermindLM, type EvermindLMConfig } from "../lm/evermind_lm.js";
import { VideoRVQCodec } from "../codec/video_rvq.js";

/** First 4 bytes of a serialised package: "EVM1". */
const PKG_MAGIC = 0x45564d31;
const PKG_VERSION = 1;

/** The kinds of model an `.evermind` package can carry. */
export type EvermindModelType = "shared-expert-moe" | "evermind-lm";

/** Output modality. `text` (default) needs no codec; `video`/`image` bundle a VRQ codec. */
export type EvermindModality = "text" | "video" | "image";

/** Human-facing description published with the model (the "model card"). */
export interface EvermindModelCard {
  description: string;
  /** What it was trained on / intended for. */
  trainingSummary?: string;
  /** SPDX id or free text (e.g. "MIT", "proprietary"). */
  license?: string;
  author?: string;
  tags?: string[];
}

/** Self-describing header for a published Evermind model. */
export interface EvermindModelManifest {
  schema: "evermind.model/1";
  name: string;
  version: string;
  modelType: EvermindModelType;
  /** Output modality; absent ⇒ "text". Video/image also carry a `codec` section. */
  modality?: EvermindModality;
  /** Flat numeric model config (the constructor args), serialised verbatim. */
  config: Record<string, number>;
  /** Total trainable scalar parameters (for sizing / pricing / display). */
  paramCount: number;
  checkpointFormat: "MoE0" | "EVL0";
  checkpointFp16: boolean;
  /**
   * Byte length of the checkpoint section — REQUIRED when a codec section follows
   * (so the reader knows where the checkpoint ends). Absent ⇒ checkpoint runs to
   * end-of-blob (the original text-only layout, still read verbatim).
   */
  checkpointBytes?: number;
  /** 32-bit FNV-1a over the checkpoint bytes — integrity for download/purchase. */
  checksum: number;
  /** Media codec section format (present for video/image packages). */
  codecFormat?: "VRQ0";
  /** 32-bit FNV-1a over the codec bytes — integrity for the bundled codec. */
  codecChecksum?: number;
  card: EvermindModelCard;
  /** ISO timestamp, caller-supplied (the engine avoids Date for determinism). */
  createdAt?: string;
}

export interface PackageMeta {
  name: string;
  version: string;
  card: EvermindModelCard;
  /** Store the checkpoint in fp16 (half the size). Default false. */
  fp16?: boolean;
  createdAt?: string;
}

export interface ValidationResult {
  ok: boolean;
  errors: string[];
}

/** 32-bit FNV-1a hash — dependency-free integrity check over the checkpoint. */
function fnv1a(bytes: Uint8Array): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i]!;
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * A trained model packaged for publishing: manifest + checkpoint. Build one with
 * {@link EvermindModelPackage.fromModel}, ship {@link toBlob}, and a buyer
 * reconstitutes it with {@link fromBlob} → {@link validate} → {@link loadModel}.
 */
export class EvermindModelPackage {
  constructor(
    readonly manifest: EvermindModelManifest,
    readonly checkpoint: ArrayBuffer,
    /** Serialized media codec ("VRQ0" blob), present for video/image packages. */
    readonly codec?: ArrayBuffer,
  ) {}

  /** Package a trained model with its publishing metadata. */
  static fromModel(model: SharedExpertMoE, meta: PackageMeta): EvermindModelPackage {
    const fp16 = meta.fp16 ?? false;
    const checkpoint = model.exportWeights({ fp16 });
    const manifest: EvermindModelManifest = {
      schema: "evermind.model/1",
      name: meta.name,
      version: meta.version,
      modelType: "shared-expert-moe",
      config: model.config,
      paramCount: model.parameters().reduce((n, p) => n + p.numel, 0),
      checkpointFormat: "MoE0",
      checkpointFp16: fp16,
      checksum: fnv1a(new Uint8Array(checkpoint)),
      card: meta.card,
      ...(meta.createdAt ? { createdAt: meta.createdAt } : {}),
    };
    return new EvermindModelPackage(manifest, checkpoint);
  }

  /** Package a trained generative {@link EvermindLM} — the runnable marketplace AI. */
  static fromLM(lm: EvermindLM, meta: PackageMeta): EvermindModelPackage {
    const fp16 = meta.fp16 ?? false;
    const checkpoint = lm.exportWeights({ fp16 });
    const manifest: EvermindModelManifest = {
      schema: "evermind.model/1",
      name: meta.name,
      version: meta.version,
      modelType: "evermind-lm",
      config: lm.config as unknown as Record<string, number>,
      paramCount: lm.parameters().reduce((n, p) => n + p.data.length, 0),
      checkpointFormat: "EVL0",
      checkpointFp16: fp16,
      checksum: fnv1a(new Uint8Array(checkpoint)),
      card: meta.card,
      ...(meta.createdAt ? { createdAt: meta.createdAt } : {}),
    };
    return new EvermindModelPackage(manifest, checkpoint);
  }

  /**
   * Package a video/image generative model: its {@link EvermindLM} weights AND the
   * {@link VideoRVQCodec} needed to turn generated tokens back into frames. Without
   * the bundled codec a served media model would emit undecodable token ids — this
   * is what makes an `.evermind` media artifact self-contained and servable.
   */
  static fromMediaLM(
    lm: EvermindLM,
    codec: VideoRVQCodec,
    meta: PackageMeta & { modality: "video" | "image" },
  ): EvermindModelPackage {
    if (lm.config.vocabSize !== codec.vocabSize) {
      throw new Error(
        `fromMediaLM: LM vocabSize (${lm.config.vocabSize}) must equal codec.vocabSize (${codec.vocabSize})`,
      );
    }
    const fp16 = meta.fp16 ?? false;
    const checkpoint = lm.exportWeights({ fp16 });
    const codecBlob = codec.serialize();
    const manifest: EvermindModelManifest = {
      schema: "evermind.model/1",
      name: meta.name,
      version: meta.version,
      modelType: "evermind-lm",
      modality: meta.modality,
      config: lm.config as unknown as Record<string, number>,
      paramCount: lm.parameters().reduce((n, p) => n + p.data.length, 0),
      checkpointFormat: "EVL0",
      checkpointFp16: fp16,
      checkpointBytes: checkpoint.byteLength,
      checksum: fnv1a(new Uint8Array(checkpoint)),
      codecFormat: "VRQ0",
      codecChecksum: fnv1a(new Uint8Array(codecBlob)),
      card: meta.card,
      ...(meta.createdAt ? { createdAt: meta.createdAt } : {}),
    };
    return new EvermindModelPackage(manifest, checkpoint, codecBlob);
  }

  /** Serialise to a single `.evermind` blob: magic, version, manifest, checkpoint[, codec]. */
  toBlob(): ArrayBuffer {
    const manifestBytes = new TextEncoder().encode(JSON.stringify(this.manifest));
    const headerBytes = 12; // magic, version, manifestLen
    const codecBytes = this.codec?.byteLength ?? 0;
    const out = new ArrayBuffer(headerBytes + manifestBytes.byteLength + this.checkpoint.byteLength + codecBytes);
    const head = new Uint32Array(out, 0, 3);
    head[0] = PKG_MAGIC;
    head[1] = PKG_VERSION;
    head[2] = manifestBytes.byteLength;
    let o = headerBytes;
    new Uint8Array(out, o, manifestBytes.byteLength).set(manifestBytes);
    o += manifestBytes.byteLength;
    new Uint8Array(out, o, this.checkpoint.byteLength).set(new Uint8Array(this.checkpoint));
    o += this.checkpoint.byteLength;
    if (this.codec) new Uint8Array(out, o, codecBytes).set(new Uint8Array(this.codec));
    return out;
  }

  /** Parse a `.evermind` blob. Throws on bad magic / truncation. */
  static fromBlob(buffer: ArrayBuffer): EvermindModelPackage {
    if (buffer.byteLength < 12) throw new Error("EvermindModelPackage.fromBlob: truncated (no header)");
    const head = new Uint32Array(buffer, 0, 3);
    if (head[0] !== PKG_MAGIC) throw new Error("EvermindModelPackage.fromBlob: bad magic (not an .evermind package)");
    const manifestLen = head[2]!;
    const headerBytes = 12;
    if (headerBytes + manifestLen > buffer.byteLength) {
      throw new Error("EvermindModelPackage.fromBlob: truncated (manifest length exceeds blob)");
    }
    const manifestBytes = new Uint8Array(buffer, headerBytes, manifestLen);
    const manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as EvermindModelManifest;
    const bodyStart = headerBytes + manifestLen;
    // A codec section follows the checkpoint only when the manifest fixed the
    // checkpoint length; otherwise the checkpoint runs to end-of-blob (text layout).
    const cpBytes = manifest.checkpointBytes;
    if (cpBytes != null) {
      if (bodyStart + cpBytes > buffer.byteLength) {
        throw new Error("EvermindModelPackage.fromBlob: truncated (checkpointBytes exceeds blob)");
      }
      const checkpoint = buffer.slice(bodyStart, bodyStart + cpBytes);
      const codec = bodyStart + cpBytes < buffer.byteLength ? buffer.slice(bodyStart + cpBytes) : undefined;
      return new EvermindModelPackage(manifest, checkpoint, codec);
    }
    return new EvermindModelPackage(manifest, buffer.slice(bodyStart));
  }

  /** Verify integrity + structural sanity before trusting a downloaded package. */
  validate(): ValidationResult {
    const errors: string[] = [];
    if (this.manifest.schema !== "evermind.model/1") errors.push(`unknown schema: ${this.manifest.schema}`);
    const c = this.manifest.config;
    if (this.manifest.modelType === "shared-expert-moe") {
      if (!c || c.modelDim! <= 0 || c.hiddenDim! <= 0 || c.numExperts! <= 0 || c.topK! < 1 || c.topK! > c.numExperts!) {
        errors.push("invalid config");
      }
    } else if (this.manifest.modelType === "evermind-lm") {
      if (!c || c.vocabSize! <= 0 || c.dModel! <= 0 || c.numLayers! <= 0 || c.numExperts! <= 0 || c.topK! < 1 || c.topK! > c.numExperts!) {
        errors.push("invalid config");
      }
    } else {
      errors.push(`unsupported modelType: ${String(this.manifest.modelType)}`);
    }
    const actual = fnv1a(new Uint8Array(this.checkpoint));
    if (actual !== this.manifest.checksum) {
      errors.push(`checksum mismatch (manifest ${this.manifest.checksum}, actual ${actual}) — corrupt or tampered checkpoint`);
    }
    // Media packages must carry an integral codec — a served media model can't
    // decode tokens → frames without it.
    const modality = this.manifest.modality ?? "text";
    if (modality === "video" || modality === "image") {
      if (!this.codec) {
        errors.push(`modality '${modality}' package is missing its codec section`);
      } else {
        const codecActual = fnv1a(new Uint8Array(this.codec));
        if (codecActual !== this.manifest.codecChecksum) {
          errors.push(`codec checksum mismatch (manifest ${this.manifest.codecChecksum}, actual ${codecActual}) — corrupt codec`);
        }
      }
    }
    return { ok: errors.length === 0, errors };
  }

  /** Reconstruct the bare MoE layer. Validates first; throws if invalid / wrong type. */
  loadModel(): SharedExpertMoE {
    const v = this.validate();
    if (!v.ok) throw new Error(`EvermindModelPackage.loadModel: ${v.errors.join("; ")}`);
    if (this.manifest.modelType !== "shared-expert-moe") {
      throw new Error(`loadModel: package is '${this.manifest.modelType}', use loadLM()`);
    }
    const model = new SharedExpertMoE(this.manifest.config as Partial<MoEConfig>);
    model.loadWeights(this.checkpoint);
    return model;
  }

  /** Reconstruct the runnable generative model — what a marketplace buyer runs. */
  loadLM(): EvermindLM {
    const v = this.validate();
    if (!v.ok) throw new Error(`EvermindModelPackage.loadLM: ${v.errors.join("; ")}`);
    if (this.manifest.modelType !== "evermind-lm") {
      throw new Error(`loadLM: package is '${this.manifest.modelType}', use loadModel()`);
    }
    const lm = new EvermindLM(this.manifest.config as unknown as EvermindLMConfig);
    lm.loadWeights(this.checkpoint);
    return lm;
  }

  /**
   * Reconstruct a video/image model: the generative {@link EvermindLM} plus its
   * {@link VideoRVQCodec}. Pair with `generateVideo` / `generateImage` to produce
   * frames. This is what a media-serving path (gateway) loads and runs.
   */
  loadMediaLM(): { lm: EvermindLM; codec: VideoRVQCodec; modality: EvermindModality } {
    const v = this.validate();
    if (!v.ok) throw new Error(`EvermindModelPackage.loadMediaLM: ${v.errors.join("; ")}`);
    const modality = this.manifest.modality ?? "text";
    if (modality !== "video" && modality !== "image") {
      throw new Error(`loadMediaLM: package modality is '${modality}', use loadLM()`);
    }
    return { lm: this.loadLM(), codec: VideoRVQCodec.deserialize(this.codec!), modality };
  }
}
