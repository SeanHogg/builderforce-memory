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

/** First 4 bytes of a serialised package: "EVM1". */
const PKG_MAGIC = 0x45564d31;
const PKG_VERSION = 1;

/** The kinds of model an `.evermind` package can carry. */
export type EvermindModelType = "shared-expert-moe" | "evermind-lm";

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
  /** Flat numeric model config (the constructor args), serialised verbatim. */
  config: Record<string, number>;
  /** Total trainable scalar parameters (for sizing / pricing / display). */
  paramCount: number;
  checkpointFormat: "MoE0" | "EVL0";
  checkpointFp16: boolean;
  /** 32-bit FNV-1a over the checkpoint bytes — integrity for download/purchase. */
  checksum: number;
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

  /** Serialise to a single `.evermind` blob: magic, version, manifest, checkpoint. */
  toBlob(): ArrayBuffer {
    const manifestBytes = new TextEncoder().encode(JSON.stringify(this.manifest));
    const headerBytes = 12; // magic, version, manifestLen
    const out = new ArrayBuffer(headerBytes + manifestBytes.byteLength + this.checkpoint.byteLength);
    const head = new Uint32Array(out, 0, 3);
    head[0] = PKG_MAGIC;
    head[1] = PKG_VERSION;
    head[2] = manifestBytes.byteLength;
    new Uint8Array(out, headerBytes, manifestBytes.byteLength).set(manifestBytes);
    new Uint8Array(out, headerBytes + manifestBytes.byteLength).set(new Uint8Array(this.checkpoint));
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
    const checkpoint = buffer.slice(headerBytes + manifestLen);
    return new EvermindModelPackage(manifest, checkpoint);
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
}
