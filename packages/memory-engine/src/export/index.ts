/**
 * export/index.ts — the model-export registry.
 *
 * One entry point — {@link exportEvermind} — turns a trained {@link EvermindLM}
 * (+ its tokenizer) into a named set of files in a requested format. The runtime
 * is untouched: export reads the model through its public surface only. This is
 * the *publishing* seam — training/serving stay in the WebGPU/TS runtime; export
 * is offered as a separate step that emits portable artifacts (ONNX, safetensors,
 * GGUF) and a complete Hugging Face repo bundle.
 *
 * Formats:
 *   • "safetensors"  → model.safetensors
 *   • "onnx"         → model.onnx
 *   • "gguf"         → model.gguf
 *   • "huggingface"  → a full HF repo (safetensors + onnx + gguf + config +
 *                      tokenizer.json + generation_config + README), ready to push
 */

import type { EvermindLM } from "../lm/evermind_lm.js";
import type { BPETokenizer } from "../tokenizer/bpe.js";
import { paramCount } from "./tensors.js";
import { exportSafetensors } from "./safetensors.js";
import { exportOnnx } from "./onnx.js";
import { exportGguf } from "./gguf.js";
import { configJson, generationConfigJson, tokenizerJson, modelCardMarkdown, type HfMeta } from "./hf.js";

export type ExportFormat = "safetensors" | "onnx" | "gguf" | "huggingface";

/** One emitted file. `data` is bytes (binary) or a string (text). */
export interface ExportFile {
  path: string;
  data: Uint8Array | string;
  contentType: string;
}

export interface ExportResult {
  format: ExportFormat;
  files: ExportFile[];
  /** Total trainable scalar parameters of the exported model. */
  paramCount: number;
}

export interface ExportOptions extends HfMeta {
  /** Store weights as float16 where the format supports it. Default false. */
  fp16?: boolean;
}

/** Catalog of formats for UI pickers (id + human label + extension). */
export const EXPORT_FORMATS: { id: ExportFormat; label: string; description: string; ext: string }[] = [
  { id: "huggingface", label: "Hugging Face repo", description: "Full publishable repo: safetensors + ONNX + GGUF + config + tokenizer + model card", ext: "/" },
  { id: "onnx", label: "ONNX", description: "Runnable graph for onnxruntime / transformers.js (input_ids → logits)", ext: ".onnx" },
  { id: "safetensors", label: "Safetensors", description: "HF-native weight format (lossless F32 / half-size F16)", ext: ".safetensors" },
  { id: "gguf", label: "GGUF", description: "llama.cpp container (custom architecture; for GGUF tooling)", ext: ".gguf" },
];

const JSON_CT = "application/json";

function jsonFile(path: string, obj: unknown): ExportFile {
  return { path, data: JSON.stringify(obj, null, 2), contentType: JSON_CT };
}

/**
 * Export a trained model in the requested format. `tokenizer` is required for the
 * "huggingface" bundle (to emit tokenizer.json); the weight-only formats ignore it.
 */
export function exportEvermind(
  lm: EvermindLM,
  format: ExportFormat,
  opts: ExportOptions = {},
  tokenizer?: BPETokenizer,
): ExportResult {
  const params = paramCount(lm);
  const base = (): ExportResult => ({ format, files: [], paramCount: params });

  switch (format) {
    case "safetensors": {
      const r = base();
      r.files.push({
        path: "model.safetensors",
        data: exportSafetensors(lm, { fp16: opts.fp16 }),
        contentType: "application/octet-stream",
      });
      return r;
    }
    case "onnx": {
      const r = base();
      r.files.push({
        path: "model.onnx",
        data: exportOnnx(lm, { producerVersion: opts.version }),
        contentType: "application/octet-stream",
      });
      return r;
    }
    case "gguf": {
      const r = base();
      r.files.push({
        path: "model.gguf",
        data: exportGguf(lm, { name: opts.name, fp16: opts.fp16 }),
        contentType: "application/octet-stream",
      });
      return r;
    }
    case "huggingface": {
      if (!tokenizer) throw new Error("exportEvermind('huggingface'): a tokenizer is required for tokenizer.json");
      const r = base();
      r.files.push(
        { path: "model.safetensors", data: exportSafetensors(lm, { fp16: opts.fp16 }), contentType: "application/octet-stream" },
        { path: "model.onnx", data: exportOnnx(lm, { producerVersion: opts.version }), contentType: "application/octet-stream" },
        { path: "model.gguf", data: exportGguf(lm, { name: opts.name, fp16: opts.fp16 }), contentType: "application/octet-stream" },
        jsonFile("config.json", configJson(lm)),
        jsonFile("generation_config.json", generationConfigJson()),
        jsonFile("tokenizer.json", tokenizerJson(tokenizer)),
        { path: "README.md", data: modelCardMarkdown(lm, opts), contentType: "text/markdown" },
      );
      return r;
    }
    default:
      throw new Error(`exportEvermind: unknown format "${String(format)}"`);
  }
}

export { exportSafetensors, tensorsToSafetensors } from "./safetensors.js";
export { exportOnnx } from "./onnx.js";
export { exportGguf } from "./gguf.js";
export { configJson, generationConfigJson, tokenizerJson, modelCardMarkdown } from "./hf.js";
export type { HfMeta } from "./hf.js";
export { namedTensors, evermindTensorSpec, archOf, paramCount } from "./tensors.js";
export type { NamedTensor, EvermindArch, TensorSpec } from "./tensors.js";
