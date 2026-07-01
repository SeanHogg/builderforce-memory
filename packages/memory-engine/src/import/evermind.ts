/**
 * import/evermind.ts — build a live {@link EvermindLM} FROM imported weights.
 *
 * The inverse of the export path: given a named/shaped tensor list (e.g. from
 * {@link safetensorsToTensors}), reconstruct the model. This is the warm-start /
 * weight-port seam — it lets Evermind be *initialised* from an existing
 * checkpoint instead of only from random init + training.
 *
 * Two entry points:
 *   • round-trip our OWN exports — the tensor names already match the canonical
 *     {@link evermindTensorSpec}; arch is inferred from the tensor shapes.
 *   • warm-start a FOREIGN SSM checkpoint (e.g. Codestral-Mamba) — pass a
 *     `rename` map that translates its tensor names into our canonical names.
 *     Shapes must then line up with {@link evermindTensorSpec}; a mismatch throws
 *     with the exact offending tensor (the honest failure for an incompatible
 *     mixer, rather than silently loading garbage).
 */

import { EvermindLM } from "../lm/evermind_lm.js";
import { evermindTensorSpec, type EvermindArch, type NamedTensor } from "../export/tensors.js";
import { safetensorsToTensors } from "./safetensors.js";

export interface ImportOptions {
  /**
   * Rename each source tensor before matching against the canonical spec. Return
   * `null` to drop a tensor. Use this to warm-start a foreign SSM checkpoint whose
   * weight names differ from Evermind's.
   */
  rename?: (name: string) => string | null;
  /**
   * `topK` (active experts per token) is not encoded in the weights — supply it
   * when importing (default: min(2, numExperts)).
   */
  topK?: number;
  /** Override the inferred architecture entirely (advanced). */
  arch?: EvermindArch;
}

/**
 * Infer the architecture from the tensor shapes alone, so a bare safetensors
 * file (no config.json) can be imported. `topK` cannot be recovered from weights.
 */
export function inferArchFromTensors(tensors: NamedTensor[], topK?: number): EvermindArch {
  const byName = new Map(tensors.map((t) => [t.name, t]));
  const emb = byName.get("token_embedding.weight");
  if (!emb || emb.shape.length !== 2) throw new Error("import: missing token_embedding.weight [vocab, dModel]");
  const [vocabSize, dModel] = emb.shape as [number, number];

  // Count layers by the `layers.<n>.` prefix; experts by `moe.experts.<e>`.
  let numLayers = 0;
  let numExperts = 0;
  for (const t of tensors) {
    const lm = /^layers\.(\d+)\./.exec(t.name);
    if (lm) numLayers = Math.max(numLayers, Number(lm[1]) + 1);
    const em = /^layers\.\d+\.moe\.experts\.(\d+)\./.exec(t.name);
    if (em) numExperts = Math.max(numExperts, Number(em[1]) + 1);
  }
  if (numLayers === 0) throw new Error("import: no `layers.<n>.` tensors found");

  const conv = byName.get("layers.0.conv.weight");
  if (!conv || conv.shape.length !== 2) throw new Error("import: missing layers.0.conv.weight [dModel, convKernel]");
  const convKernel = conv.shape[1]!;

  const w1 = byName.get("layers.0.moe.shared.w1");
  if (!w1 || w1.shape.length !== 2) throw new Error("import: missing layers.0.moe.shared.w1 [hiddenDim, dModel]");
  const hiddenDim = w1.shape[0]!;

  return {
    vocabSize,
    dModel,
    numLayers,
    convKernel,
    hiddenDim,
    numExperts,
    topK: Math.min(topK ?? 2, Math.max(1, numExperts)),
  };
}

/**
 * Reconstruct an {@link EvermindLM} from a named tensor list. Validates every
 * canonical parameter is present with the right element count, then copies the
 * data into the model's parameter buffers (canonical order).
 */
export function importEvermindTensors(tensors: NamedTensor[], opts: ImportOptions = {}): EvermindLM {
  const renamed: NamedTensor[] = [];
  for (const t of tensors) {
    const name = opts.rename ? opts.rename(t.name) : t.name;
    if (name != null) renamed.push({ ...t, name });
  }
  const byName = new Map(renamed.map((t) => [t.name, t]));

  const arch = opts.arch ?? inferArchFromTensors(renamed, opts.topK);
  const specs = evermindTensorSpec(arch);

  // Validate presence + element count BEFORE constructing, so we fail with the
  // exact missing/mismatched tensor rather than loading a corrupt model.
  const ordered: Float32Array[] = [];
  for (const s of specs) {
    const t = byName.get(s.name);
    if (!t) throw new Error(`import: missing tensor "${s.name}" (expected shape [${s.shape.join(", ")}])`);
    const numel = s.shape.reduce((n, d) => n * d, 1);
    if (t.data.length !== numel) {
      throw new Error(`import: tensor "${s.name}" has ${t.data.length} elements, expected ${numel} ([${s.shape.join(", ")}])`);
    }
    ordered.push(t.data);
  }

  const lm = new EvermindLM({
    vocabSize: arch.vocabSize,
    dModel: arch.dModel,
    numLayers: arch.numLayers,
    convKernel: arch.convKernel,
    hiddenDim: arch.hiddenDim,
    numExperts: arch.numExperts,
    topK: arch.topK,
  });
  const params = lm.parameters();
  if (params.length !== ordered.length) {
    throw new Error(`import: parameter-count drift (model ${params.length} vs imported ${ordered.length})`);
  }
  for (let i = 0; i < params.length; i++) params[i]!.data.set(ordered[i]!);
  return lm;
}

/** Convenience: import an {@link EvermindLM} directly from a `.safetensors` buffer. */
export function importEvermind(bytes: Uint8Array, opts: ImportOptions = {}): EvermindLM {
  return importEvermindTensors(safetensorsToTensors(bytes), opts);
}
