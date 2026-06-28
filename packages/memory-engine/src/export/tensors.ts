/**
 * export/tensors.ts — the canonical named-tensor view of an EvermindLM.
 *
 * Every export format (safetensors / ONNX / GGUF / HF) consumes the SAME named,
 * shaped tensor list — defined ONCE here so the formats never disagree about a
 * weight's name, shape, or order. The runtime is left untouched: we read the
 * model purely through its public {@link EvermindLM.config} and
 * {@link EvermindLM.parameters} (the AdamW-ordered flat buffers), and re-attach
 * the names/shapes that order implies.
 *
 * Parameter order (must mirror EvermindLM.parameters / SharedExpertMoE.parameters):
 *   emb,
 *   per layer l: conv[l], normConv[l], normMoe[l],
 *                moe: router, shared.{w1,b1,w2,b2}, expert{e}.{w1,b1,w2,b2}…
 */

import type { EvermindLM } from "../lm/evermind_lm.js";

/** A named, shaped weight tensor (row-major, flat f32). */
export interface NamedTensor {
  name: string;
  /** Row-major dimensions, e.g. [vocab, dModel]. */
  shape: number[];
  data: Float32Array;
}

/** The flat numeric architecture an export needs (a superset of the LM config). */
export interface EvermindArch {
  vocabSize: number;
  dModel: number;
  numLayers: number;
  convKernel: number;
  hiddenDim: number;
  numExperts: number;
  topK: number;
}

/** Pull the architecture out of a live LM (or any config-shaped object). */
export function archOf(lm: EvermindLM): EvermindArch {
  const c = lm.config;
  return {
    vocabSize: c.vocabSize,
    dModel: c.dModel,
    numLayers: c.numLayers,
    convKernel: c.convKernel,
    hiddenDim: c.hiddenDim,
    numExperts: c.numExperts,
    topK: c.topK,
  };
}

/** A weight name + its expected shape, in canonical parameter order. */
export interface TensorSpec {
  name: string;
  shape: number[];
}

/** The name+shape of every parameter, in EvermindLM.parameters() order. */
export function evermindTensorSpec(a: EvermindArch): TensorSpec[] {
  const { vocabSize: V, dModel: D, numLayers: L, convKernel: K, hiddenDim: H, numExperts: E } = a;
  const specs: TensorSpec[] = [{ name: "token_embedding.weight", shape: [V, D] }];
  const ffn = (prefix: string): TensorSpec[] => [
    { name: `${prefix}.w1`, shape: [H, D] },
    { name: `${prefix}.b1`, shape: [H] },
    { name: `${prefix}.w2`, shape: [D, H] },
    { name: `${prefix}.b2`, shape: [D] },
  ];
  for (let l = 0; l < L; l++) {
    const p = `layers.${l}`;
    specs.push({ name: `${p}.conv.weight`, shape: [D, K] });
    specs.push({ name: `${p}.norm_conv.weight`, shape: [D] });
    specs.push({ name: `${p}.norm_moe.weight`, shape: [D] });
    specs.push({ name: `${p}.moe.router.weight`, shape: [E, D] });
    specs.push(...ffn(`${p}.moe.shared`));
    for (let e = 0; e < E; e++) specs.push(...ffn(`${p}.moe.experts.${e}`));
  }
  return specs;
}

/**
 * The full named/shaped tensor list of a trained LM, zipping the deterministic
 * {@link evermindTensorSpec} against the live {@link EvermindLM.parameters}. The
 * data buffers are the model's own (no copy) — treat them as read-only.
 */
export function namedTensors(lm: EvermindLM): NamedTensor[] {
  const specs = evermindTensorSpec(archOf(lm));
  const params = lm.parameters();
  if (specs.length !== params.length) {
    throw new Error(
      `export/tensors: parameter-count drift (spec ${specs.length} vs model ${params.length}) — ` +
        `EvermindLM.parameters() order changed; update evermindTensorSpec`,
    );
  }
  return specs.map((s, i) => {
    const numel = s.shape.reduce((n, d) => n * d, 1);
    const data = params[i]!.data;
    if (data.length !== numel) {
      throw new Error(`export/tensors: shape mismatch for ${s.name} (expected ${numel}, got ${data.length})`);
    }
    return { name: s.name, shape: s.shape, data };
  });
}

/** Total trainable scalar parameters (for sizing / model-card display). */
export function paramCount(lm: EvermindLM): number {
  return lm.parameters().reduce((n, p) => n + p.data.length, 0);
}
