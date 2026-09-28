/**
 * import/foreign/infer.ts — a Mamba-1 checkpoint's config, read off its tensors.
 *
 * `state-spaces/mamba-*` repositories publish `model.safetensors` with a
 * `config.json` in the original Mamba-SSM vocabulary (`d_model`, `n_layer`, …)
 * that no adapter reads, and a dropped safetensors file often arrives with no
 * config at all. Every geometry fact the port needs is in the tensor shapes, so
 * this derives the Hugging Face `MambaConfig` the {@link mambaAdapter} expects
 * instead of anyone restating the derivation.
 */

import type { NamedTensor } from "../../export/tensors.js";
import type { ForeignConfig } from "./adapter.js";
import { normaliseSourceName } from "./plan.js";

function fail(detail: string): never {
  throw new Error(`import/foreign: cannot read a Mamba-1 config from these tensors — ${detail}`);
}

/**
 * The `MambaForCausalLM` config implied by a Mamba-1 checkpoint's tensors:
 * embedding `[vocab, d_model]`, per-layer `in_proj [2·d_inner, d_model]`,
 * `conv1d [d_inner, (1,) d_conv]`, `A_log [d_inner, d_state]` and
 * `x_proj [dt_rank + 2·d_state, d_inner]`.
 */
export function mambaConfigFromTensors(tensors: readonly NamedTensor[]): ForeignConfig {
  const byName = new Map(tensors.map((t) => [normaliseSourceName(t.name), t]));
  const shape = (name: string) => byName.get(name)?.shape;

  const emb = shape("backbone.embeddings.weight") ?? shape("backbone.embedding.weight");
  if (!emb || emb.length !== 2) fail("no backbone.embedding(s).weight [vocab, d_model]");
  const [vocabSize, dModel] = emb as [number, number];

  let numLayers = 0;
  while (byName.has(`backbone.layers.${numLayers}.mixer.in_proj.weight`)) numLayers++;
  if (numLayers === 0) fail("no backbone.layers.N.mixer.in_proj.weight");

  const m = "backbone.layers.0.mixer";
  const inProj = shape(`${m}.in_proj.weight`)!;
  const conv = shape(`${m}.conv1d.weight`);
  const aLog = shape(`${m}.A_log`);
  const xProj = shape(`${m}.x_proj.weight`);
  if (!conv || !aLog || !xProj) fail(`layer 0 is missing conv1d.weight, A_log or x_proj.weight`);

  const dInner = inProj[0]! / 2;
  const dState = aLog[1]!;
  const dConv = conv[conv.length - 1]!;
  const dtRank = xProj[0]! - 2 * dState;
  if (!Number.isInteger(dInner) || dtRank <= 0) fail(`inconsistent shapes (in_proj ${inProj.join("×")}, x_proj ${xProj.join("×")})`);

  return {
    model_type: "mamba",
    architectures: ["MambaForCausalLM"],
    vocab_size: vocabSize,
    hidden_size: dModel,
    num_hidden_layers: numLayers,
    intermediate_size: dInner,
    state_size: dState,
    conv_kernel: dConv,
    time_step_rank: dtRank,
  };
}
