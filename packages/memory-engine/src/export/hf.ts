/**
 * export/hf.ts — the Hugging Face repository sidecar files.
 *
 * The JSON/markdown that turns raw weights into an ingestible HF model repo:
 *   • config.json            — architecture config (model_type "evermind")
 *   • generation_config.json — default decoding params
 *   • tokenizer.json         — the byte-level BPE in HF `tokenizers` format
 *   • README.md              — the model card (YAML front-matter + body)
 *
 * The tokenizer uses the same GPT-2 byte→unicode alphabet as {@link BPETokenizer}
 * (a ByteLevel pre-tokenizer/decoder), so HF `tokenizers` / transformers.js
 * reproduce our encode/decode.
 */

import type { EvermindLM } from "../lm/evermind_lm.js";
import type { BPETokenizer } from "../tokenizer/bpe.js";
import { archOf, paramCount } from "./tensors.js";

export interface HfMeta {
  name?: string;
  version?: string;
  license?: string;
  author?: string;
  description?: string;
  tags?: string[];
}

/** HF `config.json` describing the Evermind architecture. */
export function configJson(lm: EvermindLM): Record<string, unknown> {
  const a = archOf(lm);
  return {
    model_type: "evermind",
    architectures: ["EvermindForCausalLM"],
    vocab_size: a.vocabSize,
    d_model: a.dModel,
    hidden_size: a.dModel,
    num_hidden_layers: a.numLayers,
    conv_kernel: a.convKernel,
    intermediate_size: a.hiddenDim,
    num_experts: a.numExperts,
    num_experts_per_tok: a.topK,
    rms_norm_eps: 1e-5,
    tie_word_embeddings: true,
    torch_dtype: "float32",
    transformers_version: "4.0.0",
  };
}

/** HF `generation_config.json` — greedy by default, matching the reference. */
export function generationConfigJson(): Record<string, unknown> {
  return { do_sample: false, temperature: 1.0, max_new_tokens: 64 };
}

/** HF `tokenizer.json` (the `tokenizers` fast-tokenizer format) for the BPE. */
export function tokenizerJson(tok: BPETokenizer): Record<string, unknown> {
  const vocab: Record<string, number> = {};
  for (const [token, id] of tok.vocab) vocab[token] = id;
  const merges = [...tok.merges.keys()]; // already "a b" space-separated pairs

  const added: { id: number; content: string }[] = [];
  for (const content of [tok.unkToken, tok.bosToken, tok.eosToken, tok.padToken]) {
    const id = tok.vocab.get(content);
    if (id !== undefined && !added.some((t) => t.id === id)) added.push({ id, content });
  }

  return {
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: added.map((t) => ({
      id: t.id,
      content: t.content,
      single_word: false,
      lstrip: false,
      rstrip: false,
      normalized: false,
      special: true,
    })),
    normalizer: null,
    pre_tokenizer: { type: "ByteLevel", add_prefix_space: false, trim_offsets: true, use_regex: true },
    post_processor: { type: "ByteLevel", add_prefix_space: true, trim_offsets: false, use_regex: true },
    decoder: { type: "ByteLevel", add_prefix_space: true, trim_offsets: true, use_regex: true },
    model: {
      type: "BPE",
      dropout: null,
      unk_token: null,
      continuing_subword_prefix: null,
      end_of_word_suffix: null,
      fuse_unk: false,
      byte_fallback: false,
      vocab,
      merges,
    },
  };
}

/** HF model card (`README.md`) with YAML front-matter. */
export function modelCardMarkdown(lm: EvermindLM, meta: HfMeta = {}): string {
  const a = archOf(lm);
  const params = paramCount(lm);
  const name = meta.name ?? "Evermind";
  const license = meta.license ?? "mit";
  const tags = ["evermind", "ssm", "mixture-of-experts", "text-generation", ...(meta.tags ?? [])];
  const front = [
    "---",
    `license: ${license}`,
    "library_name: builderforce-memory",
    "pipeline_tag: text-generation",
    "tags:",
    ...tags.map((t) => `  - ${t}`),
    "---",
  ].join("\n");

  const body = `
# ${name}

${meta.description ?? "A self-updating SSM language model built with the BuilderForce Evermind engine."}

An **Evermind** model: a shared-expert hybrid SSM (depthwise causal conv token-mixer
+ top-k Mixture-of-Experts channel-mixer, tied embeddings). Trained and run natively
in TypeScript/WebGPU; this repository is an export for the Hugging Face ecosystem.

## Architecture

| field | value |
| --- | --- |
| vocab size | ${a.vocabSize} |
| d_model | ${a.dModel} |
| layers | ${a.numLayers} |
| conv kernel | ${a.convKernel} |
| FFN hidden | ${a.hiddenDim} |
| experts (top-k) | ${a.numExperts} (${a.topK}) |
| parameters | ${params.toLocaleString("en-US")} |

## Files

- \`model.safetensors\` — weights (HF-native, F32).
- \`model.onnx\` — runnable graph (\`input_ids\` → \`logits\`); load with onnxruntime / transformers.js.
- \`config.json\`, \`generation_config.json\`, \`tokenizer.json\` — HF sidecars.
- \`model.gguf\` — GGUF container (custom architecture; for GGUF tooling).
- \`model.evermind\` — the native package; runs in \`@seanhogg/builderforce-memory\` (full self-update).

## Usage (ONNX, no KV-cache — re-run full context per step)

\`\`\`js
import * as ort from "onnxruntime-web";
const session = await ort.InferenceSession.create("model.onnx");
const ids = BigInt64Array.from([/* token ids */].map(BigInt));
const input_ids = new ort.Tensor("int64", ids, [1, ids.length]);
const { logits } = await session.run({ input_ids });
\`\`\`
${meta.author ? `\n## Author\n\n${meta.author}\n` : ""}`;

  return `${front}\n${body}`;
}
