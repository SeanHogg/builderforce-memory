# @seanhogg/builderforce-memory-engine

> BuilderForce Agent Memory — **engine layer**. WebGPU-accelerated Mamba-1/2/3 and Hybrid SSM kernels, written in **TypeScript**, compiled for use in any JavaScript application. *(Formerly published as `@seanhogg/mambacode.js`.)*

[![npm](https://img.shields.io/npm/v/@seanhogg/builderforce-memory-engine)](https://www.npmjs.com/package/@seanhogg/builderforce-memory-engine)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

`@seanhogg/builderforce-memory-engine` is a **TypeScript-first** library that brings the Mamba family of State Space Models to the browser via WebGPU. It ships **Mamba-2** (SSD), **Mamba-3** (complex-valued MIMO + ET discretisation), and **hybrid attention** layers, while remaining fully backward-compatible with Mamba-1 checkpoints. It is the zero-runtime-dep engine consumed by [`@seanhogg/builderforce-memory`](../memory) (the runtime layer).

> **Technical report & peer review.** The kernels and discretisations here are specified mathematically in the Evermind technical report ([`../../publication/evermind/`](../../publication/evermind)). Its adversarial [`PEER-REVIEW.md`](../../publication/evermind/PEER-REVIEW.md) flags engine-level items worth knowing before depending on them at scale: the "parallel" selective scan walks tiles sequentially, `softplus` is the unstable `log(1+exp(v))` form, int8 quantisation uses a single per-tensor scale (weights only), and GPU buffers are allocated per call (no pool). Tracked as `EVM-1/EVM-8`. Several of these were addressed in the v2026.6.33–v2026.6.35 hardening (e.g. stable softplus); the engine also now ships a benchmarking harness (`src/bench/`). See the resolution addendum in [`PEER-REVIEW.md`](../../publication/evermind/PEER-REVIEW.md).

---

## What's New in v2.0.0

| Feature | Detail |
|---|---|
| **Mamba-2 (SSD)** | Structured State Space Duality — chunked matmul scan, multi-head, scalar A, inner RMSNorm |
| **Mamba-3** | Complex-valued states (ℂ^N), ET discretisation, MIMO recurrence, 2× smaller state size |
| **AttentionBlock** | Causal multi-head attention for hybrid (Jamba/Zamba) layer schedules |
| **HybridMambaModel** | Per-layer type schedule — mix mamba1/2/3/attention freely |
| **MBJS v2 format** | Layer-type metadata in checkpoint header; v1 files still load unchanged |
| **`MambaBlock` alias** | Kept as deprecated alias for `Mamba1Block` until 3.0.0 |

---

## Key Features

| Feature | Detail |
|---|---|
| **TypeScript-first** | Full type declarations shipped with the package |
| **Plain JS compatible** | Import the compiled `dist/` in any JavaScript project |
| **SSM variants** | Mamba-1 (S6), Mamba-2 (SSD), Mamba-3 (complex MIMO+ET) |
| **Hybrid models** | Jamba/Zamba-style mixed SSM + attention schedules |
| **Hardware target** | WebGPU (WGSL) — Chrome 113+, Edge 113+, Firefox Nightly |
| **No heavy frameworks** | Zero TensorFlow.js / Transformers.js dependencies |
| **On-device training** | Tape-based autograd + AdamW GPU optimizer |
| **Quantization** | FP16 weights, Int8 activations |
| **Tokenizer** | Browser-side BPE (Qwen2.5-Coder compatible) |
| **WSLA mode** | Fast-adapt: trains only the selective projection rows |
| **Benchmarking** | Held-out perplexity, bits-per-token, top-1/top-k accuracy, throughput + model A/B |

---

## Benchmarking

`src/bench/` measures a trained model on held-out data — the question the "beats a frozen LLM" thesis rests on: *how good is this model?* It scores any object exposing `forward(tokens) => { logits }` (the engine's `EvermindLM` satisfies it directly), and is dependency-free.

```ts
import { benchmarkModel, compareModels, trainAndBenchmark } from '@seanhogg/builderforce-memory-engine';

// Score a model on held-out token sequences
const report = benchmarkModel(model, heldOutSequences);
// → { perplexity, bitsPerToken, top1Accuracy, topKAccuracy, tokensPerSecond, ... }

// A/B two checkpoints (lower perplexity wins)
const cmp = compareModels(candidate, baseline, heldOutSequences);
console.log(cmp.summary); // "Candidate wins: 8.3% lower perplexity (…)"

// One-call build + score from raw text (enforced held-out split)
const result = trainAndBenchmark(corpus, { epochs: 30, heldOutRatio: 0.25 });
```

Metrics: held-out perplexity, bits-per-token, top-1/top-k next-token accuracy, and forward throughput. 14 unit tests in `tests/bench.test.ts`. This is the measurement instrument behind the Evermind technical report's H2/H3 hypotheses, and it drives the Studio's on-device benchmark scorecard.

---

## Installation

```bash
npm install @seanhogg/builderforce-memory-engine
```

Build from source:

```bash
npm run build   # compiles TypeScript → dist/
```

---

## Quick Start

### Mamba-1 (backward-compatible, unchanged)

```ts
import { MambaModel, MambaTrainer, BPETokenizer, initWebGPU } from '@seanhogg/builderforce-memory-engine';

const { device } = await initWebGPU();
const tokenizer  = new BPETokenizer();
await tokenizer.load('/vocab.json', '/merges.txt');

const model = new MambaModel(device, {
  vocabSize : tokenizer.vocabSize,
  dModel    : 512,
  numLayers : 8,
});

await model.loadWeights(await (await fetch('/checkpoint.bin')).arrayBuffer());
const ids = await model.generate(tokenizer.encode('function add('), 200);
console.log(tokenizer.decode(ids));
```

### Mamba-2 (SSD)

```ts
import { HybridMambaModel } from '@seanhogg/builderforce-memory-engine';

const model = new HybridMambaModel(device, {
  vocabSize : tokenizer.vocabSize,
  dModel    : 512,
  numLayers : 8,
  nHeads    : 8,
  layers    : Array(8).fill({ type: 'mamba2' }),
});
```

### Mamba-3 (complex states)

```ts
const model = new HybridMambaModel(device, {
  vocabSize : tokenizer.vocabSize,
  dModel    : 512,
  numLayers : 8,
  nHeads    : 8,
  layers    : Array(8).fill({ type: 'mamba3' }),
});
```

### Hybrid (Jamba-style: every 4th layer is attention)

```ts
const model = new HybridMambaModel(device, {
  vocabSize : tokenizer.vocabSize,
  dModel    : 512,
  numLayers : 12,
  nHeads    : 8,
  layers    : Array.from({ length: 12 }, (_, i) => ({
    type: i % 4 === 3 ? 'attention' : 'mamba2',
  })),
});
```

---

## Architecture Reference

### Mamba-1 Block (S6)

```
Input (B, L, D)
  └─ RMSNorm
  └─ in_proj → x, z (gate)
                x → conv1d → SiLU → x_proj → Δ, B, C
                                              Δ → dt_proj → softplus
                                              Selective Scan S6
                                              h_t = Ā·h_{t-1} + B̄·x_t
                                              y_t = C·h_t + D·x_t
  └─ y * SiLU(z)
  └─ out_proj + residual
```

### Mamba-2 Block (SSD)

```
Input (B, L, D)
  └─ RMSNorm
  └─ in_proj → [x (D_inner), B (G·N), C (G·N), dt (H)]
               conv1d over x, B, C (fused)
               SSD scan: A_bar = exp(-softplus(A) · softplus(dt))
                          h_t = A_bar · h_{t-1} + B · x_t
                          y_t = C · h_t
  └─ inner RMSNorm
  └─ out_proj + residual
```

### Mamba-3 Block (complex MIMO, ET)

```
Input (B, L, D)
  └─ Same structure as Mamba-2 but:
     • A ∈ ℂ (log|A|, arg(A)) per head
     • A_bar = exp(Δ·A)  [complex]
     • B_bar = (A_bar − 1)·A⁻¹·B  [ET, exact]
     • h_t ∈ ℂ^(N/2), y_t = Re(C·h_t)
```

### AttentionBlock (causal MHA)

```
Input (B, L, D)
  └─ RMSNorm
  └─ wQKV → Q, K, V (B, L, H, d_head)
  └─ scores = Q·Kᵀ / √d_head  (causal mask)
  └─ softmax → weighted V sum
  └─ concat heads → wO + residual
  [optional FFN sublayer]
```

---

## File Structure

```
src/
├── index.ts                         ← public API entry point (v2.0.0)
├── kernels/
│   ├── selective_scan.ts            ← WGSL: S6 forward/backward (Mamba-1)
│   ├── ssd.ts                       ← WGSL: chunked SSD forward/backward (Mamba-2)
│   ├── complex_ssd.ts               ← WGSL: complex SSD + ET + MIMO (Mamba-3)
│   ├── attention.ts                 ← WGSL: tiled causal MHA forward/backward
│   ├── conv1d.ts                    ← WGSL: 1D causal convolution (+ groups param)
│   ├── linear_projection.ts         ← WGSL: tiled GEMM
│   ├── weight_update.ts             ← WGSL: AdamW + gradient clipping
│   └── activations.ts               ← WGSL: SiLU, RMSNorm, Softmax
├── model/
│   ├── sequence_layer.ts            ← SequenceLayer interface (LayerType, LayerParam)
│   ├── mamba1_block.ts              ← Mamba1Block (renamed from MambaBlock)
│   ├── mamba2_block.ts              ← Mamba2Block (SSD)
│   ├── mamba3_block.ts              ← Mamba3Block (complex + MIMO + ET)
│   ├── attention_block.ts           ← AttentionBlock (causal MHA)
│   └── mamba_model.ts               ← HybridMambaModel + MambaModel alias
├── training/
│   ├── autograd.ts                  ← Tape-based AD + loss helpers
│   └── trainer.ts                   ← MambaTrainer (AdamW, WSLA)
├── bench/
│   ├── metrics.ts                   ← perplexity, bits-per-token, top-1/top-k accuracy
│   └── harness.ts                   ← benchmarkModel / compareModels / trainAndBenchmark
├── tokenizer/
│   └── bpe.ts                       ← Browser-side BPE tokenizer
└── utils/
    ├── gpu_utils.ts                 ← WebGPU device/buffer management
    └── quantization.ts              ← FP16 / Int8 quantization

tools/                               ← Model building & checkpoint tooling
├── generate-bin.js                  ← CLI: generate an MBJS v2 checkpoint from scratch
├── pretrain.html                    ← Browser: pretrain a model on a text corpus
└── convert.html                     ← Browser: convert HuggingFace Mamba → MBJS format

tests/
├── kernels.test.ts
├── autograd.test.ts
├── bpe.test.ts
├── quantization.test.ts
└── bench.test.ts

docs/
├── getting-started.md
├── integration-architecture.md
├── weight-lifecycle.md
├── api-reference.md
└── prd-mambacode-v2-v3-hybrid.md    ← PRD: Mamba-2/3/hybrid implementation spec
```

---

## Tools

The `tools/` directory contains model-building and checkpoint utilities that operate at the engine level. These are **not part of the public API** — they are for authors who want to build, pretrain, or convert model weights.

### `tools/generate-bin.js` — Generate a blank MBJS checkpoint

Creates a properly-shaped MBJS v2 `.bin` file with randomly initialised weights. Useful as a starting point before pretraining.

```bash
node tools/generate-bin.js                        # nano → model.bin
node tools/generate-bin.js --size small           # small preset
node tools/generate-bin.js --size nano --out my.bin
```

The weights are **not pretrained** — use `pretrain.html` to run language-model training.

### `tools/pretrain.html` — Browser pretraining UI

In-browser training loop over a text corpus. Requires a WebGPU-capable browser.

```bash
npm run build
npm run serve
# Open http://localhost:3000/tools/pretrain.html
# Load a corpus (e.g. TinyStories), configure size/epochs, click Start Training
# Download the resulting .bin checkpoint
```

### `tools/convert.html` — HuggingFace → MBJS converter

Converts `state-spaces/mamba` safetensors checkpoints to MBJS format.

```bash
# Open http://localhost:3000/tools/convert.html
# Drop model.safetensors from huggingface.co/state-spaces/mamba-130m
# Download converted .bin
```

---

## WGSL Kernels

| Kernel file | Entry points | Used by |
|---|---|---|
| `selective_scan.ts` | `forward_scan`, `forward_reduce`, `selective_scan_backward` | Mamba-1 |
| `ssd.ts` | `ssd_chunk_forward`, `ssd_chunk_backward` | Mamba-2 |
| `complex_ssd.ts` | `complex_ssd_forward`, `complex_ssd_backward` | Mamba-3 |
| `attention.ts` | `attention_forward`, `attention_value`, `attention_backward` | Attention |
| `conv1d.ts` | `conv1d_forward`, `conv1d_backward_dx`, `conv1d_backward_dw` | All SSM |
| `linear_projection.ts` | `linear_forward`, `linear_backward_dX`, `linear_backward_dW` | All layers |
| `activations.ts` | `silu_forward`, `rmsnorm_forward`, `softmax_forward_simple` | All layers |
| `weight_update.ts` | `adamw_update`, `grad_norm_reduce`, `grad_clip_scale` | Training |

---

## MBJS Binary Format

### Version 1 (legacy, still readable)

```
[0..3]   magic   = 0x4D424A53 ('MBJS')
[4..7]   version = 1
[8..11]  nParams : uint32
[12 ..]  numel[i] : uint32  (×nParams)
[data]   float32 values
```

### Version 2 (written by default from v2.0.0)

```
[0..3]   magic   = 0x4D424A53
[4..7]   version = 2
[8..11]  nLayers : uint32
[12 ..]  layerType[i] : uint8  (0=mamba1, 1=mamba2, 2=mamba3, 3=attention)
[pad]    aligned to 4 bytes
[next4]  nParams : uint32
[next..]  numel[i] : uint32  (×nParams)
[data]   float32 values
```

Version 1 files are loaded transparently — all layers assumed `mamba1`.

---

## Migration from v1.x

```ts
// v1.x — no change needed (mamba1 default is preserved)
const model = new MambaModel(device, config);

// v2.x — opt into Mamba-2
const model = new HybridMambaModel(device, { ...config, layers: Array(8).fill({ type: 'mamba2' }) });

// v2.x — MambaBlock is a deprecated alias for Mamba1Block; both still work
import { MambaBlock, Mamba1Block } from '@seanhogg/builderforce-memory-engine';
```

---

## Testing

```bash
npm test        # unit tests (no GPU required)
npm run build   # compile TypeScript → dist/
npm run lint    # ESLint
```

---

## Browser Compatibility

| Browser | Version | Status |
|---|---|---|
| Chrome | 113+ | ✅ Supported |
| Edge | 113+ | ✅ Supported |
| Firefox | Nightly | ✅ (flag: `dom.webgpu.enabled`) |
| Safari | 18+ | ⚠️ Partial |
| Node.js | — | ❌ Not supported |

---

## Acknowledgements

- **Mamba-3** — Lahoti et al., *Mamba: The Hard Way* (arXiv 2603.15569, ICLR 2026)
- **Mamba-2** — Dao & Gu, *Transformers are SSMs* (arXiv 2405.21060, 2024)
- **Mamba-1** — Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (arXiv 2312.00752, 2023)

---

## Professional Platform

**Want managed infrastructure for your SSM models?**

[**Builderforce.ai**](https://builderforce.ai) is the professional enterprise platform built on this engine. It provides:

- **In-browser LoRA training** — fine-tune up to 2B-parameter models on instruction datasets using the engine's WebGPU kernels, entirely client-side
- **Hybrid Local Brain** — the Mamba State Engine runs a selective scan alongside Transformers.js inference for persistent agent memory, powered by these WGSL kernels
- **Dataset generation** — LLM-assisted JSONL instruction dataset creation with streaming progress
- **Workforce Registry** — publish trained models as specialist AI agents; discoverable and hirable by the community
- **Agent portability** — `AgentPackage` bundles the LoRA adapter, `MambaStateSnapshot`, and agent profile into a single portable JSON artifact
- **BuilderForce Agents mesh** — trained agents deploy as self-hosted coding agents via [BuilderForce Agents](https://builderforce.ai), orchestrated from Builderforce

Use this engine to build and experiment locally. Use Builderforce.ai to deploy, manage, and share at scale.

```
@seanhogg/builderforce-memory-engine (WebGPU kernels)
      ↓
@seanhogg/builderforce-memory (session API + runtime + memory)
      ↓
 Builderforce.ai (enterprise IDE + training + registry)
      ↓
   BuilderForce Agents (self-hosted agent mesh)
```

---

## License

MIT
