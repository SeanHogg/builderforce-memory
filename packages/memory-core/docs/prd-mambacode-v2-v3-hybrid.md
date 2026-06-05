# PRD: MambaCode.js — Mamba-2, Mamba-3, and Hybrid Model Support

**Status:** Draft
**Authors:** MambaKit Engineering
**Reference paper:** Mamba-3 — arXiv 2603.15569v1 (ICLR 2026, Lahoti et al.)

---

## 1. Executive Summary

MambaCode.js currently implements **Mamba-1** (S6 selective scan, Gu & Dao 2023). The research
landscape has advanced significantly:

- **Mamba-2** (Dao & Gu 2024) replaces the S6 scan with the Structured State Space Duality (SSD)
  algorithm, introducing multi-head SSMs, scalar A per head, chunked computation, and a simpler
  block structure — yielding better throughput on training hardware.
- **Mamba-3** (Lahoti et al., ICLR 2026) takes an *inference-first* approach, adding complex-valued
  states, MIMO recurrence, and exponential-trapezoidal discretisation. At 1.5B params it achieves
  equal perplexity to Mamba-2 with **2× smaller state size** and approximately 7× the inference
  speed of Transformers.
- **Hybrid architectures** (Jamba, Zamba-2, etc.) interleave SSM layers with full attention
  layers. Experiments show complementary strengths: attention for in-context retrieval, SSM for
  long-range compression.

This PRD specifies the exact changes required in `mambacode.js` to support all three SSM variants
and a first-class hybrid model API, while remaining backward-compatible with existing Mamba-1
checkpoints.

---

## 2. Goals

- Ship `Mamba2Block` and `Mamba3Block` alongside the existing (renamed) `Mamba1Block`.
- Add an `AttentionBlock` suitable for interleaving with SSM layers.
- Introduce a `HybridMambaModel` that accepts a per-layer type schedule.
- Upgrade the MBJS binary format to version 2 with layer-type metadata.
- Keep `MambaModel` (Mamba-1-only) as a non-breaking alias.
- Surface the new blocks through the `MambaKit` session facade with minimal API changes.

## 3. Non-Goals

- Full Flash-Attention 3 implementation (keep attention naive or tiled for WebGPU).
- Mixture-of-Experts (future PRD).
- Mobile / WASM target (WebGPU only for this release).
- Multi-GPU / sharded inference.

---

## 4. Architecture Reference

### 4.1 Mamba-1 (current implementation)

```
Input (B, L, D)
  └─ RMSNorm
  └─ in_proj  → split → [x (B,L,D_inner), z (B,L,D_inner)]
                 x ──→ conv1d (causal, kernel K)
                       └─ SiLU
                       └─ x_proj → [Δ (R), B (N), C (N)]
                                    Δ ──→ dt_proj (D)
                                         └─ softplus
                                    Selective Scan S6
                                         h_t = A_bar·h_{t-1} + B_bar·x_t
                                         y_t = C·h_t + D·x_t
                 y ──→ ⊗ SiLU(z)   ← gate
  └─ out_proj
  └─ + residual
```

**Per-block weight tensors (13 total):**

| Tensor | Shape | Notes |
|---|---|---|
| `wInProj` | `(2·D_inner, D_model)` | projects to x and z |
| `bInProj` | `(2·D_inner,)` | |
| `wConv` | `(D_inner, K)` | depthwise causal conv |
| `bConv` | `(D_inner,)` | |
| `wXProj` | `(dt_rank + 2·N, D_inner)` | produces Δ, B, C |
| `bXProj` | `(dt_rank + 2·N,)` | |
| `wDtProj` | `(D_inner, dt_rank)` | expands Δ to full dim |
| `bDtProj` | `(D_inner,)` | |
| `A_log` | `(D_inner, N)` | diagonal A, log-space |
| `D_vec` | `(D_inner,)` | skip-connection |
| `wOutProj` | `(D_model, D_inner)` | |
| `bOutProj` | `(D_model,)` | |
| `normWeight` | `(D_model,)` | pre-norm γ |

**WGSL kernels used:** `selective_scan` (forward + backward), `conv1d`, `linear`, `activations`.

---

### 4.2 Mamba-2 (SSD — Structured State Space Duality)

SSD reformulates the SSM as a block-diagonal matrix multiplication, enabling a chunked
algorithm that is more cache-friendly on GPU and directly equivalent to a constrained form
of linear attention.

**Key structural differences from Mamba-1:**

| Dimension | Mamba-1 | Mamba-2 |
|---|---|---|
| Heads | 1 implicit | H explicit heads |
| A shape | `(D_inner, N)` diagonal | `(H,)` scalar per head |
| D (skip) | `(D_inner,)` per channel | `(H,)` per head |
| Δ | `(D_inner,)` per channel | `(H,)` per head |
| Gate (z) | SiLU gate after scan | Removed; inner RMSNorm instead |
| B, C | `(N,)` shared across D | `(n_groups, N)` grouped |
| Algorithm | Parallel prefix scan (Kogge-Stone) | Chunked SSD (block-diagonal matmul) |

**Block data flow:**

```
Input (B, L, D_model)
  └─ RMSNorm
  └─ in_proj → [x (D_inner), B (n_groups·N), C (n_groups·N), dt (H)]
               ↓ causal conv on x, B, C (same kernel)
               ↓ SSD scan
                   A_bar = exp(-exp(A) · softplus(dt))   [scalar per head]
                   h_t   = A_bar · h_{t-1} + B · x_t    [MIMO per head]
                   y_t   = C · h_t
               ↓ inner RMSNorm (D_inner)
  └─ out_proj
  └─ + residual
```

**Per-block weight tensors (Mamba-2, 9 tensors):**

| Tensor | Shape | Notes |
|---|---|---|
| `wInProj` | `(D_inner + 2·n_groups·N + H, D_model)` | single fused projection |
| `wConv` | `(D_inner + 2·n_groups·N, K)` | conv on x, B, C (not dt) |
| `bConv` | `(D_inner + 2·n_groups·N,)` | |
| `A_log` | `(H,)` | scalar per head, log-space |
| `dt_bias` | `(H,)` | additive bias for dt |
| `D_vec` | `(H,)` | skip per head |
| `wOutProj` | `(D_model, D_inner)` | |
| `normWeight` | `(D_inner,)` | inner RMSNorm γ |
| `preNormWeight` | `(D_model,)` | pre-block RMSNorm γ |

**New WGSL kernels required:**
- `ssd_forward` — chunked SSD matmul (replaces `selective_scan`)
- `ssd_backward` — gradient through chunked SSD
- `grouped_conv1d_forward` / `grouped_conv1d_backward` — conv over grouped B, C

---

### 4.3 Mamba-3 (Complex-Valued MIMO SSM, inference-first)

Mamba-3 makes three targeted improvements over Mamba-2, all aimed at inference efficiency:

#### 4.3.1 Complex-Valued States

The hidden state `h` transitions from `ℝ^N` to `ℂ^(N/2)` (equivalent parameter count but higher
expressiveness). The state update becomes:

```
h_t = A_bar · h_{t-1} + B · x_t     (complex multiply)
y_t = Re(C · h_t)                    (take real part for output)
```

This requires representing A, B, C, h as pairs of float32 values (real, imag). The state space
can be **2× smaller** while solving tasks (parity, modular arithmetic) that real-valued Mamba-2
fails completely.

Implementation: store each complex float as two adjacent f32 values in GPU buffers.
The `A_log` tensor changes shape from `(H,)` to `(H, 2)` encoding `(log|A|, arg(A))`.

#### 4.3.2 MIMO (Multi-Input Multi-Output) Recurrence

Mamba-2 uses SISO (single-input, single-output) per head: one scalar input drives one scalar
output per state. MIMO allows a group of G inputs to drive a group of G outputs jointly, forming
a `G×G` transition matrix within each head.

```
h_t ∈ ℂ^(G × N/G)           [G groups of N/G complex states]
h_t = A_bar ⊗ h_{t-1} + B_block · x_t
y_t = C_block · Re(h_t)
```

where `B_block` and `C_block` are `G×G` block matrices, increasing the recurrent expressiveness
without increasing the recurrent state size.

Benefits:
- Addresses the memory-bound bottleneck of autoregressive decoding (each decode step touches the
  full recurrent state; MIMO makes this compute-bound by batching state reads).
- +1.2 percentage points downstream accuracy on top of complex values alone.

#### 4.3.3 Exponential-Trapezoidal Discretisation

Mamba-1 and Mamba-2 use Zero-Order Hold (ZOH) discretisation:
```
A_bar = exp(Δ · A)
B_bar ≈ Δ · B             (first-order approx)
```

Mamba-3 uses exponential-trapezoidal (ET) discretisation for higher accuracy:
```
A_bar = exp(Δ · A)
B_bar = (A_bar - I) · A⁻¹ · B    (exact, not approximated)
```

This is a small kernel change (one extra division per element) but meaningfully reduces
discretisation error at longer time steps.

**Per-block weight tensors (Mamba-3, 9 tensors — same count as Mamba-2):**

| Tensor | Shape | Notes |
|---|---|---|
| `wInProj` | `(D_inner + 2·n_groups·N_complex + H, D_model)` | N_complex = N/2 of Mamba-2 |
| `wConv` | `(D_inner + 2·n_groups·N_complex, K)` | |
| `bConv` | `(D_inner + 2·n_groups·N_complex,)` | |
| `A_log` | `(H, 2)` | complex: `[log\|A\|, arg(A)]` per head |
| `dt_bias` | `(H,)` | |
| `D_vec` | `(H,)` | |
| `wOutProj` | `(D_model, D_inner)` | |
| `normWeight` | `(D_inner,)` | |
| `preNormWeight` | `(D_model,)` | |

**New WGSL kernels required (beyond Mamba-2):**
- `complex_ssd_forward` — SSD scan over complex states with ET discretisation
- `complex_ssd_backward` — backward through complex SSD

---

### 4.4 Attention Block (for Hybrid Models)

A standard multi-head causal self-attention block. Kept intentionally simple for WebGPU —
naive O(L²) attention or a tiled variant, no Flash-Attention 3 dependency.

```
Input (B, L, D_model)
  └─ RMSNorm
  └─ wQKV → Q (H, D_head), K (H, D_head), V (H, D_head)
  └─ causal attention (masked)
  └─ concat heads → wO → D_model
  └─ + residual
  [optional feed-forward sublayer]
```

**Per-block weight tensors (6 tensors):**

| Tensor | Shape | Notes |
|---|---|---|
| `wQKV` | `(3·D_model, D_model)` | fused Q, K, V |
| `bQKV` | `(3·D_model,)` | |
| `wO` | `(D_model, D_model)` | output projection |
| `bO` | `(D_model,)` | |
| `normWeight` | `(D_model,)` | pre-norm |
| `ffnWeights` (optional) | varies | MLP sublayer if enabled |

**New WGSL kernels required:**
- `attention_forward` — causal multi-head attention (tiled, bf16 optional)
- `attention_backward` — gradient through attention
- `softmax_forward` / `softmax_backward`

---

## 5. Required Changes to mambacode.js

### 5.1 Layer Abstraction

Introduce a `SequenceLayer` interface that all block types implement. This is the foundation
for the hybrid model.

**New file: `src/model/sequence_layer.ts`**

```typescript
export interface LayerForwardResult {
  output: GPUBuffer;
  cache: unknown;  // type-specific per layer variant
}

export interface LayerParam {
  buf   : GPUBuffer;
  numel : number;
  name  : string;
}

/**
 * Common interface implemented by Mamba1Block, Mamba2Block, Mamba3Block,
 * and AttentionBlock. MambaModel iterates layers through this interface.
 */
export interface SequenceLayer {
  readonly layerType: 'mamba1' | 'mamba2' | 'mamba3' | 'attention';

  forward(xBuf: GPUBuffer, batch: number, seqLen: number): LayerForwardResult;
  parameters(): LayerParam[];
  getTrainableParams(): LayerParam[];
  setWSLAMode(enabled: boolean): void;
  destroy(): void;
}
```

### 5.2 Rename Existing Block

`MambaBlock` → **`Mamba1Block`** (breaking rename, but version-gated via the package version).

- `src/model/mamba_block.ts` → `src/model/mamba1_block.ts`
- Export `MambaBlock` as a deprecated alias: `export { Mamba1Block as MambaBlock }`.
- `Mamba1Block` implements `SequenceLayer` with `layerType = 'mamba1'`.

No functional changes to the existing Mamba-1 logic.

### 5.3 New File: `src/model/mamba2_block.ts`

Implements `SequenceLayer` for Mamba-2 (SSD).

Key config fields (extends a new `Mamba2BlockConfig`):

```typescript
export interface Mamba2BlockConfig {
  dModel   : number;
  dState   : number;   // N — state dim per group
  dConv    : number;   // K — conv kernel width
  expand   : number;   // dInner = expand * dModel
  nHeads   : number;   // H — number of SSM heads (dInner must be divisible by H)
  nGroups  : number;   // number of B/C groups (default 1, i.e. MQA-style)
  chunkLen : number;   // SSD chunk length (default 256)
}
```

The `forward()` method runs:
1. Pre-block RMSNorm
2. Fused `wInProj` → split into `x`, `B`, `C`, `dt`
3. Causal conv1d over `x`, `B`, `C`
4. `ssd_forward` kernel (chunked)
5. Inner RMSNorm on scan output
6. `wOutProj`
7. Residual add

The `parameters()` method returns the 9 tensors listed in §4.2.

### 5.4 New File: `src/model/mamba3_block.ts`

Implements `SequenceLayer` for Mamba-3.

Config (extends `Mamba2BlockConfig`):

```typescript
export interface Mamba3BlockConfig extends Mamba2BlockConfig {
  mimoGroup : number;   // G — MIMO group size (default 1 = SISO, same as Mamba-2)
  // dState in this config is the *complex* state count per head:
  // actual float pairs stored = dState (since complex = 2 floats each)
}
```

The `forward()` method mirrors Mamba-2 except:
- Uses `complex_ssd_forward` kernel (ET discretisation, complex arithmetic)
- `A_log` interpreted as `(H, 2)` — magnitude log + phase angle
- `B`, `C` projections output interleaved real/imag pairs
- Output taken as `Re(C · h_t)`

The `parameters()` method returns the 9 tensors listed in §4.3 (note `A_log` has different numel:
`H * 2` vs Mamba-2's `H`).

### 5.5 New File: `src/model/attention_block.ts`

Implements `SequenceLayer` for causal multi-head self-attention.

Config:

```typescript
export interface AttentionBlockConfig {
  dModel  : number;
  nHeads  : number;   // must divide dModel evenly
  dHead   : number;   // = dModel / nHeads
  hasFfn  : boolean;  // include a 4×dModel FFN sublayer (default: false)
  ffnMult?: number;   // FFN expansion (default 4)
}
```

The `forward()` method uses the `attention_forward` kernel. Causal masking is baked into
the kernel via a triangular mask (no extra buffer needed — handled in WGSL with a conditional).

### 5.6 Modified File: `src/model/mamba_model.ts` → `HybridMambaModel`

Replace the fixed `MambaBlock[]` array with `SequenceLayer[]` built from a **layer schedule**.

**New interface:**

```typescript
export type LayerType = 'mamba1' | 'mamba2' | 'mamba3' | 'attention';

export interface LayerSpec {
  type    : LayerType;
  /** Override any config field for this specific layer. */
  config? : Partial<Mamba1BlockConfig | Mamba2BlockConfig | Mamba3BlockConfig | AttentionBlockConfig>;
}

export interface HybridMambaModelConfig {
  vocabSize       : number;
  dModel          : number;
  numLayers       : number;
  /**
   * Per-layer type schedule. Length must equal numLayers.
   * If omitted, defaults to all 'mamba1' (backward-compatible).
   *
   * Examples:
   *   // Pure Mamba-2
   *   layers: Array(12).fill({ type: 'mamba2' })
   *
   *   // Jamba-style: every 4th layer is attention
   *   layers: Array.from({ length: 12 }, (_, i) =>
   *     ({ type: i % 4 === 3 ? 'attention' : 'mamba2' }))
   *
   *   // Mamba-3 with attention every 6 layers
   *   layers: Array.from({ length: 24 }, (_, i) =>
   *     ({ type: i % 6 === 5 ? 'attention' : 'mamba3' }))
   */
  layers?         : LayerSpec[];

  // Shared defaults applied to all layers of each type
  // (individual LayerSpec.config overrides take precedence)
  defaultMamba1?  : Partial<Mamba1BlockConfig>;
  defaultMamba2?  : Partial<Mamba2BlockConfig>;
  defaultMamba3?  : Partial<Mamba3BlockConfig>;
  defaultAttention?: Partial<AttentionBlockConfig>;

  eosId?          : number;
}
```

**`MambaModel` remains a type alias:**

```typescript
// Backward-compatible alias — treats all layers as mamba1
export class MambaModel extends HybridMambaModel {
  constructor(device: GPUDevice, config: MambaModelConfig) {
    super(device, { ...config, layers: Array(config.numLayers).fill({ type: 'mamba1' }) });
  }
}
```

**`forward()` change:**

```typescript
for (const layer of this.layers) {
  const { output, cache } = layer.forward(hidden, batch, seqLen);
  caches.push(cache);
  hidden.destroy();
  hidden = output;
}
```

No other changes to `forward()`, `generate()`, or `embedTokens()`.

---

## 6. New WGSL Kernels

### 6.1 `src/kernels/ssd.ts` — Mamba-2 chunked SSD

| Kernel | Entry point | Purpose |
|---|---|---|
| `SSD_FORWARD_WGSL` | `ssd_chunk_forward` | Chunked SSD forward per head |
| `SSD_BACKWARD_WGSL` | `ssd_chunk_backward` | Gradient through chunked SSD |

**Algorithm sketch (forward):**
- Dispatch: `(numChunks, H, B)` workgroups
- Each workgroup processes one chunk of `chunkLen` time steps for one head and batch item
- Within each chunk: compute `A_bar` scalars (exp), then perform block-diagonal matmul between
  `B·x` and the lower-triangular `A_bar` decay matrix to get partial `y`
- Carry-over state `h` is written to a `state_carry` buffer and incorporated by the next chunk

**Workgroup shared memory:** `chunkLen` × `dState` f32 values for B·x accumulation.

### 6.2 `src/kernels/complex_ssd.ts` — Mamba-3 complex SSD

| Kernel | Entry point | Purpose |
|---|---|---|
| `COMPLEX_SSD_FORWARD_WGSL` | `complex_ssd_forward` | ET discretisation + complex state update |
| `COMPLEX_SSD_BACKWARD_WGSL` | `complex_ssd_backward` | Gradient |

**Differences from real SSD:**
- `A_log` read as `(log_mag, phase)` pairs → `A = exp(log_mag) · exp(i·phase)`
- `A_bar = exp(Δ · A)` (complex multiply)
- `B_bar = (A_bar - 1) · A⁻¹ · B` (ET, exact — complex division)
- `h_t = A_bar ⊗ h_{t-1} + B_bar · x_t` (all complex)
- `y_t = Re(C · h_t)` (real-part projection)
- Each complex value stored as two adjacent f32 in all buffers

### 6.3 `src/kernels/attention.ts` — Causal Self-Attention

| Kernel | Entry point | Purpose |
|---|---|---|
| `ATTENTION_FORWARD_WGSL` | `attention_forward` | Tiled causal multi-head attention |
| `ATTENTION_BACKWARD_WGSL` | `attention_backward` | Gradient |
| `SOFTMAX_WGSL` | `softmax_forward` | Row-wise softmax with causal mask |

**Tile size:** 16×16 (fits WebGPU 16KB shared memory per workgroup on most adapters).

**Dispatch:** `(ceil(L/16), H, B)`.

### 6.4 Additions to `src/kernels/activations.ts`

- `softmax_forward` / `softmax_backward` (also used by attention)

### 6.5 Grouped conv1d

The existing `conv1d` kernel handles shape `(B, L, D)`. Mamba-2 runs conv over a fused
`(x, B, C)` buffer. Add an overloaded dispatch path in `conv1d.ts` that accepts a `groups`
uniform (identical math, different buffer stride) to avoid code duplication.

---

## 7. Binary Format: Version 2

### 7.1 Current MBJS format (version 1)

```
[0..3]   magic   = 0x4D424A53 ('MBJS')
[4..7]   version = 1
[8..11]  nParams : uint32
[12 .. 12 + 4·nParams - 1]  numel[i] : uint32
[12 + 4·nParams ..]          float32 data, parameter-order
```

Limitation: no metadata about layer types. Loading a Mamba-2 checkpoint into a Mamba-1 model
(different `nParams`) fails with a generic "parameter count mismatch" error.

### 7.2 Proposed MBJS format (version 2)

```
[0..3]   magic   = 0x4D424A53
[4..7]   version = 2
[8..11]  nLayers : uint32          ← new
[12 .. 12 + nLayers - 1]  layerType[i] : uint8
                           0 = mamba1, 1 = mamba2, 2 = mamba3, 3 = attention
[12 + nLayers .. aligned to 4]  padding
[next 4]  nParams : uint32
[next 4·nParams]  numel[i] : uint32
[data]  float32 data
```

**Rationale:** `nLayers` + `layerType[]` lets `loadWeights()` reconstruct the correct per-layer
parameter layout before doing any size checks, producing meaningful error messages when there is
a mismatch.

### 7.3 Backward compatibility

- `loadWeights()` checks version:
  - version 1 → assumes all layers are `mamba1`, current behaviour.
  - version 2 → reads layer type array, validates per-layer parameter counts.
- `exportWeights()` always writes version 2.
- `MambaKit` session.ts: no visible API change — the binary format is internal.

---

## 8. Config System Changes (MambaKit facade)

### 8.1 `MambaSessionOptions` extensions

```typescript
export interface MambaSessionOptions {
  // ... existing fields ...

  /**
   * SSM variant applied to all layers when no layerSchedule is given.
   * Default: 'mamba1' (existing behaviour).
   */
  mambaVersion?: 'mamba1' | 'mamba2' | 'mamba3';

  /**
   * Per-layer schedule. Length must equal the resolved numLayers.
   * Overrides mambaVersion when provided.
   *
   * Shorthand helpers (see LayerSchedulePreset below):
   *   'jamba'    — every 4th layer is attention, rest mamba2
   *   'zamba'    — every 6th layer is attention, rest mamba3
   */
  layerSchedule?: LayerSpec[] | 'jamba' | 'zamba';
}
```

### 8.2 Preset schedules

```typescript
export type LayerSchedulePreset = 'jamba' | 'zamba';

function resolveLayerSchedule(
  schedule : LayerSpec[] | LayerSchedulePreset | undefined,
  numLayers: number,
  defaultType: LayerType,
): LayerSpec[] {
  if (!schedule) return Array(numLayers).fill({ type: defaultType });
  if (schedule === 'jamba') {
    return Array.from({ length: numLayers }, (_, i) =>
      ({ type: i % 4 === 3 ? 'attention' : 'mamba2' } as LayerSpec));
  }
  if (schedule === 'zamba') {
    return Array.from({ length: numLayers }, (_, i) =>
      ({ type: i % 6 === 5 ? 'attention' : 'mamba3' } as LayerSpec));
  }
  return schedule;
}
```

### 8.3 `MambaModelConfig` extension

The existing `MambaModelConfig` gains optional fields for per-variant defaults, all optional
to stay backward-compatible:

```typescript
export interface MambaModelConfig {
  // existing fields unchanged
  vocabSize  : number;
  dModel     : number;
  numLayers  : number;
  dState?    : number;
  dConv?     : number;
  expand?    : number;
  eosId?     : number;

  // new fields
  mambaVersion?  : 'mamba1' | 'mamba2' | 'mamba3';
  layerSchedule? : LayerSpec[] | 'jamba' | 'zamba';
  nHeads?        : number;   // for mamba2/mamba3/attention
  nGroups?       : number;   // for mamba2/mamba3 B/C grouping
  chunkLen?      : number;   // for mamba2/mamba3 SSD chunk
  mimoGroup?     : number;   // for mamba3 MIMO (default 1)
}
```

### 8.4 Preset table update (`src/kit/presets.ts`)

```
Preset  | dModel | numLayers | nHeads | Notes
--------|--------|-----------|--------|-------------------------------
nano    | 128    | 4         | 4      | default mamba1
small   | 256    | 6         | 8      |
medium  | 512    | 8         | 8      |
large   | 768    | 12        | 12     |
```

`nHeads` is ignored for Mamba-1 (single-head) and used for Mamba-2/3 and Attention.
`dHead = dModel / nHeads` must divide evenly — `resolveModelConfig` validates this.

---

## 9. Public API Surface (MambaKit → mambacode.js)

### 9.1 New exports from `mambacode.js`

```typescript
// New block classes
export class Mamba1Block { ... }   // renamed from MambaBlock
export class Mamba2Block { ... }
export class Mamba3Block { ... }
export class AttentionBlock { ... }
export { Mamba1Block as MambaBlock }  // deprecated alias

// Hybrid model
export class HybridMambaModel { ... }
export { MambaModel }  // unchanged constructor signature, extends HybridMambaModel

// New kernel sources
export const SSD_FORWARD_WGSL           : string;
export const SSD_BACKWARD_WGSL          : string;
export const COMPLEX_SSD_FORWARD_WGSL   : string;
export const COMPLEX_SSD_BACKWARD_WGSL  : string;
export const ATTENTION_FORWARD_WGSL     : string;
export const ATTENTION_BACKWARD_WGSL    : string;
export const SOFTMAX_WGSL               : string;

// New types
export type LayerType = 'mamba1' | 'mamba2' | 'mamba3' | 'attention';
export interface LayerSpec { type: LayerType; config?: Partial<...>; }
export interface HybridMambaModelConfig { ... }
export interface Mamba2BlockConfig { ... }
export interface Mamba3BlockConfig { ... }
export interface AttentionBlockConfig { ... }
export interface SequenceLayer { ... }
```

### 9.2 No changes to existing `mambacode.js` exports

`BPETokenizer`, `MambaTrainer`, `initWebGPU`, all GPU utilities, quantization helpers,
existing WGSL sources — all unchanged.

### 9.3 MambaKit `session.ts` changes (minimal)

`MambaSession.create()` resolves `layerSchedule` from options and passes it to
`HybridMambaModel`. The `internals` getter exposes `model: HybridMambaModel` (typed via the
existing `SessionInternals` interface — just widening the return type).

```typescript
// session.ts — create() change
const config = resolveModelConfig(options, vocabSize);
const model  = new HybridMambaModel(device, config);  // was: new MambaModel(...)
```

No changes to `complete`, `completeStream`, `adapt`, `evaluate`, `save`, `load`, `destroy`.

---

## 10. Trainer Changes (`src/training/trainer.ts`)

`MambaTrainer` currently hard-codes backward passes for Mamba-1 kernels. The trainer needs
to dispatch the correct backward kernel based on `layer.layerType`.

**Required change:**

```typescript
// Current: trainer knows the block internals directly
// New: trainer queries block for its trainable params and backward entry points

for (const layer of model.layers) {
  switch (layer.layerType) {
    case 'mamba1':    await backwardMamba1(layer, ...);  break;
    case 'mamba2':    await backwardMamba2(layer, ...);  break;
    case 'mamba3':    await backwardMamba3(layer, ...);  break;
    case 'attention': await backwardAttention(layer, ...); break;
  }
}
```

WSLA mode (`setWSLAMode`) applies only to SSM layers (`mamba1`, `mamba2`, `mamba3`).
Attention layers are always fully trained when included (they have no low-rank subset).

---

## 11. Conversion Tool Updates (`tools/convert.html`)

The existing converter handles `state-spaces/mamba` (Mamba-1) → MBJS v1.

Additional converters needed (out of scope for this PRD but planned):

- `state-spaces/mamba-2` → MBJS v2 (same repo, different checkpoint format)
- Hybrid model checkpoint assembly from separately trained SSM + attention weights

---

## 12. Testing Requirements

### Unit tests

| Test file | Coverage |
|---|---|
| `mamba2_block.test.ts` | Forward pass shape, parameter count, gradient check |
| `mamba3_block.test.ts` | Complex state update, ET discretisation, MIMO shape |
| `attention_block.test.ts` | Causal mask, multi-head split, gradient |
| `hybrid_model.test.ts` | Mixed schedule forward, `exportWeights` / `loadWeights` v2 roundtrip |
| `ssd_kernel.test.ts` | SSD output matches reference (chunked = unchunked for small L) |
| `complex_ssd_kernel.test.ts` | Complex update matches CPU reference on parity task |

### Integration tests

- `MambaSession.create({ mambaVersion: 'mamba2' })` — full pipeline, train 10 steps on
  a 100-char corpus, perplexity decreases.
- `MambaSession.create({ layerSchedule: 'jamba', modelSize: 'nano' })` — mixed schedule,
  save/load roundtrip, perplexity preserved to within floating-point tolerance.

### Regression tests

- Version 1 `.bin` file still loads correctly into `MambaModel` (backward compatibility).
- `MambaSession.create({})` (no `mambaVersion`) behaves identically to current release.

---

## 13. Migration & Versioning

| Item | Action |
|---|---|
| `mambacode.js` version | Bump to **2.0.0** (breaking: `MambaBlock` renamed, format change) |
| `MambaKit` version | Bump to **2.0.0** simultaneously |
| MBJS file format | v1 files remain readable; v2 files written by default |
| `MambaBlock` export | Kept as deprecated alias until 3.0.0 |
| Existing checkpoints | Load unchanged under `mambaVersion: 'mamba1'` (default) |

### Recommended upgrade path for consumers

```typescript
// Before (mambacode.js 1.x / mambakit 1.x)
const session = await MambaSession.create({ modelSize: 'nano' });

// After — no change needed (mamba1 is still default)
const session = await MambaSession.create({ modelSize: 'nano' });

// Opt-in to Mamba-2
const session = await MambaSession.create({ modelSize: 'nano', mambaVersion: 'mamba2' });

// Opt-in to hybrid
const session = await MambaSession.create({ modelSize: 'small', layerSchedule: 'jamba' });
```

---

## 14. Open Questions

| # | Question | Owner |
|---|---|---|
| 1 | What chunk length (`chunkLen`) is optimal for WebGPU on typical consumer GPUs? Needs benchmarking against L ∈ {128, 256, 512}. | Performance |
| 2 | Should `AttentionBlock` include a feed-forward sublayer by default? Jamba uses FFN in attention layers; Zamba omits it. | Architecture |
| 3 | MIMO group size G for Mamba-3 — paper uses G=1 at 1.5B scale. What is optimal for nano/small? | Research |
| 4 | Backward pass for complex SSD requires computing gradients through complex division (ET B_bar). Verify numerical stability at bf16. | Engineering |
| 5 | WSLA for Mamba-2/3 — which tensors are the low-rank "fast adapt" subset? `wXProj` equivalent is the fused `wInProj`; only B/C rows are selective. Proposal: WSLA trains only the rows of `wInProj` corresponding to B and C projections. | Research |
