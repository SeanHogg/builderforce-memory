# MambaKit — Product Requirements Document

> **A prescriptive, opinionated facade over MambaCode.js that removes all boilerplate and makes on-device AI accessible in a single import.**

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Proposed API Design](#4-proposed-api-design)
5. [Architecture](#5-architecture)
6. [Opinionated Defaults](#6-opinionated-defaults)
7. [Integration Scenarios](#7-integration-scenarios)
8. [Configuration Reference](#8-configuration-reference)
9. [Error Handling Strategy](#9-error-handling-strategy)
10. [Persistence Strategy](#10-persistence-strategy)
11. [Event & Progress Model](#11-event--progress-model)
12. [Performance Considerations](#12-performance-considerations)
13. [Migration Path from Low-Level API](#13-migration-path-from-low-level-api)
14. [Implementation Plan](#14-implementation-plan)
15. [File Structure](#15-file-structure)
16. [Testing Requirements](#16-testing-requirements)

---

## 1. Background & Motivation

MambaCode.js exposes a powerful, flexible low-level API for running and fine-tuning Mamba State Space Models entirely in the browser using WebGPU. The library provides every building block: GPU device management, BPE tokenization, the Mamba block forward pass, an autograd tape, AdamW optimization, weight serialization, and quantization utilities.

However, "flexible and powerful" is also "verbose and complex". A consumer who simply wants to load a code model and start generating completions must currently:

1. Call `initWebGPU()` and destructure the device
2. Instantiate `BPETokenizer` and await `tokenizer.load(vocabUrl, mergesUrl)`
3. Manually construct a `MambaModelConfig` object with at least three required fields
4. Instantiate `MambaModel(device, config)`
5. `fetch()` a checkpoint URL and await `model.loadWeights(buffer)`
6. Instantiate `MambaTrainer(model, tokenizer)` if any training is needed
7. Manually call `tokenizer.encode()`, `model.generate()`, and `tokenizer.decode()`
8. Manually handle `model.exportWeights()` and storage (IndexedDB, download, File System API)

That is eight distinct async steps before a single token can be generated — each with its own error modes, configuration objects, and resource lifecycle responsibilities. This level of ceremony is appropriate for a library author or advanced integrator, but it is a significant barrier for:

- Front-end engineers building a code-completion widget
- VSCode extension authors who want on-device suggestions
- Teams at builderforce.ai integrating the model into a hosted platform UI
- Educators and researchers who want to experiment rapidly

**MambaKit** is a thin, opinionated facade layer that collapses this setup into a single class and single `create()` call, while remaining 100 % backed by the existing MambaCode.js implementation.

---

## 2. Problem Statement

### Current Boilerplate (minimum viable example)

```ts
// Eight async steps to reach first token
import {
  initWebGPU, BPETokenizer, MambaModel, MambaTrainer,
  type MambaModelConfig,
} from 'mambacode.js';

const { device } = await initWebGPU();

const tokenizer = new BPETokenizer();
await tokenizer.load('/vocab.json', '/merges.txt');

const config: MambaModelConfig = {
  vocabSize : tokenizer.vocabSize,
  dModel    : 512,
  numLayers : 8,
  dState    : 16,
  dConv     : 4,
  expand    : 2,
};
const model = new MambaModel(device, config);

const response = await fetch('/models/checkpoint.bin');
await model.loadWeights(await response.arrayBuffer());

const trainer  = new MambaTrainer(model, tokenizer);
const promptIds = tokenizer.encode('function fibonacci(');
const outputIds = await model.generate(promptIds, 200, { temperature: 0.8 });
console.log(tokenizer.decode(outputIds));
```

### Pain Points

| Pain point | Impact |
|---|---|
| 8-step async setup sequence | Developer must know the internals before writing a single line of business logic |
| Manual tokenizer encode/decode | Leaks implementation detail: consumers must think in token IDs |
| Manual device management | `GPUDevice` is an advanced WebGPU concept irrelevant to application code |
| Manual checkpoint fetch | Every project reinvents the same `fetch + arrayBuffer + loadWeights` pattern |
| No built-in persistence | Saving weights requires knowing about IndexedDB, Blob URLs, and File System API |
| No progress events | Training is a black box until `onEpochEnd` is wired manually |
| No retry / error recovery | Any step that fails leaves partially initialised objects |
| Dual object management | Trainer and model are separate objects consumers must keep in sync |

---

## 3. Goals & Non-Goals

### Goals

- **G1 — Zero-boilerplate setup.** A working generate call must be achievable in ≤ 3 lines of code.
- **G2 — Text in, text out.** All public methods accept and return plain strings. Token IDs are an implementation detail hidden inside the facade.
- **G3 — Opinionated defaults.** Sensible defaults for model size, sampling, training hyperparameters, and persistence mean most consumers never need to touch configuration.
- **G4 — Single object lifecycle.** One `MambaSession` instance manages the GPU device, tokenizer, model, and trainer. The consumer never holds references to internal objects.
- **G5 — Built-in persistence.** `save()` and `load()` abstract over IndexedDB (default), download links, and File System API — chosen automatically based on environment capabilities.
- **G6 — Progress events.** All long-running operations emit typed events via a standard `addEventListener`-style interface, making progress UI trivially easy to add.
- **G7 — Escape hatch.** Advanced users can reach the underlying `MambaModel`, `MambaTrainer`, and `BPETokenizer` instances through a `.internals` property to use the full low-level API at any time.
- **G8 — Tree-shakeable.** MambaKit is a separate entry point (`mambacode.js/kit`). Consumers of the low-level API do not import any MambaKit code.
- **G9 — Full TypeScript coverage.** Every public symbol ships with declaration files.
- **G10 — No new dependencies.** MambaKit wraps MambaCode.js only; it does not introduce any npm dependencies beyond the existing devDependencies.

### Non-Goals

- **NG1 — Not a replacement.** MambaKit does not change, rewrite, or fork MambaCode.js. Every GPU operation is executed by the existing MambaCode.js classes.
- **NG2 — Not a server library.** MambaKit targets browser environments, exactly as MambaCode.js does today. Node.js support is explicitly out of scope.
- **NG3 — Not an orchestration layer.** Prompt chaining, RAG, tool calls, and multi-turn conversation management are application concerns outside this PRD.
- **NG4 — Not a new weight format.** MambaKit uses the existing `.bin` weight format unchanged.

---

## 4. Proposed API Design

### 4.1 Primary Class: `MambaSession`

`MambaSession` is the single entry point for all MambaKit functionality. It is always created via the static async factory `MambaSession.create()`.

```ts
import { MambaSession } from 'mambacode.js/kit';

const session = await MambaSession.create({
  checkpointUrl : '/models/mamba-coder-base.bin',
  vocabUrl      : '/vocab.json',
  mergesUrl     : '/merges.txt',
});
```

From that point:

```ts
// Generate a code completion
const completion = await session.complete('function fibonacci(n: number): number {');

// Fine-tune on private code
await session.adapt(myCodeString);

// Evaluate quality
const perplexity = await session.evaluate(heldOutCode);

// Save the current weights (persists to IndexedDB by default)
await session.save();

// Reload a previously saved session
await session.load();

// Tear down GPU resources
session.destroy();
```

---

### 4.2 `MambaSession.create(options)` — Static Factory

**Signature**

```ts
static async create(options: MambaSessionOptions): Promise<MambaSession>
```

**What it does internally**

1. Calls `initWebGPU()` with the power preference from options
2. Creates and loads `BPETokenizer` from the supplied vocab/merges URLs (or in-memory objects)
3. Builds `MambaModelConfig` from options, filling in all opinionated defaults
4. Constructs `MambaModel(device, config)`
5. If `checkpointUrl` is supplied, fetches and calls `model.loadWeights(buffer)`, with automatic retry
6. Constructs `MambaTrainer(model, tokenizer)`
7. Returns the fully-initialised `MambaSession`

If any step fails the factory throws a single typed `MambaKitError` (see [Section 9](#9-error-handling-strategy)).

**Progress events during create**

The factory emits `'progress'` events so a loading bar can be displayed:

```ts
const session = await MambaSession.create(options, {
  onProgress: (event) => {
    // event.stage: 'gpu' | 'tokenizer' | 'model' | 'weights'
    // event.progress: 0.0 – 1.0
    updateLoadingBar(event.progress);
  },
});
```

---

### 4.3 `session.complete(prompt, options?)` — Text Generation

```ts
async complete(prompt: string, options?: CompleteOptions): Promise<string>
```

- Calls `tokenizer.encode(prompt)` internally
- Calls `model.generate(promptIds, maxNewTokens, samplingOpts)` internally
- Calls `tokenizer.decode(outputIds)` internally
- Returns the **continuation only** (not the original prompt), matching standard completion API conventions

```ts
const result = await session.complete('function add(a: number, b: number)', {
  maxNewTokens : 100,
  temperature  : 0.7,
  topK         : 40,
  topP         : 0.9,
});
// result: ": number {\n  return a + b;\n}"
```

---

### 4.4 `session.completeStream(prompt, options?)` — Streaming Generation

Returns an `AsyncIterable<string>` that yields one decoded token string at a time, enabling real-time streaming UIs.

```ts
async *completeStream(prompt: string, options?: CompleteOptions): AsyncIterable<string>
```

```ts
for await (const token of session.completeStream('function fibonacci(')) {
  process.stdout.write(token);
}
```

---

### 4.5 `session.adapt(text, options?)` — Fine-Tuning (WSLA by default)

```ts
async adapt(text: string, options?: AdaptOptions): Promise<AdaptResult>
```

- Wraps `trainer.train(text, opts)`
- Defaults to WSLA mode (`wsla: true`) for fast, low-memory adaptation
- Returns an `AdaptResult` containing the per-epoch loss array

```ts
const result = await session.adapt(myPrivateCodebase, {
  epochs       : 3,
  learningRate : 1e-4,
  onProgress   : (epoch, loss) => console.log(`Epoch ${epoch}: ${loss.toFixed(4)}`),
});
console.log('Final loss:', result.losses.at(-1));
```

---

### 4.6 `session.evaluate(text)` — Perplexity Evaluation

```ts
async evaluate(text: string): Promise<number>
```

Delegates to `trainer.evaluate(text)`. Returns perplexity (lower = better).

```ts
const ppl = await session.evaluate(heldOutCode);
console.log(`Perplexity: ${ppl.toFixed(2)}`);
```

---

### 4.7 `session.save(options?)` — Persist Weights

```ts
async save(options?: SaveOptions): Promise<void>
```

Persists model weights. Storage target is chosen automatically unless overridden:

| Priority | Storage mechanism | When used |
|---|---|---|
| 1 | `indexedDB` | Always available in browser (default) |
| 2 | `download` | Explicit option or if IndexedDB is unavailable |
| 3 | `fileSystem` | Explicit option; uses File System Access API |

```ts
// Default — saves to IndexedDB under the session name
await session.save();

// Explicit download
await session.save({ storage: 'download', filename: 'my-model.bin' });

// File System Access API (prompts for save location)
await session.save({ storage: 'fileSystem' });
```

---

### 4.8 `session.load(options?)` — Restore Weights

```ts
async load(options?: LoadOptions): Promise<boolean>
```

Loads a previously saved checkpoint. Returns `true` if a checkpoint was found and loaded, `false` if no checkpoint exists (so the model continues with its current weights).

```ts
const restored = await session.load();
if (!restored) {
  console.log('No saved checkpoint found — using base model');
}
```

---

### 4.9 `session.destroy()` — Resource Cleanup

```ts
destroy(): void
```

Destroys all GPU buffers and releases the `GPUDevice`. Should be called when the session is no longer needed (e.g. on page unload).

```ts
window.addEventListener('unload', () => session.destroy());
```

---

### 4.10 `session.internals` — Low-Level Escape Hatch

```ts
readonly internals: {
  device    : GPUDevice;
  model     : MambaModel;
  trainer   : MambaTrainer;
  tokenizer : BPETokenizer;
}
```

Exposes all underlying MambaCode.js objects for advanced use cases. This is the official "escape hatch" — any feature not supported by MambaKit can be accessed through `internals` without losing the MambaKit lifecycle management.

```ts
// Access a raw GPU buffer directly
const params = session.internals.model.parameters();

// Use WGSL kernel helpers
import { createComputePipeline } from 'mambacode.js';
const pipeline = createComputePipeline(session.internals.device, CUSTOM_WGSL, 'main');
```

---

### 4.11 Complete TypeScript Type Definitions

```ts
// ── Session Options ────────────────────────────────────────────────────────────

export interface MambaSessionOptions {
  /** URL to a .bin checkpoint file. Optional — model starts with random weights if omitted. */
  checkpointUrl?  : string;

  /** URL to vocab.json (Qwen3.5-Coder compatible). Required unless vocabObject is supplied. */
  vocabUrl?       : string;

  /** URL to merges.txt. Required unless mergesArray is supplied. */
  mergesUrl?      : string;

  /** In-memory vocabulary object — alternative to vocabUrl. */
  vocabObject?    : Record<string, number>;

  /** In-memory merges array — alternative to mergesUrl. */
  mergesArray?    : string[];

  /** Unique name for this session, used as the IndexedDB key. Default: 'default'. */
  name?           : string;

  /**
   * Model size preset. Overrides individual model config fields.
   * - 'nano'    : dModel=128,  numLayers=4   (~6M params, fastest)
   * - 'small'   : dModel=256,  numLayers=6   (~20M params)
   * - 'medium'  : dModel=512,  numLayers=8   (~50M params, default)
   * - 'large'   : dModel=768,  numLayers=12  (~120M params)
   * - 'custom'  : use modelConfig directly
   */
  modelSize?      : 'nano' | 'small' | 'medium' | 'large' | 'custom';

  /** Fine-grained model configuration. Only used when modelSize is 'custom'. */
  modelConfig?    : Partial<MambaModelConfig>;

  /** WebGPU power preference. Default: 'high-performance'. */
  powerPreference?: 'high-performance' | 'low-power';

  /** Number of times to retry a failed checkpoint fetch. Default: 2. */
  fetchRetries?   : number;
}

// ── Complete Options ───────────────────────────────────────────────────────────

export interface CompleteOptions {
  maxNewTokens? : number;    // Default: 200
  temperature?  : number;    // Default: 0.8
  topK?         : number;    // Default: 50
  topP?         : number;    // Default: 0.9
}

// ── Adapt Options ──────────────────────────────────────────────────────────────

export interface AdaptOptions {
  epochs?       : number;    // Default: 3
  learningRate? : number;    // Default: 1e-4
  seqLen?       : number;    // Default: 512
  wsla?         : boolean;   // Default: true  (WSLA fast-adapt mode)
  fullTrain?    : boolean;   // Convenience alias: sets wsla=false and epochs=5
  onProgress?   : (epoch: number, loss: number) => void;
}

export interface AdaptResult {
  losses     : number[];
  epochCount : number;
  durationMs : number;
}

// ── Save / Load Options ────────────────────────────────────────────────────────

export type StorageTarget = 'indexedDB' | 'download' | 'fileSystem';

export interface SaveOptions {
  storage?  : StorageTarget;    // Default: 'indexedDB'
  filename? : string;           // Used by 'download' and 'fileSystem'. Default: '<name>.bin'
  key?      : string;           // IndexedDB key override. Default: session name
}

export interface LoadOptions {
  storage?  : StorageTarget;    // Default: 'indexedDB'
  url?      : string;           // Used when storage is not 'indexedDB'
  key?      : string;           // IndexedDB key override. Default: session name
}

// ── Progress Event ─────────────────────────────────────────────────────────────

export type CreateStage = 'gpu' | 'tokenizer' | 'model' | 'weights';

export interface CreateProgressEvent {
  stage    : CreateStage;
  progress : number;   // 0.0 – 1.0 within the current stage
  message  : string;   // Human-readable description
}

// ── Result Types ───────────────────────────────────────────────────────────────

export interface SessionInternals {
  device    : GPUDevice;
  model     : MambaModel;
  trainer   : MambaTrainer;
  tokenizer : BPETokenizer;
}
```

---

## 5. Architecture

### 5.1 Facade Pattern

MambaKit implements a classic **Facade** pattern. The `MambaSession` class is the facade: it holds references to the low-level subsystems and provides a simplified, unified interface. None of the existing MambaCode.js code changes.

```
┌────────────────────────────────────────────────────────────────────┐
│                         Consumer Code                              │
│                                                                    │
│  const session = await MambaSession.create(options);               │
│  const text    = await session.complete(prompt);                   │
│  await session.adapt(code);                                        │
│  await session.save();                                             │
└──────────────────────────────┬─────────────────────────────────────┘
                               │  delegates to
┌──────────────────────────────▼─────────────────────────────────────┐
│                      MambaSession (MambaKit)                       │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  _device    : GPUDevice       (from initWebGPU)             │   │
│  │  _tokenizer : BPETokenizer    (from tokenizer.load)         │   │
│  │  _model     : MambaModel      (from new MambaModel)         │   │
│  │  _trainer   : MambaTrainer    (from new MambaTrainer)       │   │
│  └──────────────────────────────────────────┬──────────────────┘   │
│                                             │  calls               │
└─────────────────────────────────────────────┼────────────────────--┘
                                              │
┌─────────────────────────────────────────────▼──────────────────────┐
│                  MambaCode.js (unchanged)                           │
│                                                                    │
│  initWebGPU()         BPETokenizer.load()     MambaModel           │
│  MambaModel.forward() MambaModel.generate()   MambaTrainer.train() │
│  MambaTrainer.evaluate()  MambaModel.exportWeights()  ...          │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 Module Layout

MambaKit is implemented as a separate sub-path export so it is fully tree-shakeable:

```
src/
└── kit/
    ├── index.ts           ← public entry point for 'mambacode.js/kit'
    ├── session.ts         ← MambaSession class
    ├── presets.ts         ← modelSize preset configurations
    ├── persistence.ts     ← IndexedDB / download / FileSystem save/load helpers
    ├── streaming.ts       ← completeStream AsyncIterable adapter
    └── errors.ts          ← MambaKitError typed error class
```

The top-level `src/index.ts` is **not modified**. The new `package.json` exports field gains an additional entry:

```json
"exports": {
  ".": {
    "import": "./dist/index.js",
    "types":  "./dist/index.d.ts"
  },
  "./kit": {
    "import": "./dist/kit/index.js",
    "types":  "./dist/kit/index.d.ts"
  }
}
```

### 5.3 Initialisation Sequence

```
MambaSession.create(options)
    │
    ├─ emit progress: { stage: 'gpu', progress: 0.0 }
    ├─ initWebGPU({ powerPreference })
    ├─ emit progress: { stage: 'gpu', progress: 1.0 }
    │
    ├─ emit progress: { stage: 'tokenizer', progress: 0.0 }
    ├─ new BPETokenizer()
    ├─ tokenizer.load(vocabUrl, mergesUrl)   [or loadFromObjects]
    ├─ emit progress: { stage: 'tokenizer', progress: 1.0 }
    │
    ├─ emit progress: { stage: 'model', progress: 0.0 }
    ├─ resolveModelConfig(options)           [applies preset + defaults]
    ├─ new MambaModel(device, config)
    ├─ new MambaTrainer(model, tokenizer)
    ├─ emit progress: { stage: 'model', progress: 1.0 }
    │
    ├─ [if checkpointUrl provided]
    │    ├─ emit progress: { stage: 'weights', progress: 0.0 }
    │    ├─ fetch(checkpointUrl) with retry
    │    ├─ model.loadWeights(buffer)
    │    └─ emit progress: { stage: 'weights', progress: 1.0 }
    │
    └─ return new MambaSession(...)
```

### 5.4 `complete()` Call Flow

```
session.complete(prompt, opts)
    │
    ├─ tokenizer.encode(prompt)  →  promptIds: number[]
    ├─ model.generate(promptIds, maxNewTokens, samplingOpts)  →  outputIds: number[]
    ├─ tokenizer.decode(outputIds)  →  fullText: string
    └─ return fullText.slice(prompt.length)   // continuation only
```

### 5.5 `adapt()` Call Flow

```
session.adapt(text, opts)
    │
    ├─ [record start time]
    ├─ trainer.train(text, {
    │      epochs, learningRate, seqLen,
    │      wsla: opts.wsla ?? true,       // WSLA by default
    │      onEpochEnd: opts.onProgress,
    │  })  →  losses: number[]
    └─ return { losses, epochCount, durationMs }
```

### 5.6 `save()` / `load()` Call Flow

```
session.save(opts)
    │
    ├─ model.exportWeights()  →  buffer: ArrayBuffer
    ├─ [switch on opts.storage]
    │    ├─ 'indexedDB'  →  openDB + put(key, buffer)
    │    ├─ 'download'   →  Blob + URL.createObjectURL + <a>.click()
    │    └─ 'fileSystem' →  window.showSaveFilePicker + writable.write
    └─ [done]

session.load(opts)
    │
    ├─ [switch on opts.storage]
    │    ├─ 'indexedDB'  →  openDB + get(key)  →  buffer | undefined
    │    ├─ 'url'        →  fetch(opts.url) + arrayBuffer()
    │    └─ 'fileSystem' →  window.showOpenFilePicker + read
    ├─ [if buffer found]
    │    └─ model.loadWeights(buffer)
    └─ return (buffer !== undefined)
```

---

## 6. Opinionated Defaults

A key design principle of MambaKit is that **zero configuration should produce a working result**. The following defaults are baked in:

### Model Size Presets

| Preset | `dModel` | `numLayers` | `dState` | `dConv` | `expand` | Approx params |
|---|---|---|---|---|---|---|
| `nano` | 128 | 4 | 16 | 4 | 2 | ~6 M |
| `small` | 256 | 6 | 16 | 4 | 2 | ~20 M |
| `medium` *(default)* | 512 | 8 | 16 | 4 | 2 | ~50 M |
| `large` | 768 | 12 | 16 | 4 | 2 | ~120 M |

If `modelSize` is omitted, `'medium'` is used.

### Generation Defaults

| Option | Default |
|---|---|
| `maxNewTokens` | `200` |
| `temperature` | `0.8` |
| `topK` | `50` |
| `topP` | `0.9` |

### Training / Adapt Defaults

| Option | Default | Rationale |
|---|---|---|
| `wsla` | `true` | Fast adaptation with minimal compute — best default for most use cases |
| `epochs` | `3` | Adequate for WSLA; not so many that it overfits a small snippet |
| `learningRate` | `1e-4` | Standard conservative default |
| `seqLen` | `512` | Balances context length and VRAM usage |

### Persistence Defaults

| Option | Default |
|---|---|
| `storage` | `'indexedDB'` |
| `key` / `name` | `'default'` |
| `filename` (download / fileSystem) | `'<sessionName>.bin'` |

### WebGPU Defaults

| Option | Default |
|---|---|
| `powerPreference` | `'high-performance'` |
| `fetchRetries` | `2` |

---

## 7. Integration Scenarios

### 7.1 Scenario — Minimal Code-Completion Widget

The most common integration: a `<textarea>` that suggests completions as the user types.

```ts
import { MambaSession } from 'mambacode.js/kit';

let session: MambaSession | null = null;

async function init() {
  session = await MambaSession.create({
    checkpointUrl : '/models/mamba-coder-base.bin',
    vocabUrl      : '/vocab.json',
    mergesUrl     : '/merges.txt',
  });
}

async function onKeyUp(e: KeyboardEvent) {
  if (!session) return;
  const prompt     = (e.target as HTMLTextAreaElement).value;
  const suggestion = await session.complete(prompt, { maxNewTokens: 50 });
  showGhostText(suggestion);
}
```

**Before MambaKit:** 8 setup steps + manual encode/decode on every keypress.
**After MambaKit:** `create()` + `complete()`.

---

### 7.2 Scenario — VSCode Extension (Webview)

VSCode extensions can host a browser Webview. MambaKit runs inside the Webview, keeping all code local to the developer's machine.

```ts
// webview/main.ts
import { MambaSession } from 'mambacode.js/kit';

const session = await MambaSession.create({
  checkpointUrl : panel.webview.asWebviewUri(
    vscode.Uri.joinPath(context.extensionUri, 'models', 'checkpoint.bin')
  ).toString(),
  vocabUrl  : panel.webview.asWebviewUri(
    vscode.Uri.joinPath(context.extensionUri, 'assets', 'vocab.json')
  ).toString(),
  mergesUrl : panel.webview.asWebviewUri(
    vscode.Uri.joinPath(context.extensionUri, 'assets', 'merges.txt')
  ).toString(),
  name: 'vscode-session',
});

// Restore any previously saved fine-tuned weights
await session.load();

// Handle completion requests from the extension host
window.addEventListener('message', async ({ data }) => {
  if (data.type === 'complete') {
    const completion = await session.complete(data.prompt);
    vscode.postMessage({ type: 'completion', text: completion });
  }
});
```

**Key advantage:** `session.load()` automatically restores the IndexedDB checkpoint from the previous VS Code session, so the model remembers the user's codebase fine-tuning across restarts.

---

### 7.3 Scenario — Fine-Tune on User's Codebase (builderforce.ai-style)

A drag-and-drop UI where the user drops their project files and the model adapts to their coding style — all in the browser.

```ts
import { MambaSession } from 'mambacode.js/kit';

let session: MambaSession;

document.getElementById('start-btn')!.addEventListener('click', async () => {
  session = await MambaSession.create({
    checkpointUrl : '/models/mamba-coder-base.bin',
    vocabUrl      : '/vocab.json',
    mergesUrl     : '/merges.txt',
  });
});

document.getElementById('drop-zone')!.addEventListener('drop', async (e) => {
  const files   = Array.from(e.dataTransfer!.files);
  const codeStr = (await Promise.all(files.map(f => f.text()))).join('\n\n');

  const result = await session.adapt(codeStr, {
    epochs     : 3,
    onProgress : (epoch, loss) => {
      document.getElementById('status')!.textContent =
        `Epoch ${epoch} — loss ${loss.toFixed(4)}`;
    },
  });

  await session.save();   // persists fine-tuned weights to IndexedDB
  document.getElementById('status')!.textContent =
    `Done! Final loss: ${result.losses.at(-1)!.toFixed(4)}`;
});
```

---

### 7.4 Scenario — Streaming Code Generation UI

Real-time token-by-token rendering for a chat or code editor interface.

```ts
import { MambaSession } from 'mambacode.js/kit';

const session = await MambaSession.create({ ... });
const output  = document.getElementById('output')!;

document.getElementById('generate-btn')!.addEventListener('click', async () => {
  output.textContent = '';
  const prompt = (document.getElementById('prompt') as HTMLInputElement).value;

  for await (const token of session.completeStream(prompt)) {
    output.textContent += token;
  }
});
```

---

### 7.5 Scenario — Team Checkpoint Sharing

Download a fine-tuned checkpoint and share it with colleagues.

```ts
// Team member A — fine-tune and share
const session = await MambaSession.create({ checkpointUrl: '/base.bin', ... });
await session.adapt(teamCodebase);
await session.save({ storage: 'download', filename: 'team-v1.bin' });

// Team member B — load the shared checkpoint
const session = await MambaSession.create({
  checkpointUrl : 'https://your-file-server.com/team-v1.bin',
  vocabUrl      : '/vocab.json',
  mergesUrl     : '/merges.txt',
});
const completion = await session.complete('const apiClient = new');
```

---

### 7.6 Scenario — Custom Model Configuration (advanced user)

A researcher who wants fine-grained control but still benefits from the simplified API:

```ts
const session = await MambaSession.create({
  vocabUrl    : '/vocab.json',
  mergesUrl   : '/merges.txt',
  modelSize   : 'custom',
  modelConfig : {
    dModel    : 768,
    numLayers : 16,
    dState    : 32,
    expand    : 4,
  },
});

// Use the escape hatch to run a custom WGSL kernel
const { device, model } = session.internals;
```

---

### 7.7 Scenario — Loading Bar During Initialisation

```ts
import { MambaSession, type CreateProgressEvent } from 'mambacode.js/kit';

const progressBar = document.getElementById('progress') as HTMLProgressElement;

const session = await MambaSession.create(
  {
    checkpointUrl : '/models/mamba-coder-base.bin',
    vocabUrl      : '/vocab.json',
    mergesUrl     : '/merges.txt',
  },
  {
    onProgress: (event: CreateProgressEvent) => {
      progressBar.value   = event.progress * 100;
      progressBar.title   = event.message;
    },
  }
);
```

---

### 7.8 Scenario — Quality Monitoring Over Multiple Adapt Cycles

```ts
const session = await MambaSession.create({ ... });

const snapshots: { cycle: number; perplexity: number }[] = [];

for (let cycle = 0; cycle < 5; cycle++) {
  const newCode = await fetchLatestCommitDiff();  // hypothetical
  await session.adapt(newCode);

  const ppl = await session.evaluate(heldOutTestSuite);
  snapshots.push({ cycle, perplexity: ppl });
  console.log(`Cycle ${cycle}: perplexity = ${ppl.toFixed(2)}`);

  await session.save({ key: `checkpoint-cycle-${cycle}` });
}
```

---

## 8. Configuration Reference

### `MambaSessionOptions` — Full Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `checkpointUrl` | `string` | — | URL to a `.bin` weight file. Optional — random weights if omitted |
| `vocabUrl` | `string` | — | URL to `vocab.json`. Required unless `vocabObject` supplied |
| `mergesUrl` | `string` | — | URL to `merges.txt`. Required unless `mergesArray` supplied |
| `vocabObject` | `Record<string,number>` | — | In-memory vocab. Alternative to `vocabUrl` |
| `mergesArray` | `string[]` | — | In-memory merges. Alternative to `mergesUrl` |
| `name` | `string` | `'default'` | Session name — used as IndexedDB key and default filename |
| `modelSize` | `'nano'\|'small'\|'medium'\|'large'\|'custom'` | `'medium'` | Preset model size |
| `modelConfig` | `Partial<MambaModelConfig>` | — | Fine-grained overrides (only when `modelSize='custom'`) |
| `powerPreference` | `'high-performance'\|'low-power'` | `'high-performance'` | WebGPU adapter preference |
| `fetchRetries` | `number` | `2` | Retry count for checkpoint fetch failures |

### `CompleteOptions` — Full Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `maxNewTokens` | `number` | `200` | Maximum tokens to generate |
| `temperature` | `number` | `0.8` | Sampling temperature (higher = more random) |
| `topK` | `number` | `50` | Top-K token filtering |
| `topP` | `number` | `0.9` | Nucleus sampling probability mass |

### `AdaptOptions` — Full Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `epochs` | `number` | `3` | Training epochs |
| `learningRate` | `number` | `1e-4` | AdamW learning rate |
| `seqLen` | `number` | `512` | Sequence chunk length |
| `wsla` | `boolean` | `true` | WSLA fast-adapt (B & C matrices only) |
| `fullTrain` | `boolean` | `false` | Convenience alias: sets `wsla=false`, `epochs=5` |
| `onProgress` | `(epoch, loss) => void` | — | Per-epoch callback |

### `SaveOptions` / `LoadOptions` — Full Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `storage` | `StorageTarget` | `'indexedDB'` | Where to save/load weights |
| `filename` | `string` | `'<name>.bin'` | Filename for `download`/`fileSystem` targets |
| `key` | `string` | session `name` | IndexedDB key override |
| `url` | `string` | — | Remote URL for `load()` with `storage:'url'` |

---

## 9. Error Handling Strategy

### 9.1 `MambaKitError` Class

All MambaKit errors are instances of `MambaKitError`, which extends `Error` and carries a typed `code` discriminant:

```ts
export class MambaKitError extends Error {
  constructor(
    public readonly code: MambaKitErrorCode,
    message: string,
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = 'MambaKitError';
  }
}

export type MambaKitErrorCode =
  | 'GPU_UNAVAILABLE'        // navigator.gpu not present or adapter request failed
  | 'TOKENIZER_LOAD_FAILED'  // vocab.json or merges.txt could not be fetched/parsed
  | 'CHECKPOINT_FETCH_FAILED'// checkpoint URL returned non-OK response after retries
  | 'CHECKPOINT_INVALID'     // loadWeights threw (bad magic, version, or size mismatch)
  | 'INPUT_TOO_SHORT'        // adapt() input encodes to fewer than 2 tokens
  | 'STORAGE_UNAVAILABLE'    // IndexedDB or File System Access API not available
  | 'SESSION_DESTROYED'      // method called after destroy()
  | 'UNKNOWN';               // unexpected error (original in .cause)
```

### 9.2 Retry Logic

Checkpoint fetches are retried up to `fetchRetries` times (default 2) with exponential back-off (500 ms, 1000 ms). If all retries fail, a `MambaKitError` with code `CHECKPOINT_FETCH_FAILED` is thrown.

### 9.3 Graceful Degradation

- If the IndexedDB write fails in `save()`, the error is surfaced (not silently swallowed) so the consumer can offer a fallback download.
- If `load()` finds no saved checkpoint it returns `false` rather than throwing. The consumer decides what to do (show a first-run prompt, fall back to base weights, etc.).

---

## 10. Persistence Strategy

### 10.1 IndexedDB Schema

MambaKit uses a single IndexedDB database named `'mambakit'` with one object store named `'checkpoints'`. Keys are the session `name` string. Values are raw `ArrayBuffer` weight data.

```
Database: mambakit
  Object store: checkpoints
    Key:   'default'            →  ArrayBuffer (weights)
    Key:   'team-v1'            →  ArrayBuffer (weights)
    Key:   'checkpoint-cycle-3' →  ArrayBuffer (weights)
```

### 10.2 Storage Size

A `medium`-preset model (~50 M float32 parameters) occupies approximately **200 MB** on disk. Browsers typically allow 10–20 % of available disk space for IndexedDB. If quota is exceeded, `save()` throws `MambaKitError('STORAGE_UNAVAILABLE', ...)` and suggests switching to `'download'`.

---

## 11. Event & Progress Model

All progress reporting uses plain callbacks rather than `EventTarget` to keep the implementation simple and avoid DOM coupling:

```ts
// During create()
onProgress?: (event: CreateProgressEvent) => void

// During adapt()
onProgress?: (epoch: number, loss: number) => void
```

No global event bus, no `addEventListener`. Each callback is scoped to the operation that triggers it.

---

## 12. Performance Considerations

### 12.1 `complete()` Latency

The facade adds negligible overhead to `model.generate()`. The `tokenizer.encode()` and `tokenizer.decode()` calls are CPU-bound and typically complete in < 1 ms for prompts under 512 tokens. The dominant cost remains the GPU forward passes.

### 12.2 Repeated `create()` Calls

`MambaSession.create()` acquires a `GPUDevice`. Creating multiple sessions simultaneously on the same GPU is wasteful. The consuming application should create one session and reuse it. The facade does not enforce a singleton, but the documentation should strongly recommend it.

### 12.3 WSLA as Default

The default `wsla: true` in `adapt()` means only ~2 % of parameters are updated per step, dramatically reducing GPU memory bandwidth and compute time. For a `medium`-preset model this cuts per-epoch training time from ~30 s to ~2 s on a typical integrated GPU. Users who want full fine-tuning pass `fullTrain: true`.

### 12.4 IndexedDB vs. GPU Memory

`save()` calls `model.exportWeights()` which reads all GPU buffers back to CPU. This is an intentional one-time cost at save time. During a normal session (no save/load) weights remain on the GPU throughout.

---

## 13. Migration Path from Low-Level API

Existing code using the MambaCode.js low-level API does not need to change. MambaKit is additive. A team can migrate incrementally:

**Phase 1 — New features use MambaKit**

Write all new integration code using `MambaSession`. Existing low-level code continues to work.

**Phase 2 — Migrate existing setup code**

Replace the 8-step setup sequence with `MambaSession.create()`. Access the same `device`, `model`, and `trainer` objects through `session.internals` if the low-level API is still needed.

**Phase 3 — Remove escape hatch references** (optional)

Once all use cases are covered by MambaKit methods, remove references to `session.internals`.

### Side-by-side comparison

| Low-level API | MambaKit equivalent |
|---|---|
| `initWebGPU()` | Handled inside `MambaSession.create()` |
| `new BPETokenizer()` + `load()` | Handled inside `MambaSession.create()` |
| `new MambaModel(device, config)` | Handled inside `MambaSession.create()` |
| `fetch(url)` + `model.loadWeights()` | `checkpointUrl` option in `MambaSession.create()` |
| `new MambaTrainer(model, tokenizer)` | Handled inside `MambaSession.create()` |
| `tokenizer.encode()` + `model.generate()` + `tokenizer.decode()` | `session.complete(prompt)` |
| `trainer.train(code, opts)` | `session.adapt(code, opts)` |
| `trainer.evaluate(code)` | `session.evaluate(code)` |
| `model.exportWeights()` + IndexedDB / Blob URL | `session.save()` |
| `model.loadWeights(buffer)` from IndexedDB | `session.load()` |
| `model.parameters()`, custom GPU pipelines | `session.internals.model`, `session.internals.device` |

---

## 14. Implementation Plan

This section is a step-by-step plan for the implementing agent.

### Step 1 — Scaffold `src/kit/` directory

Create the following empty files:

```
src/kit/index.ts
src/kit/session.ts
src/kit/presets.ts
src/kit/persistence.ts
src/kit/streaming.ts
src/kit/errors.ts
```

### Step 2 — Implement `errors.ts`

Define `MambaKitErrorCode` union type and `MambaKitError` class as specified in [Section 9.1](#91-mambakit-error-class).

### Step 3 — Implement `presets.ts`

Export `MODEL_PRESETS` map from preset name to `Partial<MambaModelConfig>`:

```ts
export const MODEL_PRESETS: Record<string, Partial<MambaModelConfig>> = {
  nano   : { dModel: 128, numLayers:  4, dState: 16, dConv: 4, expand: 2 },
  small  : { dModel: 256, numLayers:  6, dState: 16, dConv: 4, expand: 2 },
  medium : { dModel: 512, numLayers:  8, dState: 16, dConv: 4, expand: 2 },
  large  : { dModel: 768, numLayers: 12, dState: 16, dConv: 4, expand: 2 },
};
```

Export a `resolveModelConfig(options: MambaSessionOptions, vocabSize: number): Required<MambaModelConfig>` function that:

1. Selects the preset (or falls back to `'medium'`)
2. Merges any `modelConfig` overrides (only when `modelSize === 'custom'`)
3. Sets `vocabSize` to the tokenizer's `vocabSize`
4. Returns a `Required<MambaModelConfig>` with all defaults filled

### Step 4 — Implement `persistence.ts`

Export three functions:

```ts
export async function saveToIndexedDB(key: string, buffer: ArrayBuffer): Promise<void>
export async function loadFromIndexedDB(key: string): Promise<ArrayBuffer | undefined>
export async function triggerDownload(filename: string, buffer: ArrayBuffer): Promise<void>
export async function saveViaFileSystemAPI(filename: string, buffer: ArrayBuffer): Promise<void>
export async function loadViaFileSystemAPI(): Promise<ArrayBuffer>
```

The IndexedDB functions open (or reuse) the `'mambakit'` database at version 1 with a `'checkpoints'` object store. They must work without any external IndexedDB library — use the raw browser `indexedDB` global.

### Step 5 — Implement `streaming.ts`

Export an `async function* tokenStream(...)` generator that wraps `model.forward()` to yield token IDs one at a time, suitable for use in `completeStream()`.

Internally, this adapts the current `model.generate()` loop to `yield` each token immediately after sampling rather than accumulating into an array.

### Step 6 — Implement `session.ts`

Implement the full `MambaSession` class according to [Section 4](#4-proposed-api-design):

- `static async create(options, progressOptions?): Promise<MambaSession>`
- `async complete(prompt, opts?): Promise<string>`
- `async *completeStream(prompt, opts?): AsyncIterable<string>`
- `async adapt(text, opts?): Promise<AdaptResult>`
- `async evaluate(text): Promise<number>`
- `async save(opts?): Promise<void>`
- `async load(opts?): Promise<boolean>`
- `destroy(): void`
- `readonly internals: SessionInternals`

Guards:
- All instance methods must throw `MambaKitError('SESSION_DESTROYED', ...)` if called after `destroy()`.
- The `create()` factory must wrap every step in try/catch and re-throw as the appropriate `MambaKitError` code.

### Step 7 — Implement `index.ts`

Re-export all public symbols:

```ts
export { MambaSession }      from './session';
export { MambaKitError }     from './errors';
export type { MambaKitErrorCode } from './errors';
export type {
  MambaSessionOptions,
  CompleteOptions,
  AdaptOptions,
  AdaptResult,
  SaveOptions,
  LoadOptions,
  StorageTarget,
  CreateProgressEvent,
  CreateStage,
  SessionInternals,
} from './session';
```

### Step 8 — Update `package.json` exports

Add the `./kit` sub-path export:

```json
"exports": {
  ".": {
    "import": "./dist/index.js",
    "types":  "./dist/index.d.ts"
  },
  "./kit": {
    "import": "./dist/kit/index.js",
    "types":  "./dist/kit/index.d.ts"
  }
}
```

### Step 9 — Update `tsconfig.json` if needed

Ensure the TypeScript compiler includes `src/kit/**/*.ts`. The existing `"include": ["src/**/*"]` pattern should cover this automatically.

### Step 10 — Write unit tests

Create `tests/kit.test.ts` with the following test cases (no GPU required — mock the low-level API):

| Test | Description |
|---|---|
| `resolveModelConfig: medium preset fills all fields` | Verify all `Required<MambaModelConfig>` fields are set |
| `resolveModelConfig: custom overrides respected` | Verify `modelConfig` fields override preset values |
| `MambaKitError has correct code` | Verify `code` discriminant is set correctly |
| `MambaKitError extends Error` | Verify `instanceof Error` |
| `saveToIndexedDB + loadFromIndexedDB round-trip` | Mock `indexedDB`; verify buffer identity |
| `triggerDownload does not throw` | Mock `URL.createObjectURL`; verify no error |
| `session.internals exposes model, trainer, tokenizer, device` | Verify all four properties present after create |
| `session methods throw SESSION_DESTROYED after destroy()` | Verify `complete`, `adapt`, `evaluate`, `save`, `load` all throw |

Mock strategy: create a `__mocks__` directory or inline Jest mocks for `initWebGPU`, `MambaModel`, `MambaTrainer`, and `BPETokenizer` so tests run without a GPU.

### Step 11 — Update documentation

1. Add a new section to `README.md` under the existing Quick Start section:
   - Heading: `## Quick Start — MambaKit (Simplified API)`
   - Show the 3-line create + complete example
   - Link to the full PRD document for details

2. Update the docs table in `README.md` to include this PRD as a new row:
   - `**[MambaKit PRD](docs/mamba-kit-prd.md)**` — Simplified facade design document

### Step 12 — Build and verify

```bash
npm run build   # must compile without errors
npm test        # all existing tests + new kit tests must pass
npm run lint    # must produce zero new warnings
```

---

## 15. File Structure

After implementation, the repository structure gains:

```
src/
├── index.ts                    ← unchanged
├── kit/
│   ├── index.ts                ← MambaKit public entry point
│   ├── session.ts              ← MambaSession class (core facade)
│   ├── presets.ts              ← model size presets + config resolver
│   ├── persistence.ts          ← IndexedDB / download / File System helpers
│   ├── streaming.ts            ← AsyncIterable token streaming adapter
│   └── errors.ts               ← MambaKitError class
├── model/ ...                  ← unchanged
├── training/ ...               ← unchanged
├── tokenizer/ ...              ← unchanged
├── utils/ ...                  ← unchanged
└── kernels/ ...                ← unchanged

tests/
├── kit.test.ts                 ← NEW: MambaKit unit tests
├── autograd.test.ts            ← unchanged
├── bpe.test.ts                 ← unchanged
├── kernels.test.ts             ← unchanged
└── quantization.test.ts        ← unchanged

docs/
├── mamba-kit-prd.md            ← THIS DOCUMENT
├── getting-started.md          ← unchanged
├── api-reference.md            ← unchanged
├── integration-architecture.md ← unchanged
└── weight-lifecycle.md         ← unchanged
```

---

## 16. Testing Requirements

### Unit Tests (no GPU required)

All tests in `tests/kit.test.ts` must run in Node.js using Jest without a real `GPUDevice`. The implementing agent must mock or stub:

- `initWebGPU` → returns `{ device: mockDevice, adapter: mockAdapter }`
- `MambaModel` constructor → returns a mock object with `forward`, `generate`, `loadWeights`, `exportWeights`, `parameters`, `setWSLAMode`
- `MambaTrainer` constructor → returns a mock object with `train`, `evaluate`
- `BPETokenizer` → returns a mock object with `load`, `loadFromObjects`, `encode`, `decode`, `vocabSize`
- Browser globals (`indexedDB`, `URL.createObjectURL`, `Blob`, `document.createElement`) → minimal stubs or Jest fake environment

### Integration Tests (browser required)

GPU-dependent end-to-end tests are exercised manually in a browser with WebGPU support. The following scenarios should be verified manually:

1. `MambaSession.create()` completes without error in Chrome 113+
2. `session.complete()` returns a non-empty string
3. `session.adapt()` runs 3 epochs and returns a decreasing loss
4. `session.save()` persists to IndexedDB; `session.load()` restores it
5. `session.completeStream()` yields tokens one at a time

### Regression Tests

All 58 existing unit tests must continue to pass unchanged after the implementation is complete. MambaKit must not modify any file outside `src/kit/`, `tests/kit.test.ts`, `package.json`, and this PRD document.

---

*Back to [README](../README.md) · [API Reference](./api-reference.md) · [Getting Started](./getting-started.md) · [Integration & Architecture](./integration-architecture.md)*
