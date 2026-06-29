# @seanhogg/builderforce-memory

> **JavaScript-native AI runtime** — SSM execution + Transformer orchestration + online distillation + persistent agent memory.

[![npm](https://img.shields.io/npm/v/@seanhogg/builderforce-memory)](https://www.npmjs.com/package/@seanhogg/builderforce-memory)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

`@seanhogg/builderforce-memory` is the **runtime layer** of BuilderForce Agent Memory — a complete, self-contained AI runtime built directly on top of [`@seanhogg/builderforce-memory-engine`](https://www.npmjs.com/package/@seanhogg/builderforce-memory-engine). It includes the full session layer (previously `@seanhogg/mambakit`) as an internal layer, so you only need one package. *(Formerly published as `@seanhogg/ssmjs`.)*

> **Technical report & peer review.** The runtime's design — Write-Through Cognition, hybrid recall, the inference router, and online distillation — is specified in the Evermind technical report ([`../../publication/evermind/`](../../publication/evermind)). Its adversarial [`PEER-REVIEW.md`](../../publication/evermind/PEER-REVIEW.md) is candid about runtime-level gaps to weigh before production: recall is an `O(N)` scan with no ANN index and frequently uses the lexical (Jaccard) fallback rather than dense embeddings; the "stable subject key" is **caller-supplied and not yet canonicalized** (so the single-incumbent guarantee rests on an unenforced precondition); recalled facts are injected unsanitized (memory-poisoning surface); and the online loop has no catastrophic-forgetting guard. Tracked as `EVM-1…EVM-8`. These have since been resolved in the v2026.6.34 hardening — an HNSW ANN index (`src/retrieval/hnsw.ts`) gates the exact scan, a subject-key canonicalizer (`src/cognition/canonicalize.ts`) runs inside `commit()`, recall is fenced, and the online loop has a forgetting guard — and the engine now ships a benchmarking harness for measured quality. See the resolution addendum in [`PEER-REVIEW.md`](../../publication/evermind/PEER-REVIEW.md).

---

## Overview

`@seanhogg/builderforce-memory` is a JavaScript-native AI runtime that combines local SSM (State Space Model) inference with optional transformer bridge escalation, persistent semantic memory, and online distillation — all without leaving the browser or Node.js process.

The layered stack:

```
@seanhogg/builderforce-memory-engine  →  WebGPU kernels (WGSL, Mamba-1/2/3 SSM math)
@seanhogg/builderforce-memory        →  Session layer + Runtime orchestration (this package)
                 ├── src/session/   MambaSession, tokenizer, persistence
                 ├── src/runtime/   SSMRuntime, routing
                 ├── src/memory/    MemoryStore
                 ├── src/agent/     SSMAgent
                 └── src/distillation/  DistillationEngine
```

| Capability                   | @seanhogg/builderforce-memory |
|------------------------------|--------|
| Simple session API           | ✅     |
| WebGPU execution             | ✅     |
| SSM variants (1/2/3/hybrid)  | ✅     |
| Transformer bridge           | ✅     |
| Intelligent routing          | ✅     |
| Online distillation          | ✅     |
| Persistent semantic memory   | ✅     |
| Agent workflows              | ✅     |
| Create-an-LLM pipeline + diagnostics | ✅ |

---

## Create an LLM (workflow + diagnostics)

The `TRAIN_LLM` workflow trains a custom `EvermindLM` on **any** corpus and packages
it as a portable `.evermind` artifact — domain-agnostic (resume tailoring, support
replies, code, prose). Every run is gated by built-in diagnostics so a broken model
never ships:

```ts
import { runWorkflow, TRAIN_LLM } from '@seanhogg/builderforce-memory';

const r = await runWorkflow(TRAIN_LLM); // or cloneTemplate('train-llm', { steps: [...] })
const artifact = r.artifacts.evermind;  // ArrayBuffer → ship + serve
```

Generic build step types (compose your own pipeline via the step registry):

| Step | What it validates |
|------|-------------------|
| `train-tokenizer` | Learns a BPE vocab from your corpus |
| `dataset-quality` | Gates corpus size / sequence count / duplicate ratio **before** spending epochs |
| `train-model` | Trains the `EvermindLM` (exposes the loss curve on `ctx.bag.trainingHistory`) |
| `convergence` | Asserts training loss actually decreased (the model learned) |
| `evaluate` / `generate-check` | Output is non-empty **and** seed-reproducible |
| `roundtrip` | Packages the trained model → loads → generates; served output must match trained |
| `benchmark` | Scores the trained model on a held-out corpus (perplexity, bits-per-token, top-1/top-k accuracy, throughput); optional maxPerplexity / minTop1 gates fail the run |

---

## Installation

```bash
npm install @seanhogg/builderforce-memory
# or
pnpm add @seanhogg/builderforce-memory
```

`@seanhogg/builderforce-memory` includes the full session layer (previously `@seanhogg/mambakit`).
`@seanhogg/builderforce-memory-engine` is a peer dependency — install it alongside:

```bash
npm install @seanhogg/builderforce-memory @seanhogg/builderforce-memory-engine
```

### Node.js requirements

Node.js 18+ is required. Two additional shims are needed for Node.js:

```bash
npm install @webgpu/node fake-indexeddb
```

- `@webgpu/node` — Dawn-based WebGPU for Node.js; drives all WGSL compute kernels
- `fake-indexeddb` — in-memory IndexedDB compatible with the IDB spec; used by `MemoryStore`

---

## Quick Start

### Browser

```ts
import { SSM, AnthropicBridge, SSMAgent, MemoryStore } from '@seanhogg/builderforce-memory';

const runtime = await SSM.create({
  session: { modelSize: 'small', mambaVersion: 'mamba2' },
  bridge : new AnthropicBridge({ apiKey: 'sk-ant-...' }),
});

// Generate — routes to SSM or transformer automatically
const answer = await runtime.generate('What is a state space model?');

// Streaming — always SSM for low-latency output
for await (const token of runtime.stream('function fibonacci(')) {
  process.stdout.write(token);
}

// Fine-tune on your content
await runtime.adapt(myCodebase);
runtime.destroy();
```

### Node.js

```ts
import { create as createGPU } from '@webgpu/node';
import { IDBFactory }          from 'fake-indexeddb';
import { SSM, MemoryStore, SSMAgent } from '@seanhogg/builderforce-memory';

const gpuAdapter = await createGPU().requestAdapter({ powerPreference: 'high-performance' });
const idbFactory = new IDBFactory();

const runtime = await SSM.create({
  session: {
    gpuAdapter,
    idbFactory,
    modelSize: 'small',
  },
});

const memory = new MemoryStore({ idbFactory });
const agent  = new SSMAgent({ runtime, memory });
await agent.init();   // loads persisted history if present

const reply = await agent.think('Explain this codebase');
console.log(reply);

await agent.destroy(); // persists history, releases GPU
```

---

## Custom Tokenizers

By default, `MambaSession` uses the built-in Qwen2.5-Coder BPE tokenizer.  You can override this by passing any object that satisfies the `Tokenizer` interface:

```ts
import type { Tokenizer } from '@seanhogg/builderforce-memory';

const myTokenizer: Tokenizer = {
  encode(text: string): number[]  { /* your encode implementation */ return []; },
  decode(tokens: number[]): string { /* your decode implementation */ return ''; },
  get vocabSize(): number          { return 32000; },
};

const runtime = await SSM.create({
  session: {
    tokenizer: myTokenizer,   // replaces BPETokenizer entirely
    modelSize: 'small',
  },
});
```

Use cases:
- **HuggingFace Transformers.js** tokenizer — wrap its `encode`/`decode` in the interface
- **Unit testing** — a stub tokenizer that maps words to sequential IDs, no network needed
- **Domain-specific vocabularies** — medical, legal, multilingual tokenizers

---

## Memory System

`MemoryStore` is a persistent, TTL-aware, tagged key-value fact store backed by IndexedDB.

### Basic usage

```ts
import { MemoryStore } from '@seanhogg/builderforce-memory';

const memory = new MemoryStore({
  dbName      : 'my-app',
  defaultTtlMs: 7 * 24 * 60 * 60 * 1000,  // 7-day default TTL
  idbFactory,  // Node.js only
});

// Store facts
await memory.remember('author', 'Sean Hogg');
await memory.remember('stack', 'React + TypeScript', {
  tags      : ['tech', 'project'],
  importance: 0.8,
  ttlMs     : 30 * 24 * 60 * 60 * 1000,  // 30 days
});

// Retrieve
const entry = await memory.recall('author');

// All non-expired facts, newest first
const all = await memory.recallAll();

// N most recent non-expired facts
const recent = await memory.recallRecent(10);

// Filter by tag
const techFacts = await memory.recallByTag('tech');

// Semantic similarity search (SSM-embedding cosine via HNSW ANN index, lexical fallback)
const similar = await memory.recallSimilar('who built this?', 5, runtime);

// Purge expired entries from storage
const deletedCount = await memory.purgeExpired();
```

### Cross-session memory merge

```ts
// Export all non-expired facts from sessionA
const exported = await memoryA.exportAll();

// Import into sessionB
await memoryB.importAll(exported, 'merge');
// 'merge'     — only overwrites if incoming entry is newer
// 'overwrite' — writes all entries unconditionally
```

### Weight persistence

```ts
await memory.saveWeights(runtime);      // saves model weights to IndexedDB
const loaded = await memory.loadWeights(runtime);  // false if no checkpoint found
```

### MemoryEntry schema

```ts
interface MemoryEntry {
  key        : string;
  content    : string;
  timestamp  : number;
  ttlMs?     : number;       // optional TTL; entry filtered after timestamp + ttlMs
  type?      : FactType;     // 'text' | 'json' | 'number' | 'boolean'
  tags?      : string[];     // for grouping/filtering
  importance?: number;       // 0–1, default 0.5; higher facts appear first in prompts
}
```

---

## Inference Routing

`InferenceRouter` decides whether each request goes to the local SSM or the transformer bridge. It is built into `SSMRuntime` — you don't need to instantiate it directly.

### Routing strategies

```ts
const runtime = await SSM.create({
  session: { modelSize: 'nano' },
  bridge : claude,
  routingStrategy   : 'auto',   // 'auto' | 'ssm' | 'transformer'
  longInputThreshold: 1200,     // chars before preferring transformer (default: 1200)
  perplexityThreshold: 80,      // SSM perplexity cutoff (default: 80)
});
```

**Auto-routing heuristics (cheapest first):**
1. **Complexity patterns** — "step by step", "analyze", "compare and contrast" → transformer
2. **Input length** — over threshold → transformer
3. **SSM perplexity** — async probe; high perplexity = novel topic → transformer

### RoutingDecision type

`route()` now returns a structured `RoutingDecision` object:

```ts
interface RoutingDecision {
  target    : 'ssm' | 'transformer';
  reason    : 'strategy' | 'complexity' | 'length' | 'perplexity' | 'no_bridge';
  confidence: number;    // 0–1
  details?  : string;    // human-readable explanation
}
```

### Routing audit log

Every routing decision is appended to an in-memory audit log (last 500 entries):

```ts
const log = runtime.getRoutingAuditLog();
// log: RoutingAuditEntry[]
// { timestamp, inputLength, decision: RoutingDecision, durationMs }
```

---

## Distillation

Teach the local SSM using a transformer teacher — runs entirely in the browser or Node.js.

```ts
import { DistillationEngine } from '@seanhogg/builderforce-memory';

const distiller = new DistillationEngine(runtime, claude);

// Single pass: claude generates → SSM adapts on output
const result = await distiller.distill('Explain WebGPU compute shaders', {
  adapt      : { wsla: true, epochs: 3 },
  qualityGate: {
    minLength    : 50,    // skip if teacher output < 50 chars
    maxPerplexity: 15,    // skip if SSM perplexity already < 15 (already learned)
  },
});

console.log('skipped:', result.skipped, result.skipReason);
console.log('loss:',    result.adaptResult.losses.at(-1));

// Batch distillation
const batch = await distiller.distillBatch([
  'What is a Mamba block?',
  'Explain WSLA adaptation.',
], { adapt: { wsla: true, epochs: 5 } });

console.log(`${batch.totalEpochs} epochs in ${batch.totalMs}ms`);
```

### Quality gates

| Gate option      | Description |
|------------------|-------------|
| `minLength`      | Skip if teacher output is shorter than N characters (low-quality response) |
| `maxPerplexity`  | Skip if SSM perplexity on teacher output is already below threshold (already learned) |

### Distillation log

```ts
const log = distiller.getLog();
// log: DistillationLog[]
// { timestamp, input, teacherOutputLength, skipped, skipReason?, finalLoss?, epochs }
```

The log is bounded to the last 200 entries.

---

## SSMAgent

High-level orchestration: conversation history, routing, memory injection, and lifecycle.

```ts
import { SSMAgent, MemoryStore } from '@seanhogg/builderforce-memory';

const memory = new MemoryStore();
const agent  = new SSMAgent({
  runtime        : runtime,
  memory,
  systemPrompt   : 'You are a senior TypeScript engineer.',
  maxHistoryTurns: 20,
  persistHistory : true,  // saves/loads history via memory on destroy/init
});

// Load persisted history from a prior session
await agent.init();

// Store project context
await agent.remember('stack', 'React 18, TypeScript 5, Vite');

// Multi-turn conversation — facts with highest importance appear first in context
const reply1 = await agent.think('What stack should I use?');
const reply2 = await agent.think('How do I handle concurrent edits?');

// Streaming
for await (const token of agent.thinkStream('Show me a WebSocket hook')) {
  process.stdout.write(token);
}

// Teach the agent from content
await agent.learn(myCodebase, { wsla: true, epochs: 3 });

console.log(agent.turnCount);  // 2

// Persists history to memory, then destroys runtime
await agent.destroy();
```

### History persistence

When `persistHistory: true` (default):
- On `agent.init()`: loads `__history__` from the `MemoryStore` and restores conversation turns.
- On `agent.destroy()`: serialises `_history` to JSON and writes it under `__history__`.

This enables multi-session continuity without external state management.

### Fact injection order

Facts retrieved from `MemoryStore` are sorted by `importance` descending before being injected into the prompt. Higher-importance facts appear first, giving the model the most relevant context regardless of insertion order.

---

## Migration from MambaKit

`@seanhogg/mambakit` has been consolidated into this package. `MambaSession` and all related types are now exported directly from `@seanhogg/builderforce-memory`.

**Before:**

```bash
npm install @seanhogg/mambakit @seanhogg/builderforce-memory
```

```ts
import { MambaSession } from '@seanhogg/mambakit';
import type { MambaSessionOptions, Tokenizer } from '@seanhogg/mambakit';
```

**After:**

```bash
npm install @seanhogg/builderforce-memory
```

```ts
import { MambaSession, SessionError } from '@seanhogg/builderforce-memory';
import type { MambaSessionOptions, Tokenizer } from '@seanhogg/builderforce-memory';
```

All types are re-exported unchanged — `MambaSessionOptions`, `CompleteOptions`, `AdaptOptions`,
`AdaptResult`, `SaveOptions`, `LoadOptions`, `StorageTarget`, `CreateCallbacks`,
`LayerSchedulePreset`, `MODEL_PRESETS`, `GpuMode`, and `Tokenizer`.
No logic changes are required, only the import path.

---

## BuilderForce Agents Integration

@seanhogg/builderforce-memory is the runtime behind **Evermind** in [BuilderForce Agents](https://builderforce.ai)'s gateway. It runs the SSM as a full generative model — not memory-only: a request pinned to a published Evermind model is served here, on-device, instead of being forwarded to an external frontier vendor. It also provides the write-through knowledge memory and recall that ride alongside generation, so a corrected fact stays corrected across sessions. External frontier models (Claude/GPT) remain routable when you want them, but they are no longer the required cortex.

The `SsmMemoryService` class in BuilderForce Agents's `src/infra/ssm-memory-service.ts` wraps an `SSMRuntime` + `SSMAgent` + `MemoryStore` triplet:

```
BuilderForce Agents gateway
├── server-startup.ts       ← initSsmMemoryService() on boot
├── infra/knowledge-loop.ts ← remember() + learn() on every agent run
├── infra/ssm-memory-service.ts  ← SsmMemoryService wrapper
└── builderforce/orchestrator.ts    ← recallSimilar() injected into task prompts
```

**Data flow:**
1. Agent run completes → `KnowledgeLoopService` derives activity summary
2. Summary is stored in `.builderforce/memory/YYYY-MM-DD.md` (markdown log)
3. Summary is also passed to `ssmSvc.remember()` (tagged + importance-weighted)
4. Summary is passed to `ssmSvc.learn()` → WSLA fine-tuning adapts the SSM
5. On next workflow task, `recallSimilar(taskDescription, 5)` injects relevant memories into the prompt as a `[Memory Context]` block

GPU init is optional: if `@webgpu/node` is unavailable, the service starts in memory-only mode (`gpuAvailable: false`) and SSM inference is skipped. The gateway never crashes due to a missing GPU.

---

## Phase Roadmap

### Phase 1 — Foundations
- Session layer: `Tokenizer` interface + pluggable injection via `MambaSessionOptions.tokenizer`
- `MemoryStore`: TTL (`ttlMs`), `defaultTtlMs`, `purgeExpired()`, `recallRecent(n)`
- `MemoryStore`: `FactType`, `tags`, `importance` fields on `MemoryEntry`
- `MemoryStore`: updated `remember()` accepting `RememberOptions`
- `InferenceRouter`: `route()` now returns `RoutingDecision` object with `target`, `reason`, `confidence`, `details`
- `SSMAgent`: `persistHistory` option; `init()` loads `__history__`; `destroy()` saves it
- `SSMAgent`: fact injection sorted by `importance` descending

### Phase 2 — Semantic Memory
- `MemoryStore.recallSimilar(query, topK, runtime)` — SSM-embedding cosine recall via an HNSW ANN index (lexical fallback when embeddings are unavailable)
- `MemoryStore.recallByTag(tag)` — tag-based filtering
- `MemoryStore.exportAll()` / `importAll(entries, strategy)` — cross-session merge

### Phase 3 — BuilderForce Agents Integration
- `SsmMemoryService` in `src/infra/ssm-memory-service.ts` — singleton gateway service
- `server-startup.ts`: `initSsmMemoryService()` on boot; non-fatal GPU fallback
- `KnowledgeLoopService`: `remember()` + `learn()` on every agent run completion
- `AgentOrchestrator`: `recallSimilar()` injected as `[Memory Context]` before task dispatch

### Phase 4 — Feedback Loop
- `DistillationEngine`: `qualityGate` option (`minLength`, `maxPerplexity`)
- `DistillResult`: `skipped` + `skipReason` fields
- `DistillationEngine.getLog()` — bounded in-memory `DistillationLog[]`
- `InferenceRouter`: `RoutingAuditEntry` + `getAuditLog()` — bounded in-memory log
- `SSMRuntime.getRoutingAuditLog()` — delegates to router
- `SSMRuntime.getDistillationLog()` — stub; returns empty array (inline engine future work)

---

## API Reference

### `SSM.create(opts)` / `SSMRuntime.create(opts)`

| Option | Type | Default | Description |
|---|---|---|---|
| `session` | `MambaSessionOptions` | required | Forwarded to `MambaSession.create()` |
| `bridge` | `TransformerBridge` | — | Transformer backend for routing/distillation |
| `routingStrategy` | `'auto'\|'ssm'\|'transformer'` | `'auto'` | Routing strategy |
| `longInputThreshold` | `number` | `1200` | Chars before auto-routing prefers transformer |
| `perplexityThreshold` | `number` | `80` | SSM perplexity cutoff |
| `callbacks` | `CreateCallbacks` | — | Progress callbacks |

### `runtime.generate(input, opts?)`
Generates a full response. Routes to SSM or transformer per strategy. Returns `Promise<string>`.

### `runtime.stream(input, opts?)`
`AsyncIterable<string>` — always uses SSM path for consistent latency.

### `runtime.streamHybrid(input, opts?)`
`AsyncIterable<string>` — routes like `generate()`, streams via bridge if available.

### `runtime.adapt(data, opts?)`
Pass-through to `session.adapt()`. Returns `AdaptResult`.

### `runtime.evaluate(text)`
Returns SSM perplexity. Used internally by auto-routing.

### `runtime.getRoutingAuditLog()`
Returns `RoutingAuditEntry[]` — last 500 routing decisions with timing.

### `runtime.getDistillationLog()`
Returns `DistillationLog[]` — last 200 distillation runs (stub; use `distiller.getLog()` directly).

### `runtime.save(opts?)` / `runtime.load(opts?)`
Weight persistence pass-throughs to `MambaSession`.

### `runtime.destroy()`
Releases GPU device and all buffers.

---

## Error Handling

```ts
import { SSMError, SessionError } from '@seanhogg/builderforce-memory';

try {
  const runtime = await SSM.create({ session: { modelSize: 'nano' } });
  await runtime.generate('hello');
} catch (err) {
  if (err instanceof SSMError) {
    // Runtime-level error (bridge, distillation, memory)
    console.error(err.code);  // 'BRIDGE_REQUEST_FAILED' | 'RUNTIME_DESTROYED' | ...
  }
  if (err instanceof SessionError) {
    // Session-level error (GPU init, tokenizer, checkpoint)
    console.error(err.code);  // 'GPU_UNAVAILABLE' | 'TOKENIZER_LOAD_FAILED' | ...
  }
}
```

---

## File Structure

```
src/
├── index.ts                          ← package entry + SSM namespace
├── session/                          ← session layer (absorbed from @seanhogg/mambakit)
│   ├── session.ts                    ← MambaSession.create() — GPU, tokenizer, model, persistence
│   ├── tokenizer.ts                  ← Tokenizer interface (pluggable)
│   ├── presets.ts                    ← MODEL_PRESETS + layer schedule resolution
│   ├── persistence.ts                ← IndexedDB / download / File System API helpers
│   ├── streaming.ts                  ← AsyncIterable token streaming
│   ├── errors.ts                     ← SessionError typed error class
│   └── index.ts                      ← barrel export
├── runtime/
│   └── SSMRuntime.ts                 ← core runtime, owns MambaSession
├── bridges/
│   ├── TransformerBridge.ts          ← interface
│   ├── OpenAIBridge.ts               ← OpenAI chat completions
│   ├── AnthropicBridge.ts            ← Anthropic Messages API
│   └── FetchBridge.ts                ← generic OpenAI-compatible endpoint
├── router/
│   └── InferenceRouter.ts            ← SSM ↔ transformer routing + audit log
├── memory/
│   └── MemoryStore.ts                ← IndexedDB fact store: TTL, tags, importance, export/import
├── distillation/
│   └── DistillationEngine.ts         ← online teacher→student distillation + quality gates
├── agent/
│   └── SSMAgent.ts                   ← orchestration: history persistence + fact injection
└── errors/
    └── SSMError.ts                   ← typed error class
```

---

## Professional Platform

**@seanhogg/builderforce-memory patterns are the architectural foundation of [Builderforce.ai](https://builderforce.ai)'s Agent Runtime.**

| @seanhogg/builderforce-memory concept | Builderforce.ai equivalent |
|---|---|
| `SSMRuntime` | `AgentRuntime` (browser-native, ties to IDE project) |
| `DistillationEngine` | LLM-assisted dataset generation → in-browser LoRA training |
| `MemoryStore` | IndexedDB `MambaAgentState` + `AgentPackage` embedding |
| `SSMAgent` | Published workforce agent (Workforce Registry) |
| `TransformerBridge` | Cloudflare Workers AI / OpenRouter fallback |

---

## License

MIT
