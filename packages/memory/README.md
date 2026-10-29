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

## Evermind recall contract (`@seanhogg/builderforce-memory/evermind`)

A dependency-free subpath a browser chat client can import without the rest of the runtime:

- `EvermindRecallItem`, `EvermindRecallResult` — what a project Evermind recall returns.
- `rankEvermindRecall`, `hashRecallQuery` — the lexical recall ranker.
- `countReconciledMemories` — which recalled memories an answer restates (write-through reconcile).
- `formatEvermindMemoryBlock` — the recalled-memory prompt block, each item neutralised with `sanitizeRecalledFact`.
- `EVERMIND_MIN_TEACH_CHARS` — the teach floor below which a turn is never contributed.

## Working memory and code maps (`@seanhogg/builderforce-memory/working`)

A run has two memories. Both are pure (no I/O) and run in a Worker, a browser and Node.

### Short-term: `WorkingMemory.brief()` and the volatile tail

`WorkingMemory` records what this run has read, searched and changed. Record **successful** tool calls only. `brief()` restates that record once the run is long enough for compaction to have dropped its own results. Before that it returns `null`.

The brief changes every turn, so where it goes decides what the prompt cache can reuse. The rule:

- **Tag it and put it last.** Deliver the brief, and any other per-turn content (affect, budget line, task state), inside ONE trailing `user` message built with `volatileTail()` from `/handoff`. That message opens with `<run-state>`. The prompt-cache breakpoint is placed before a tagged block, so the history before it stays cached. `packHandoff` also never treats a tagged block as "the latest user directive".
- **Never persist it into the transcript.** Build the tail for each request and throw it away. A brief written into history grows every later turn and moves the cached prefix each time.
- **Never put it in the system prompt.** The system prompt is the start of the cached prefix, so content that changes per turn there invalidates the cache on every request.
- **Hosts that can only insert into the transcript should not use `brief()`.** Some SDKs only offer insertion points that persist, for example the Claude Agent SDK's `PostToolUse` `additionalContext`, which is written into the session transcript. A brief delivered there breaks both rules above. Such a host should skip the brief and still record into `WorkingMemory` for the run-end code map.

### Long-term: code maps

`consolidate(snapshot, { root?, at?, minFiles? })` turns a finished run into ONE memory per code area under the key `codemap:<area>`. The content is a map: changed files first, then reads ranked by how often the run opened them. The area is the deepest directory (2 to 4 segments) holding most of the files.

- **Minimum size.** A run must touch at least `MIN_CONSOLIDATED_FILES` (3) distinct files, counting changed, deleted and read together, or `consolidate` returns `null`. This applies to every run, including one that changed files. Pass `minFiles` to override it.
- **Root area.** A run whose files all sit at the repository root is filed under `codemap:(root)` (`CONSOLIDATED_ROOT_AREA`). `areaOfPath('package.json')` resolves to the same area, so the map can be recalled.
- **Merging.** A keyed store applies `mergeConsolidated(key, prior, next)` on write. The new run's entries come first, then the earlier map's entries it did not touch. An earlier `changed` is restated as `changed earlier`, files this run deleted leave the map, and the list is capped at 12. Pass `prior` only when the stored map is still live (not expired or superseded). Any non-`codemap:` key is stored as sent.
- **Reading a stored map.** `consolidatedFiles(content)` lists the files a map names. A store can anchor the map to them. `withoutConsolidatedFiles(content, paths)` drops files that changed since the map was written and keeps the rest. It returns `null` when no file is left.
- **Choosing maps for a run.** `consolidatedFactsFor(facts, areas)` keeps the maps whose area overlaps the areas a run is about, most specific first. `areasOfPathsIn(text)` gives the areas of the repo-relative paths a request names. `areasNamedIn(text, areas)` finds stored areas a request names by word (`"the run store"` matches `…/runStore`). Generic directory names such as `src` never match.

## Knowledge graph (`@seanhogg/builderforce-memory/knowledge`)

A directory review turns a folder into a graph of facts connected to each other and to the sources they came from (PRD 38). The subpath is pure TypeScript and runs in a browser, a Worker and Node. A host supplies ports only, never graph logic: a `DirectorySource`, a `KnowledgeStore`, an `IngestManifest`, and optionally a `JudgementPort` and a `similar` function.

**Store judgement, compute perception.** Nodes and edges derived from source text (`origin: 'extracted'`) are recomputed from the sources and cached by content hash through `KnowledgeStore.getExtracted` / `putExtracted`. They are never written as knowledge, and `assertStoredBatch` throws if a batch carries one. Only what a model, a person or an agent judged (`model`, `human`, `agent`) is stored, together with the content hash it was judged from. That judgement goes stale when the hash changes.

### What a review is

`reviewDirectory(args)` walks a `DirectorySource`, diffs it against the `IngestManifest` by content hash, and reviews only the changed files:

```
plan → per changed file: read · parse · extract (cached) · judge · write · markStale
     → analyzers → compare near-duplicate claims → manifest + scope digest
```

| `ReviewArgs` field | Meaning |
|---|---|
| `source: DirectorySource` | `{ rootId, title, list(), read(path), watch?() }`. One adapter per host. |
| `store: KnowledgeStore` | Where judgement is persisted. `LocalSnapshotStore` persists to a single JSON snapshot for a folder not linked to a project. |
| `packs: KnowledgePack[]` | Parsers, partitions, findings and a `JudgementSpec`. `DOCS_PACK` is built in. A host registers its own (for example a code pack). |
| `manifest: IngestManifest` | `get` / `set` / `delete` of `ChunkFingerprint[]` (`{ id, hash }`) per key. |
| `judgement?: JudgementPort` | `judge({ kind: 'extract' \| 'compare', items })`. Without it the review extracts and writes no model facts. |
| `similar?` | Near-duplicate claim pairs from a knowledge index. Needed, with `judgement`, for the compare phase. See below. |
| `budget: { modelCalls, ms }` | Judge-call and wall-clock caps. `DEFAULT_REVIEW_MODEL_CALLS` is 200. |
| `plan?: PlanDetail` | *Continue*: the plan the person confirmed. It is refreshed against the manifest, not re-walked. |
| `force?: boolean` | *Re-review all*. Ignored when `plan` is given. |
| `signal?`, `onProgress?` | Abort and progress. Phases: `plan`, `extract`, `resolve`, `judge`, `write`, `index`, `done`. |

Abort is honoured between files and inside judgement. A file is written whole or not at all. A file cut short by the budget stays changed in the manifest, so *Continue* judges only the sections not judged yet. A read that throws never retires a file: its knowledge and manifest hash stay as they were, `ReviewReport.readErrors` counts it, and the next review tries it again. Only a file absent from a fresh plan is removed.

`ReviewReport` carries `digest`, `extracted`, `judged`, `unjudged` (sections and compare pairs left for *Continue*), `invalidJudgements`, `factsWritten`, `relationsWritten`, `staleMarked`, `readErrors`, `aborted` and `durationMs`.

### Planning (the confirm chip)

`planReview(source, manifest, packs, opts?)` returns a `ReviewPlan` (`files`, `byPack`, `unchanged`, `changed`, `removed`, `skipped`, `estimatedModelCalls`). `planDetail` returns the same plan plus what `reviewDirectory` needs to *Continue*: a `PlanDetail` of `{ plan, files: PlannedFile[], prior, extractor, carried }`. A plan reads no file whose hash is unchanged. The only reads are the ignore files.

- **`PlanOptions.force`** marks every file changed, whatever its hash. Removals still come from the real manifest, which a fake empty manifest would lose along with the compare backlog. A forced run that is cut short is forced again; *Continue* re-derives what is left from content hashes.
- **`PlanOptions.compare`** (default `true`) says whether the run will compare near-duplicate claims. **A host that will not pass `similar` must pass `compare: false`.** Otherwise the estimate counts compare calls, including the waiting backlog of up to 2000 claims, that the run never spends.
- **`refreshPlan(detail, manifest)`** is *Continue*: it re-reads `changed` and `owed` against the manifest as the last run left it, without walking the directory. A file finished since is not reviewed again. A file the last run left owed is picked up.

`PlannedFile` is `{ path, hash, size, pack, changed, owed }`. `skipped` entries carry a reason: `ignored`, `binary`, `too-large` (default cap `DEFAULT_MAX_FILE_BYTES`, 2 MB, set with `maxFileBytes`) or `no-pack`.

`estimatedModelCalls` counts:

- one call per judged section of each file whose content changed, estimated at `AVG_SECTION_BYTES` (1500) bytes per section, for packs whose `JudgementSpec` asks for `claims` or `entities`;
- for an unchanged file that is owed a re-extraction, the calls it is still owed, plus one while an extractor upgrade is pending;
- when `compare` is true and a pack asks for `contradictions`, one compare call per 10 of those sections, including the claims already waiting in the compare backlog.

### Manifest keys a host persists

The review keeps its state in the `IngestManifest` you pass, under four keys per root. The host stores each value as an opaque list of `{ id, hash }` rows and returns them unchanged. It must not interpret, parse or drop rows.

| Key | Helper | Rows |
|---|---|---|
| `knowledge:<root>` | `filesManifestKey` | `{ id: path, hash }` for every reviewed file. |
| `knowledge:<root>:judged` | `judgedManifestKey` | `{ id: sectionRef, hash: textHash }` for every judged section. |
| `knowledge:<root>:compare` | `compareManifestKey` | `{ id: claimRef, hash: '<age>:<fileHash>' }` for claims awaiting comparison. Age 0 is the oldest. |
| `knowledge:<root>:extractor` | `extractorManifestKey` | The extractor version and the owed files. |

The `:extractor` entry holds `{ id: 'version', hash: '<EXTRACTOR_VERSION>' }` plus one `{ id: 'owed:<path>', hash: '<calls>' }` per file still owed a re-extraction, where `<calls>` is the number of model calls the file is expected to cost. A file is owed when the stored version differs from `EXTRACTOR_VERSION`, or when an earlier run left it unfinished: cut by the budget, a failed judge call, a read error, or an abort before it. Every run writes the entry, including a run that aborts. Unchanged sections re-anchor for free after an upgrade; only new sections cost a call. `ExtractorState` is the parsed form.

The compare backlog keeps the age in each row because a host manifest may return rows sorted by id. The review trims the backlog only after a compare actually ran. It is capped at 2000 claims, oldest dropped first. Only packs whose `judgement.contradictions` is true queue claims. A run without ports leaves the backlog as it found it.

### `similar(ids, texts?)`

```ts
similar?: (ids: string[], texts?: ReadonlyMap<string, string>) => Promise<SimilarPair[]>
```

`ids` are the pending claim refs, sorted. `texts` maps each ref to its claim text (`subject predicate object`), so a host can embed the claims without re-reading facts. A `SimilarPair` is `{ a, b, similarity }`. A port that accepts only `ids` keeps working. Typically the port upserts the claims into a `KnowledgeIndex` (exported from the same subpath) and returns its `similarPairs(...)`.

### Re-extracting one file

A host watching edits does not need a review to refresh a single file:

```ts
import { extractFile, extractFileDetailed, knownKeysOfEntities } from '@seanhogg/builderforce-memory/knowledge';

const extracted = await extractFile({ rootId, path, hash, doc, packs, store });
```

`ExtractFileArgs` is `{ rootId, path, hash, doc, packs, store, composed?, knownKeys? }`. `extractFile` returns the file's `Extracted`, stamped with `EXTRACTOR_VERSION`, or `undefined` when no pack's parser accepts it. `extractFileDetailed` returns `{ extracted, parser, pack, sections }`. Both read and write only the extracted cache, keyed by content hash, so a repeat call for the same hash is free. A cached entry of another extractor version is ignored. `mentions` edges resolve against the store's `ent:` facts unless you pass `knownKeys`, which `knownKeysOfEntities(keys)` builds from fact keys.

### Privacy bounds

A server that stores claims applies the same bounds the review applies to judge output:

- `clampQuote(text)` keeps at most `MAX_QUOTE_LINES` (3) lines and `MAX_QUOTE_CHARS` (300) characters.
- `clampClaim(content)` bounds the claim text to `MAX_CLAIM_TEXT_CHARS` (602: three terms of 200 characters plus two separators) and clamps its quote.

Both are idempotent, and `clampClaim` passes judge output through unchanged.

### Parsers, packs and the docs pack

`composePacks(packs)` builds the one `ParserRegistry` a review and a plan share and resolves each file to a `{ parser, pack }`. The built-in text parsers decline PDF and Office media types. Those files are reviewed only when a host parser claims their media type; otherwise they are skipped as `binary`. `.builderforce/ignore` and every `.gitignore` apply, a nested one to its own folder.

The built-in `DOCS_PACK` extracts:

- inline and reference-style links, and `[[wiki]]` links resolved by file name anywhere in the root;
- links that point outside the root, kept as broken links;
- a section for the text before the first heading (`#~preamble`).

Its findings are `brokenLinkRule`, `orphanDocumentRule` and `contradictionRule`. A stale relation (`KEdge.stale`) is shown, never treated as current.

### Building and tracing the graph

```ts
const graph = buildGraph({ rootId, extracted, facts, relations, packs }, { maxNodes: 400 });
const trace = traceGraph(graph, someRef, { maxDepth: 4 });
```

`buildGraph(input, opts?)` joins the cached extracts with the stored facts and relations into a `KnowledgeGraph` (`scope`, `nodes`, `edges`, `findings`, `stats`). Importance is PageRank, so the graph truncates to `maxNodes` (default 400, cap 2000) by importance. `[[Name]]` links are re-pointed here, over the files present now, so a cached extract never depends on which other files exist. `graphDigest(input)` is a content hash of the inputs. `traceGraph(graph, from, opts?)` follows `references`, `links` and `mentions` edges (`edgeKinds`) to `maxDepth` (4) and `maxSteps` (60), and groups the steps into partition lanes; `toMermaid` draws a trace as a sequence diagram. `computeFindings` combines the generic findings (`cycle`, `clone-group`, `near-duplicate`, `partition-violation`, `unpartitioned`, `stale`) with each pack's `FindingRule`s.

### A minimal host

```ts
import {
  DOCS_PACK, planDetail, reviewDirectory,
  type DirectorySource, type JudgementPort, type KnowledgeStore, type PlanDetail,
} from '@seanhogg/builderforce-memory/knowledge';
import type { IngestManifest } from '@seanhogg/builderforce-memory';

declare const source: DirectorySource;
declare const store: KnowledgeStore;
declare const manifest: IngestManifest;
declare const judgement: JudgementPort | undefined;
declare const confirm: (calls: number) => Promise<boolean>;

const packs = [DOCS_PACK];
// This host passes no `similar`, so it must plan with compare: false.
const detail: PlanDetail = await planDetail(source, manifest, packs, { compare: false });

if (await confirm(detail.plan.estimatedModelCalls)) {
  const report = await reviewDirectory({
    source, store, packs, manifest, judgement,
    budget: { modelCalls: 200, ms: 120_000 },
    plan: detail,            // omit to let the review plan for itself
  });
  console.log(report.judged, 'judged,', report.unjudged, 'left for Continue,', report.readErrors, 'unreadable');
}
```

When `plan` is omitted, `reviewDirectory` plans with `compare` set to whether it was given both `similar` and `judgement`, so a host that does not pass `similar` is estimated correctly. When a host passes a `plan` it built itself, the `compare` option it planned with must match the ports it passes.

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
