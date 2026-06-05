# MambaCode.js — Integration & Architecture Guide

> **Mamba as a Unified Brain + Memory System**
>
> This document is a production-grade architecture reference for engineers, platform architects, and technical decision-makers embedding MambaCode.js into AI applications.

---

## Table of Contents

1. [Conceptual Overview](#1-conceptual-overview)
2. [Mamba as Brain + Memory](#2-mamba-as-brain--memory)
3. [Embedding Mamba Into a System](#3-embedding-mamba-into-a-system)
4. [Execution Flow — The New Paradigm](#4-execution-flow--the-new-paradigm)
5. [Memory Model](#5-memory-model)
6. [Integration Patterns](#6-integration-patterns)
7. [Advanced Use Cases](#7-advanced-use-cases)
8. [Comparison: Mamba vs Transformers](#8-comparison-mamba-vs-transformers)
9. [Performance Considerations](#9-performance-considerations)
10. [Design Tradeoffs](#10-design-tradeoffs)
11. [Future Architecture Vision](#11-future-architecture-vision)

---

## 1. Conceptual Overview

### What is Mamba (SSM)?

**Mamba** is a *Selective State Space Model (SSM)* — specifically the **S6** variant — that processes sequences by maintaining a compact, evolving hidden state rather than attending over all previous tokens.

At each time step, the state update equations are:

```
h_t = Ā · h_{t-1}  +  B̄ · x_t     ← absorb new input into state
y_t =  C · h_t     +   D · x_t     ← read from state to produce output
```

The matrices `B` and `C` are *input-dependent* (selective), allowing the model to decide what to remember and what to ignore at every step. This selectivity is what separates Mamba from earlier SSMs and makes it a credible alternative to the Transformer attention mechanism.

### Why Mamba Differs from Transformers

A Transformer's self-attention is a *global pairwise operation*: every token attends to every other token, giving O(N²) cost in both compute and memory. Mamba replaces this with a *linear recurrence* that runs in O(N) time and **constant** memory per inference step.

```
Transformer:          Mamba:
Token[0] ─────┐       h₀ ──►[S6]──► h₁ ──►[S6]──► h₂ ──► …
Token[1] ─────┤         ↑              ↑
Token[2] ─────┤        x₁             x₂
Token[3] ─────┘
(All pairs interact — O(N²))    (State flows forward — O(N))
```

### Why This Enables Long-Context Reasoning, Persistent Memory, and Efficient Local Execution

| Property | Transformer | Mamba (S6) |
|---|---|---|
| Context scaling | O(N²) | **O(N)** |
| Memory per token at inference | Grows with sequence | **Constant** |
| Inference speed as length increases | Slows | **Constant** |
| On-device / browser feasibility | Difficult | **Yes** |
| Knowledge retention across turns | Requires KV cache | **Embedded in state** |

Because the hidden state carries accumulated knowledge forward continuously, Mamba naturally encodes *memory* as a first-class property of the model — not as an external attachment.

---

## 2. Mamba as Brain + Memory

### The Traditional Fragmented Architecture

Most AI application stacks compose three separate systems:

```
┌────────────────────────────────────────────────────────────┐
│               Traditional AI Application Stack             │
│                                                            │
│   ┌──────────┐    prompt    ┌──────────┐                  │
│   │ Vector   │ ──────────► │   LLM    │ ──► Response      │
│   │ Database │ (retrieval)  │  (brain) │                   │
│   │ (memory) │              └──────────┘                  │
│   └──────────┘                   ▲                        │
│        ▲                         │                        │
│        │              Prompt Engineering                   │
│        │              (glue / orchestration)              │
│   User Query ──────────────────────────────────────────►  │
└────────────────────────────────────────────────────────────┘

Components:
  • LLM              — the reasoning engine (stateless per call)
  • Vector Database  — external memory store (retrieval at query time)
  • Prompt           — the glue that injects context into each call
```

This architecture has fundamental weaknesses:

- **Stateless calls**: the LLM forgets everything between requests.
- **Fragile retrieval**: relevant context is only as good as the embedding + search quality.
- **Latency overhead**: every inference requires a retrieval round-trip.
- **Privacy risk**: context must be shipped to a remote API.
- **No learning**: the model never improves from usage.

### The Mamba Unified Architecture

Mamba eliminates this fragmentation. The model *is* the memory system:

```
┌────────────────────────────────────────────────────────────┐
│               Mamba Unified Intelligence Layer             │
│                                                            │
│   User Input                                               │
│       │                                                    │
│       ▼                                                    │
│   ┌────────────────────────────────────────────────────┐   │
│   │                  MambaModel                        │   │
│   │                                                    │   │
│   │  • Processes input in O(N) time                   │   │
│   │  • Maintains hidden state h_t across calls        │   │
│   │  • Knowledge embedded directly in weights         │   │
│   │  • Adapts via WSLA without external store         │   │
│   └────────────────────────────────────────────────────┘   │
│       │                                                    │
│       ▼                                                    │
│   Response  ──►  Adapt  ──►  Improved Future Response      │
└────────────────────────────────────────────────────────────┘

Components:
  • MambaModel  — brain AND memory, unified in one model
  • No external vector DB required
  • No prompt injection for context
  • No stateless round-trips
```

**Core principle:**

> Memory is NOT separate. Context is NOT injected. Learning is NOT external.
> Knowledge is embedded directly into model weights and state transitions.

---

## 3. Embedding Mamba Into a System

### Step 1 — Add the Mamba Runtime

```js
import {
  MambaModel,
  MambaTrainer,
  BPETokenizer,
  initWebGPU,
} from 'mambacode.js';

// Initialise WebGPU (runs on-device — no cloud required)
const { device } = await initWebGPU();

// Load the Qwen3.5-Coder tokenizer vocabulary
const tokenizer = new BPETokenizer();
await tokenizer.load('/vocab.json', '/merges.txt');

// Create the model
const model = new MambaModel(device, {
  vocabSize : tokenizer.vocabSize,   // 151936
  dModel    : 512,
  numLayers : 8,
  dState    : 16,
  dConv     : 4,
  expand    : 2,
});

// Load a pre-trained checkpoint
const response = await fetch('/models/mamba-coder-checkpoint.bin');
await model.loadWeights(await response.arrayBuffer());
```

### Step 2 — Replace Stateless Calls

**Before (stateless LLM call):**

```js
// Every call is independent — the model has no memory of previous interactions
const response = await llm.generate(prompt);
```

**After (Mamba with persistent context):**

```js
// The model processes the full sequence and carries state forward
const promptIds = tokenizer.encode(input);
const outputIds = await model.generate(promptIds, maxTokens, { temperature: 0.8 });
const output    = tokenizer.decode(outputIds);
```

The model's hidden state encodes accumulated context. There is no separate retrieval step; the model's recurrent state *is* the context.

### Step 3 — Introduce Persistent State

Persist the model's learned weights between sessions so accumulated knowledge survives page reloads or application restarts:

```js
// Export learned weights after a session
const checkpoint = await model.exportWeights();

// Persist to IndexedDB for offline-capable applications
const db = await openDB('mamba-app', 1, {
  upgrade(db) { db.createObjectStore('checkpoints'); }
});
await db.put('checkpoints', checkpoint, 'user-model');

// Reload on the next session — resume exactly where the model left off
const saved   = await db.get('checkpoints', 'user-model');
await model.loadWeights(saved);
```

### Step 4 — Enable Adaptation

The model improves from usage via **WSLA** (Weight-Selective Local Adaptation). WSLA fine-tunes only the B and C matrices of the selective scan — the parameters that govern what the model chooses to remember. This is lightweight enough to run continuously in a background thread:

```js
import { MambaTrainer } from 'mambacode.js';

const trainer = new MambaTrainer(model, tokenizer);

// Rapid domain adaptation — only B and C matrices are updated
await trainer.train(newDomainData, {
  learningRate : 1e-4,
  epochs       : 3,
  wsla         : true,   // fast learning loop
});
```

**How WSLA works:**

```
Full fine-tune:  update ALL weights  (slow, high memory)
WSLA:            update B, C only   (fast, minimal memory)

B controls:  how new input is absorbed into the state
C controls:  what is read back from the state to form output

Updating only B and C lets the model learn WHAT to remember
without disturbing the general reasoning capabilities encoded
in the other weight matrices.
```

---

## 4. Execution Flow — The New Paradigm

### Traditional Flow (Stateless + RAG)

```
Query
  │
  ├──► Vector Search ──► Retrieved Chunks
  │                             │
  └─────────────────────────────►─── LLM (stateless) ──► Response
                                       (forgets everything
                                        after this call)
```

### Mamba Flow (Unified Intelligence)

```
Input ──► MambaModel ──► Output
               │
             Adapt
               │
       Improved Future Output
```

In detail:

```
                     ┌─────────────────────────────────────────┐
                     │            MambaModel                   │
                     │                                         │
 New Input ─────────►│  h_t = Ā·h_{t-1} + B̄·x_t              │
                     │  y_t = C·h_t + D·x_t                   │
                     │                                         │
                     │  State h_t carries all previous context │
                     └─────────────────┬───────────────────────┘
                                       │
                                    Output
                                       │
                           ┌───────────▼───────────┐
                           │   Adaptation (WSLA)   │
                           │   trainer.train(data) │
                           └───────────┬───────────┘
                                       │
                           Updated weights / state
                                       │
                           Better future responses
```

---

## 5. Memory Model

Mamba provides three complementary memory layers. The first two are intrinsic; the third is optional for structured data that does not fit in the model's learned parameters.

### Layer 1 — Embedded Memory (in weights)

Knowledge encoded during pre-training and fine-tuning is stored directly in the model's weight matrices. This is the model's *long-term semantic memory*: facts, patterns, and style that persist across all sessions.

```
Pre-training corpus ──► trainer.train(largeDataset)
                              │
                    Weight matrices updated
                              │
               Knowledge permanently embedded
               (survives application restart)
```

### Layer 2 — Temporal Memory (in state transitions)

The hidden state `h_t` is the model's *working memory*: it accumulates context token-by-token across an entire conversation or document. Because the recurrence is linear, this working memory has **constant size** regardless of sequence length.

```
Turn 1:  h_1  =  Ā·h_0  + B̄·x_1        (h_1 encodes Turn 1)
Turn 2:  h_2  =  Ā·h_1  + B̄·x_2        (h_2 encodes Turns 1–2)
Turn 3:  h_3  =  Ā·h_2  + B̄·x_3        (h_3 encodes Turns 1–3)
                                          ← fixed size, not growing
```

### Layer 3 — Optional External Memory

For use cases requiring structured retrieval (audit logs, exact-match lookups, multi-user shared knowledge bases), an external store can complement the model without replacing it:

```js
// Optional: augment with external structured data when needed
const relevantChunks = await structuredStore.query(userQuery);
const augmentedInput = `${relevantChunks.join('\n')}\n\n${userQuery}`;
const promptIds      = tokenizer.encode(augmentedInput);
const outputIds      = await model.generate(promptIds, maxTokens);
```

> **Design rule:** use external memory only when the data is structured, requires exact-match retrieval, or changes faster than the model can adapt. Do not use it as a substitute for the model's own contextual memory.

---

## 6. Integration Patterns

### A. Agent Systems (e.g., CoderClaw)

Mamba serves as the reasoning core of a software agent. Because the model's state persists across agent steps, it retains the full execution history without injecting transcripts into a prompt.

```js
// Agent loop — model state persists across steps
const agent = new MambaModel(device, config);
await agent.loadWeights(await fetch('/checkpoints/agent-v1.bin')
  .then(r => r.arrayBuffer()));

for (const step of agentSteps) {
  const input     = tokenizer.encode(step.observation);
  const actionIds = await agent.generate(input, 128, { temperature: 0.3 });
  const action    = tokenizer.decode(actionIds);

  await step.execute(action);

  // Adapt from the outcome of this step — continuous learning
  if (step.feedback) {
    const trainer = new MambaTrainer(agent, tokenizer);
    await trainer.train(step.feedback, { learningRate: 5e-5, epochs: 1, wsla: true });
  }
}
```

**Mamba handles:**
- Multi-step context understanding (state carries full history)
- Online learning from agent execution outcomes

### B. IDEs and Developer Tools (e.g., BuilderForce)

Mamba powers code-completion and in-editor AI assistance with personalization. The model learns a developer's codebase privately, on-device, without transmitting source code.

```js
// Fine-tune on the developer's private codebase (runs in browser)
const trainer = new MambaTrainer(model, tokenizer);
await trainer.train(projectSourceFiles.join('\n'), {
  learningRate : 1e-4,
  epochs       : 5,
  wsla         : true,
});

// Personalized code completion
editor.on('keypress', async () => {
  const context   = editor.getPrecedingText();
  const promptIds = tokenizer.encode(context);
  const nextIds   = await model.generate(promptIds, 64, { temperature: 0.5 });
  editor.showSuggestion(tokenizer.decode(nextIds));
});
```

**Mamba enables:**
- Code intelligence tuned to the developer's style
- Personalization without cloud round-trips

### C. APIs and SaaS Platforms

Deploy per-user adaptive models. Each user's model checkpoint is a small binary (~10–100 MB) that can be stored on the server and loaded on demand.

```
                   SaaS Platform
┌────────────────────────────────────────────┐
│                                            │
│  User A ──► Load checkpoint_A.bin          │
│             model.generate(inputA)         │
│             trainer.train(feedbackA, ...)  │
│             model.exportWeights() ──► save │
│                                            │
│  User B ──► Load checkpoint_B.bin          │
│             model.generate(inputB)         │
│             trainer.train(feedbackB, ...)  │
│             model.exportWeights() ──► save │
│                                            │
│  Each user has a model that has learned    │
│  exclusively from their own interactions.  │
└────────────────────────────────────────────┘
```

### D. Edge and Browser Applications

MambaCode.js runs entirely in the browser via WebGPU. No server, no network calls after the initial checkpoint download.

```
Browser Tab
─────────────────────────────────────────────────────────
  User input
      │
      ▼
  BPETokenizer (client-side BPE)
      │
      ▼
  MambaModel (WebGPU on local GPU)
      │
  ┌───┴──────────────────────────────┐
  │  Optional: MambaTrainer (WSLA)  │  ← adapts on user's own device
  └───┬──────────────────────────────┘
      │
      ▼
  Generated text / code
─────────────────────────────────────────────────────────
  ✅  Private by design. Zero data leaves the device.
```

---

## 7. Advanced Use Cases

### 7.1 Self-Healing Software Systems

The model monitors runtime failures, learns the patterns that precede them, and adjusts its code-generation recommendations to avoid repeating them.

```js
// On every CI or test failure, adapt the model
async function onTestFailure(failingCode, errorMessage) {
  const trainingText = `
// FAILURE PATTERN — do not repeat
${failingCode}
// ERROR: ${errorMessage}
// CORRECTED VERSION:
`;
  const trainer = new MambaTrainer(model, tokenizer);
  await trainer.train(trainingText, { learningRate: 5e-5, epochs: 2, wsla: true });
}

// Over time, the model learns to avoid patterns that historically caused failures
```

**Flow:**

```
Code Change ──► CI Run ──► Failure Detected
                                │
                         Adapt model on
                         failure pattern
                                │
                    Future suggestions avoid
                    the failure pattern
```

### 7.2 Personalized AI Assistants

The assistant model adapts to each user's vocabulary, communication style, and domain knowledge over time — without storing chat history in a database.

```js
// After each interaction, incorporate user feedback
async function learnFromInteraction(userMessage, modelResponse, userRating) {
  if (userRating === 'positive') {
    const trainingText = `User: ${userMessage}\nAssistant: ${modelResponse}`;
    const trainer = new MambaTrainer(model, tokenizer);
    await trainer.train(trainingText, { learningRate: 1e-5, epochs: 1, wsla: true });
  }
  // Persist updated checkpoint to IndexedDB
  const checkpoint = await model.exportWeights();
  await db.put('checkpoints', checkpoint, `user-${userId}`);
}
```

**After N interactions:**

```
Session 1:  model knows nothing about the user
Session 10: model begins to mirror the user's style
Session 50: model anticipates the user's intent and domain preferences
```

### 7.3 Autonomous Coding Agents

The agent is fine-tuned on a repository's patterns and conventions. As it generates and executes code, it continues learning from the results.

```js
// Bootstrap: learn the repository conventions
const repoTrainer = new MambaTrainer(model, tokenizer);
await repoTrainer.train(allRepositoryFiles, {
  learningRate : 1e-4,
  epochs       : 3,
  wsla         : true,
});

// Autonomous execution loop
while (!taskComplete) {
  const task   = await planner.nextTask();
  const code   = await generateCode(model, tokenizer, task.spec);
  const result = await executor.run(code);

  if (result.success) {
    // Reinforce successful patterns
    await repoTrainer.train(code, { learningRate: 5e-5, epochs: 1, wsla: true });
  } else {
    // Learn from the error
    await repoTrainer.train(
      `// ERROR: ${result.error}\n// Avoid: ${code}`,
      { learningRate: 5e-5, epochs: 1, wsla: true }
    );
  }
}
```

### 7.4 Enterprise Knowledge Systems

Replace a Vector DB + RAG pipeline with a model that has the enterprise knowledge embedded directly in its weights. New documents are ingested by fine-tuning; retrieval happens implicitly through generation.

```
Traditional RAG Pipeline:
  New Doc ──► Embed ──► Vector DB ──► Search ──► Prompt ──► LLM ──► Answer
               (lossy compression)   (approximate match)   (stateless)

Mamba Knowledge System:
  New Doc ──► trainer.train(doc, { wsla: true }) ──► Weights Updated
                                                            │
  Query ─────────────────────────────────────────► model.generate() ──► Answer
                                                   (knowledge is in the model)
```

```js
// Ingest new enterprise documents incrementally
async function ingestDocument(documentText) {
  const trainer = new MambaTrainer(model, tokenizer);
  await trainer.train(documentText, {
    learningRate : 5e-5,
    epochs       : 2,
    wsla         : true,
  });
  // Persist updated checkpoint
  const checkpoint = await model.exportWeights();
  await fs.writeFile('enterprise-knowledge.bin', Buffer.from(checkpoint));
}

// Query — no retrieval step needed
async function query(question) {
  const ids    = tokenizer.encode(question);
  const outIds = await model.generate(ids, 256, { temperature: 0.3 });
  return tokenizer.decode(outIds);
}
```

### 7.5 Real-Time Adaptive UX

The model learns which UI patterns the user responds to positively and adapts its suggestions without any server-side analytics pipeline.

```js
// Track which generated UI suggestions the user accepts
uiEngine.on('suggestionAccepted', async (prompt, suggestion) => {
  const trainingText = `UI Prompt: ${prompt}\nBest UI: ${suggestion}`;
  const trainer = new MambaTrainer(uiModel, tokenizer);
  await trainer.train(trainingText, { learningRate: 5e-5, epochs: 1, wsla: true });
});

// Future suggestions are biased towards accepted patterns
async function suggestUI(context) {
  const ids    = tokenizer.encode(context);
  const outIds = await uiModel.generate(ids, 128, { temperature: 0.4 });
  return tokenizer.decode(outIds);
}
```

---

## 8. Comparison: Mamba vs Transformers

| Feature | Transformer | Mamba (MambaCode.js) |
|---|---|---|
| **Context scaling** | O(N²) — degrades with length | **O(N) — constant cost** |
| **Memory architecture** | External KV cache or vector DB | **Embedded in model state** |
| **Per-step inference cost** | Grows with sequence length | **Constant** |
| **Adaptation speed** | Full fine-tune required | **WSLA: B & C only — fast** |
| **Adaptation cost** | High (GPU hours) | **Low (minutes on-device)** |
| **On-device feasibility** | Limited (quadratic memory) | **Yes (linear memory)** |
| **Privacy** | Typically server-side | **Local — data stays on device** |
| **Retrieval requirement** | RAG / vector DB for memory | **None — knowledge in weights** |
| **Multi-session continuity** | Lost unless explicitly stored | **Persistent via checkpoint** |
| **Operational complexity** | LLM + vector DB + orchestration | **Single model** |

---

## 9. Performance Considerations

### WebGPU and GPU Utilization

MambaCode.js executes all compute-heavy operations — embedding lookup, selective scan, projection, and AdamW updates — as WGSL compute shaders dispatched to the GPU via WebGPU.

```
CPU (JavaScript)                GPU (WGSL via WebGPU)
────────────────                ─────────────────────────────
Orchestration                   selective_scan.ts  (S6 core)
Tokenization                    conv1d.ts          (causal conv)
Weight serialization            linear_projection.ts (GEMM)
Checkpoint I/O                  weight_update.ts   (AdamW)
                                activations.ts     (SiLU, RMSNorm)
```

**Practical GPU guidelines:**

| Model size | `dModel` | `numLayers` | Approx. VRAM | Recommended hardware |
|---|---|---|---|---|
| Small | 256 | 4 | ~500 MB | Integrated GPU |
| Medium | 512 | 8 | ~1.5 GB | Mid-range discrete GPU |
| Large | 1024 | 16 | ~3 GB | High-end consumer GPU |

### Memory Footprint

- **Inference**: memory is bounded by the model weight size plus one state vector `h_t` (size: `dModel × dState`). It does **not** grow with sequence length.
- **Training (WSLA)**: only the B and C gradient buffers are allocated in addition to model weights. Full fine-tune requires optimizer state (2× weight size for AdamW moment buffers).

### Scaling Behavior

```
Transformer:
  Compute   ∝  N²  (N = sequence length)
  Memory    ∝  N   (KV cache)

Mamba:
  Compute   ∝  N   (one pass per token)
  Memory    =  const (state size fixed)
```

For practical applications with context lengths of 4K–32K tokens, Mamba's linear scaling translates to 10–100× lower compute cost compared to an equivalently-sized Transformer.

---

## 10. Design Tradeoffs

### When to Use Mamba as the Unified Intelligence Layer

✅ **Use Mamba when:**

- You need on-device, private inference (edge, browser, enterprise laptop).
- Your application benefits from continuous personalization or domain adaptation.
- Context length exceeds ~2K tokens and Transformer costs become prohibitive.
- You want to eliminate the operational overhead of a separate vector database.
- Low-latency inference is required (real-time completions, live agents).
- The privacy or compliance requirements preclude sending data to an API.

### When NOT to Use Mamba Alone

❌ **Consider a hybrid or alternative when:**

- **Exact retrieval is required**: Mamba's embedded memory is probabilistic. If you need bit-exact recall of a specific document (e.g., a legal contract), a structured database or exact-match index is more appropriate alongside the model.
- **Knowledge updates are extremely frequent**: Adapting the model takes seconds to minutes. If your data changes many times per second, a traditional database is a better primary store.
- **Huge pre-trained capability is required**: Frontier LLMs (GPT-4, Claude, Gemini) have been trained on vastly more data. For tasks requiring broad world knowledge out of the box, a large hosted LLM may outperform a locally fine-tuned Mamba of comparable size.
- **Multi-modal inputs**: MambaCode.js is text/code focused. Vision, audio, or structured tabular inputs require additional preprocessing pipelines.

### Hybrid Architecture Pattern

When exact retrieval is needed alongside adaptive generation, the two can coexist cleanly:

```js
// 1. Fast exact lookup for structured facts
const structuredFacts = await structuredDB.lookup(entityId);

// 2. Mamba for reasoning and adaptive generation
const context   = formatFacts(structuredFacts) + '\n' + userQuery;
const promptIds = tokenizer.encode(context);
const outputIds = await model.generate(promptIds, 256, { temperature: 0.4 });
const answer    = tokenizer.decode(outputIds);
```

In this pattern, Mamba handles reasoning and generation; the structured store provides ground-truth facts. The two layers are complementary, not redundant.

---

## 11. Future Architecture Vision

### The Self-Improving Software System

The architecture described in this document points toward a class of systems that continuously improve themselves:

```
┌─────────────────────────────────────────────────────────────────┐
│               Self-Improving AI System (Future)                 │
│                                                                 │
│   User / Environment Input                                      │
│           │                                                     │
│           ▼                                                     │
│   ┌────────────────────┐                                        │
│   │    MambaModel      │  ← reasoning + memory unified         │
│   │  (always on-line)  │                                        │
│   └────────┬───────────┘                                        │
│            │                                                    │
│         Output                                                  │
│            │                                                    │
│      ┌─────▼──────┐                                             │
│      │  Evaluate  │  ← measure output quality                  │
│      └─────┬──────┘                                             │
│            │                                                    │
│      ┌─────▼────────────┐                                       │
│      │  WSLA Adaptation │  ← learn from outcome                │
│      └─────┬────────────┘                                       │
│            │                                                    │
│   Updated model — better next time                              │
└─────────────────────────────────────────────────────────────────┘
```

### Key Characteristics of Post-RAG Systems

| Traditional AI Stack | Post-RAG Mamba System |
|---|---|
| Memory is a separate service | **Memory is intrinsic to the model** |
| Learning requires re-training a new model | **Learning is continuous and incremental** |
| Context is injected per request | **Context is carried in recurrent state** |
| Multiple services to operate | **Single model binary** |
| Knowledge frozen at training time | **Knowledge evolves with usage** |
| Privacy requires special handling | **Private by default — runs locally** |

### Toward Intrinsic Intelligence

The fundamental shift enabled by Mamba is the **collapse of the distinction between memory and intelligence**. In the Transformer paradigm, memory must be managed externally because the model forgets. In the Mamba paradigm, memory is a structural property of the model itself.

This enables a new class of applications:

- **Software that improves from every execution** — each run teaches the model something new about the codebase, the user's intent, and the failure modes to avoid.
- **Assistants with genuine continuity** — rather than simulating memory through prompt injection, the model actually remembers across sessions via its learned weights.
- **Autonomous agents that get better at their job** — not through hand-crafted reward functions, but through lightweight continuous adaptation on observed outcomes.

The architecture described in this document is the foundation of that future.

---

*Back to [README](../README.md) · [Getting Started](./getting-started.md) · [API Reference](./api-reference.md) · [Weight Lifecycle](./weight-lifecycle.md)*
