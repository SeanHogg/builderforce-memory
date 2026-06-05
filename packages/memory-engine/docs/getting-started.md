# Getting Started with MambaCode.js

> **New to AI and language models?** This guide walks you through everything from the basics of how large language models work, all the way to running your first on-device code model — entirely in the browser.

---

## Table of Contents

1. [What is a Language Model?](#1-what-is-a-language-model)
2. [Where Does Qwen Fit In?](#2-where-does-qwen-fit-in)
3. [What is Mamba and Why Does it Matter?](#3-what-is-mamba-and-why-does-it-matter)
4. [How MambaCode.js Brings It All Together](#4-how-mambacodejs-brings-it-all-together)
5. [Prerequisites](#5-prerequisites)
6. [Step-by-Step: Your First On-Device Model](#6-step-by-step-your-first-on-device-model)
7. [The Complete Lifecycle at a Glance](#7-the-complete-lifecycle-at-a-glance)
8. [What Happens Next?](#8-what-happens-next)
9. [Using builderforce.ai](#9-using-builderforceai)
10. [Common Questions](#10-common-questions)

---

## 1. What is a Language Model?

A **language model (LM)** is a type of AI that learns patterns in text (or code) and uses those patterns to predict what comes next. When you type a sentence and your phone suggests the next word, that is a tiny language model at work.

**Large Language Models (LLMs)** take the same idea much further: they are trained on billions of lines of text and code, giving them the ability to:

- Answer questions in natural language
- Write, explain, and debug code
- Summarise documents and translate text
- Power AI assistants and copilots

### How are they trained?

Training a language model means adjusting millions (or billions) of numerical *weights* so that the model gets better at predicting the next token. A **token** is roughly a word, part of a word, or a code symbol — the model never sees raw characters, only these numeric IDs.

```
Raw text:  "function add("
Tokens:    [1, 2543, 912, 40]   ← numeric IDs the model actually processes
```

Training is done by:

1. Feeding the model a sequence of tokens
2. Asking it to predict the next token
3. Measuring the error (the *loss*)
4. Adjusting the weights to reduce the loss (back-propagation + optimizer)

After millions of these steps the model "knows" a great deal about language and code.

---

## 2. Where Does Qwen Fit In?

**Qwen3.5-Coder** is a family of open-weight code-focused language models built by Alibaba Cloud. The 0.8 B (800 million parameter) variant is small enough to run locally yet capable enough to help with real code tasks.

MambaCode.js targets the **Qwen3.5-Coder tokenizer vocabulary** (151 936 tokens). This means:

- The tokenizer included in this library produces the same token IDs that Qwen3.5-Coder models expect.
- Checkpoints downloaded from [builderforce.ai](https://builderforce.ai) or exported from a previous MambaCode.js session can be loaded directly.
- You can fine-tune the model on your own codebase without sending any data to a server.

> **Important:** Qwen and Mamba use different architectures — Qwen is a Transformer and Mamba is a State Space Model. Their **tokenizer** vocabularies are compatible, but **raw Qwen model weights cannot be loaded into MambaCode.js directly**. See the [Weight Lifecycle guide](./weight-lifecycle.md#2-understanding-the-qwenmamba-relationship) for the full explanation.

> **In short:** Qwen provides the tokenizer vocabulary. MambaCode.js provides the engine to run and adapt a Mamba model locally — in the browser, with no data leaving your machine.

---

## 3. What is Mamba and Why Does it Matter?

Traditional LLMs use a **Transformer** architecture whose attention mechanism has quadratic cost: doubling the sequence length roughly quadruples the compute. This is manageable in a data centre but challenging on a single GPU or browser.

**Mamba** is a *State Space Model (SSM)* — specifically the **S6 (Selective Scan)** variant. Instead of attending to every previous token, Mamba maintains a compact hidden state that is updated step-by-step:

```
h_t = Ā · h_{t-1}  +  B̄ · x_t     (update state with new input)
y_t = C  · h_t     +  D · x_t     (read from state to produce output)
```

Key advantages over a Transformer for on-device use:

| Property | Transformer | Mamba (S6) |
|---|---|---|
| Context scaling | O(N²) | **O(N)** |
| Memory per token | Grows with sequence | **Constant** |
| Inference speed | Slows with length | **Constant** |
| Good for local GPU | Difficult | **Yes** |

This linear scaling is what makes it possible to run a meaningful code model inside a browser tab.

---

## 4. How MambaCode.js Brings It All Together

```
┌──────────────────────────────────────────────────────────┐
│                       Browser Tab                        │
│                                                          │
│  Your code / text                                        │
│       │                                                  │
│       ▼                                                  │
│  BPETokenizer  ──►  token IDs (Qwen3.5-Coder vocab)     │
│       │                                                  │
│       ▼                                                  │
│  MambaModel  (WebGPU / WGSL kernels on your local GPU)  │
│       │                                                  │
│  ┌────┴────────────────────┐                             │
│  │  Optional: MambaTrainer │  ← fine-tune on your code  │
│  └────┬────────────────────┘                             │
│       │                                                  │
│       ▼                                                  │
│  Generated tokens  ──►  BPETokenizer.decode  ──► text   │
│                                                          │
│  ✅  No network calls. No cloud. Private by design.      │
└──────────────────────────────────────────────────────────┘
```

The three main pieces:

| Component | Role |
|---|---|
| **BPETokenizer** | Converts text ↔ token IDs using the Qwen3.5-Coder vocabulary |
| **MambaModel** | Runs the Mamba SSM forward pass on the GPU via WebGPU |
| **MambaTrainer** | Fine-tunes the model on your local code using AdamW + autograd |

---

## 5. Prerequisites

### Browser

| Browser | Minimum Version | Notes |
|---|---|---|
| Chrome | 113+ | Recommended — best WebGPU support |
| Edge | 113+ | Same Chromium engine as Chrome |
| Firefox | Nightly | Enable `dom.webgpu.enabled` in `about:config` |
| Safari | 18+ | WebGPU in preview; some features may be limited |

> Open `chrome://gpu` and look for **WebGPU** to confirm it is enabled.

### Hardware

A dedicated GPU is not required, but it dramatically speeds things up. The model fits within **≤ 3 GB VRAM**, which covers most modern integrated and discrete graphics.

### Node.js (for local development only)

If you want to run the test suite or bundle the library:

```bash
node --version   # must be >= 18
```

---

## 6. Step-by-Step: Your First On-Device Model

### Step 1 — Install

**Building a web app (npm):**

```bash
npm install mambacode.js
```

**Local development / exploring the source:**

```bash
git clone https://github.com/SeanHogg/Mamba.git
cd Mamba
npm install
npm run build   # compile TypeScript → dist/
```

**Direct browser import (no bundler):**

```html
<script type="module">
  import { MambaModel, MambaTrainer, BPETokenizer, initWebGPU }
    from 'https://cdn.jsdelivr.net/npm/mambacode.js@1.0.0/dist/index.js';
</script>
```

---

### Step 2 — Obtain the Qwen vocabulary files

Download `vocab.json` and `merges.txt` from HuggingFace and serve them from your web server:

```bash
# Download via curl
curl -L -o public/vocab.json  "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B/resolve/main/vocab.json"
curl -L -o public/merges.txt  "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B/resolve/main/merges.txt"
```

> See the [Weight Lifecycle guide](./weight-lifecycle.md#1-before-mamba--obtaining-qwen-vocabulary-files) for full instructions including the HuggingFace CLI method.

---

### Step 3 — Check WebGPU is available

```js
import { initWebGPU } from 'mambacode.js';

const { device } = await initWebGPU();
console.log('WebGPU ready ✅', device.label);
```

If this throws, check that your browser supports WebGPU (see [Prerequisites](#5-prerequisites)).

---

### Step 4 — Load the tokenizer

```js
import { BPETokenizer } from 'mambacode.js';

const tokenizer = new BPETokenizer();
await tokenizer.load('/vocab.json', '/merges.txt');

console.log('Vocabulary size:', tokenizer.vocabSize);  // 151936
```

---

### Step 5 — Create the model

```js
import { MambaModel } from 'mambacode.js';

const model = new MambaModel(device, {
  vocabSize : tokenizer.vocabSize,   // 151936
  dModel    : 512,   // hidden dimension
  numLayers : 8,     // number of Mamba blocks
  dState    : 16,    // SSM state size
  dConv     : 4,     // causal conv kernel width
  expand    : 2,     // inner expansion factor
});
```

> **Tip:** Start with the small configuration above (≈ 50 M parameters). Increase `dModel` or `numLayers` only if you have a discrete GPU with more VRAM.

---

### Step 6 — Load a pre-trained checkpoint

A randomly-initialised model generates gibberish. Load a pre-trained checkpoint first:

```js
// Option A: download from builderforce.ai
import { BuilderForceClient } from 'https://cdn.builderforce.ai/sdk/v1/client.js';

const client    = new BuilderForceClient({ apiKey: 'YOUR_API_KEY' });
const modelMeta = await client.models.get('mamba-coder-0.8b-base');
const response  = await fetch(modelMeta.downloadUrl);
const buffer    = await response.arrayBuffer();
await model.loadWeights(buffer);

// Option B: load a locally-hosted checkpoint file
const response = await fetch('/models/mamba-coder-checkpoint.bin');
const buffer   = await response.arrayBuffer();
await model.loadWeights(buffer);
```

---

### Step 7 — (Optional) Fine-tune on your code

This is the real power of MambaCode.js: you can teach the model about *your* private codebase without sending anything to a server.

```js
import { MambaTrainer } from 'mambacode.js';

// myCodeString can be any text, e.g. the contents of your project files
const myCodeString = `
function greet(name) {
  return \`Hello, \${name}!\`;
}
`;

const trainer = new MambaTrainer(model, tokenizer);
const losses  = await trainer.train(myCodeString, {
  learningRate : 1e-4,
  epochs       : 5,
  onEpochEnd   : (epoch, loss) =>
    console.log(`Epoch ${epoch}: loss = ${loss.toFixed(4)}`),
});

console.log('Training complete. Final loss:', losses.at(-1).toFixed(4));
```

For rapid adaptation with minimal compute, use **WSLA** mode (only updates the B and C matrices):

```js
await trainer.train(myCodeString, {
  learningRate : 1e-4,
  epochs       : 3,
  wsla         : true,   // faster, lower memory
});
```

---

### Step 8 — Generate code

```js
const prompt     = 'function fibonacci(';
const promptIds  = tokenizer.encode(prompt);
const outputIds  = await model.generate(promptIds, 200, { temperature: 0.8 });
const outputText = tokenizer.decode(outputIds);

console.log(prompt + outputText);
```

You should see the model continue the function — using everything it has learned from the pre-trained checkpoint *plus* whatever you trained it on in Step 7.

---

### Step 9 — Save your fine-tuned weights

After training, serialise the weights so you can reload them later without re-training:

```js
// Save via download link
const weightBuffer = await model.exportWeights();
const blob  = new Blob([weightBuffer], { type: 'application/octet-stream' });
const url   = URL.createObjectURL(blob);
const a     = document.createElement('a');
a.href      = url;
a.download  = 'mamba-finetuned.bin';
a.click();
URL.revokeObjectURL(url);
```

Load this file back in a future session using `model.loadWeights(buffer)` (Step 6, Option B).

---

## 7. The Complete Lifecycle at a Glance

```
Before MambaCode.js
────────────────────────────────────────────────────────────
  Download vocab files  ──►  vocab.json + merges.txt
  (from HuggingFace)         (served from your web server)

Using MambaCode.js
────────────────────────────────────────────────────────────
  1. initWebGPU()             Acquire GPUDevice
         │
         ▼
  2. tokenizer.load()         Load Qwen vocab + merges
         │
         ▼
  3. new MambaModel()         Create model with your config
         │
         ▼
  4. model.loadWeights()      Load pre-trained checkpoint
         │
         ▼
  5. trainer.train()          (Optional) Fine-tune on private code
         │                    Runs entirely on local GPU
         │                    No data leaves the browser
         ▼
  6. model.generate()         Generate code from a prompt
         │
         ▼
  7. tokenizer.decode()       Convert token IDs → text

After MambaCode.js
────────────────────────────────────────────────────────────
  8. model.exportWeights()    Save checkpoint for next session
         │
         ▼
  Share .bin with team  ──►  Team loads with model.loadWeights()
```

---

## 8. What Happens Next?

Once you have the basics working, here are some natural next steps:

### Build a code-completion UI

Wire the model to a `<textarea>` and call `model.generate()` on every keypress (or on a timer) to show real-time suggestions — entirely client-side.

### Evaluate the model quality

Use `trainer.evaluate()` to measure perplexity on a held-out code snippet. Lower perplexity means the model is more confident about the code it generates:

```js
const perplexity = await trainer.evaluate(heldOutCode);
console.log(`Perplexity: ${perplexity.toFixed(2)}`);
```

### Share fine-tuned weights with your team

Export the checkpoint and upload it via [builderforce.ai](https://builderforce.ai) Team Sharing, or your own file server. See the [Weight Lifecycle guide](./weight-lifecycle.md#7-sharing-weights-with-your-team) for code examples.

### Explore the WGSL kernels

If you want to go deeper, the `src/kernels/` directory contains hand-written WGSL shaders for every operation. Reading these is a great introduction to GPU programming for ML.

### Read the full API reference

The [API Reference](./api-reference.md) documents every exported class, interface, and function with TypeScript and JavaScript examples.

---

## 9. Using builderforce.ai

[**builderforce.ai**](https://builderforce.ai) is the platform built around MambaCode.js, designed to make on-device AI development accessible without requiring a machine-learning background.

### What builderforce.ai provides

| Feature | How it helps |
|---|---|
| **Model library** | Browse and download pre-trained Mamba checkpoints |
| **Fine-tune UI** | Upload your code files and fine-tune a model through a web interface — no JavaScript required |
| **Prompt playground** | Experiment with code-generation prompts against any model in your library |
| **Team sharing** | Share fine-tuned model checkpoints with colleagues |
| **Integration guides** | Step-by-step recipes for VSCode extensions, CI pipelines, and web apps |

### Getting started with builderforce.ai

1. **Visit [builderforce.ai](https://builderforce.ai)** and create a free account.
2. Navigate to the **Model Library** and download a starter Mamba-Coder checkpoint.
3. Drop the downloaded `.bin` file into your project and load it with `model.loadWeights()`.
4. Use the **Fine-tune UI** to upload your private code — the fine-tuning runs in your browser; builderforce.ai never receives the code itself.
5. Share the resulting checkpoint with your team through the **Team Sharing** panel.

---

## 10. Common Questions

**Q: Do I need to understand machine learning to use this?**
No. Follow the steps in [Section 6](#6-step-by-step-your-first-on-device-model) and you will have a working code model in minutes. The deeper concepts in Sections 1–3 are background reading, not prerequisites.

---

**Q: Does MambaCode.js send my code anywhere?**
No. Everything — tokenization, model weights, training, and inference — runs locally in your browser using WebGPU. The only network calls are the initial fetches for vocab files and model weights from URLs *you* choose.

---

**Q: What is the difference between Mamba and a Transformer (like GPT or Qwen)?**
Both are language models, but Mamba processes sequences in linear time (O(N)) instead of quadratic (O(N²)). This makes Mamba significantly faster for long contexts and more practical on consumer hardware. See [Section 3](#3-what-is-mamba-and-why-does-it-matter) for details.

---

**Q: Why Qwen3.5-Coder specifically?**
Qwen3.5-Coder-0.8B strikes a good balance between model capability and size. Its tokenizer vocabulary (151 936 tokens) is designed for code, handling identifiers, operators, and indentation efficiently. MambaCode.js uses the same tokenizer to maintain compatibility with the broader Qwen ecosystem.

---

**Q: Can I load Qwen model weights directly into MambaCode.js?**
No — Qwen is a Transformer and Mamba is an SSM. The architectures are fundamentally different and their weight matrices have incompatible shapes. What *is* shared is the tokenizer vocabulary. Use [builderforce.ai](https://builderforce.ai) to download pre-trained Mamba checkpoints, or train from scratch. See the [Weight Lifecycle guide](./weight-lifecycle.md#2-understanding-the-qwenmamba-relationship) for details.

---

**Q: Can I use a different vocabulary / model architecture?**
Yes, with some code changes. The `BPETokenizer` class accepts any `vocab.json` / `merges.txt` pair. The `MambaModel` config can be adjusted to match different layer counts and hidden dimensions. If you need help adapting the library, check the [builderforce.ai integration guides](https://builderforce.ai).

---

**Q: The model generates gibberish. What is wrong?**
A freshly-initialised model with random weights always generates gibberish. You need pre-trained weights — see [Step 6](#step-6--load-a-pre-trained-checkpoint). If you have loaded a checkpoint and still see poor output, verify the model config matches the one used when exporting the checkpoint.

---

*Back to [README](../README.md) · [Integration & Architecture](./integration-architecture.md) · [API Reference](./api-reference.md) · [Weight Lifecycle](./weight-lifecycle.md)*
