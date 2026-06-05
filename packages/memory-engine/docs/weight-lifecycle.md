# MambaCode.js — Weight Lifecycle Guide

This guide covers the **complete lifecycle of model weights** in MambaCode.js:

1. [Before Mamba — obtaining Qwen vocabulary files](#1-before-mamba--obtaining-qwen-vocabulary-files)
2. [Understanding the Qwen–Mamba relationship](#2-understanding-the-qwenmamba-relationship)
3. [Loading a pre-trained checkpoint](#3-loading-a-pre-trained-checkpoint)
4. [Fine-tuning on your own code](#4-fine-tuning-on-your-own-code)
5. [Exporting your fine-tuned weights](#5-exporting-your-fine-tuned-weights)
6. [Resuming from a checkpoint](#6-resuming-from-a-checkpoint)
7. [Sharing weights with your team](#7-sharing-weights-with-your-team)
8. [Using builderforce.ai for weight management](#8-using-builderforceai-for-weight-management)
9. [Weight file format reference](#9-weight-file-format-reference)

---

## 1. Before Mamba — obtaining Qwen vocabulary files

MambaCode.js uses the **Qwen3.5-Coder tokenizer vocabulary** (151 936 tokens). You need two files before you can tokenize any text:

| File | Description |
|---|---|
| `vocab.json` | Maps every token string to its integer ID |
| `merges.txt` | BPE merge rules, one per line, ordered by priority |

### Download from HuggingFace

The official vocabulary files are published alongside the Qwen3.5-Coder model family on HuggingFace.

**Via the HuggingFace website:**

1. Go to [https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B) (or any Qwen Coder variant — the tokenizer files are shared across the family).
2. Click the **Files and versions** tab.
3. Download `vocab.json` and `merges.txt`.

**Via the HuggingFace CLI:**

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-Coder-0.5B vocab.json merges.txt --local-dir ./qwen-vocab
```

**Direct URL (for scripting):**

```bash
curl -L -o vocab.json  "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B/resolve/main/vocab.json"
curl -L -o merges.txt  "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B/resolve/main/merges.txt"
```

> **Note:** `vocab.json` is approximately 2.8 MB and `merges.txt` is approximately 1.7 MB. Both should be served from your own web server or bundled with your application. Do not load them from third-party CDNs in production.

### Serve the files from your web server

Place `vocab.json` and `merges.txt` in your web server's public folder, for example:

```
your-app/
  public/
    vocab.json
    merges.txt
  src/
    ...
```

Then load them at runtime:

```js
import { BPETokenizer } from 'mambacode.js';

const tokenizer = new BPETokenizer();
await tokenizer.load('/vocab.json', '/merges.txt');
console.log('Vocabulary size:', tokenizer.vocabSize);  // 151936
```

---

## 2. Understanding the Qwen–Mamba relationship

It is important to understand **what is shared** between Qwen and MambaCode.js, and what is not.

| Aspect | Qwen3.5-Coder | MambaCode.js |
|---|---|---|
| **Architecture** | Transformer (attention-based) | Mamba SSM (state-space, recurrent) |
| **Weights** | Transformer weight matrices | Mamba weight matrices — **different shape and meaning** |
| **Tokenizer vocabulary** | 151 936 BPE tokens | **Same 151 936 BPE tokens** ✅ |
| **Token IDs** | Standard Qwen encoding | **Identical encoding** ✅ |
| **Context scaling** | O(N²) | O(N) |

### What this means in practice

- ✅ The **tokenizer** (`vocab.json` + `merges.txt`) is **fully compatible**. The same files power both Qwen and MambaCode.js.
- ❌ **Raw Qwen model weights cannot be loaded into MambaCode.js directly.** Qwen is a Transformer and Mamba is an SSM — the matrix shapes and semantics are completely different. There is no weight-conversion path.
- ✅ If you use **builderforce.ai**, you can download pre-trained Mamba checkpoints that have already been trained on the same code corpora as Qwen-family models. These files are in the MambaCode.js native format (`.bin`) and load directly with `model.loadWeights()`.

### Why use the Qwen tokenizer then?

Sharing the tokenizer means:
- MambaCode.js produces the same token IDs as Qwen models.
- Any prompt engineering or evaluation tooling built for Qwen works unchanged with MambaCode.js.
- Vocabulary size (151 936) is already optimised for code, handling identifiers, operators, and indentation efficiently.

---

## 3. Loading a pre-trained checkpoint

A freshly-created `MambaModel` has **random weights** and will generate nonsense. Always load a checkpoint before generating or fine-tuning.

### Option A — Download from builderforce.ai

[builderforce.ai](https://builderforce.ai) provides pre-trained Mamba checkpoints in the native `.bin` format:

```js
// 1. Get the download URL from the builderforce.ai API
import { BuilderForceClient } from 'https://cdn.builderforce.ai/sdk/v1/client.js';

const client    = new BuilderForceClient({ apiKey: 'YOUR_API_KEY' });
const modelMeta = await client.models.get('mamba-coder-0.8b-base');

// 2. Fetch and load
const response = await fetch(modelMeta.downloadUrl);
const buffer   = await response.arrayBuffer();
await model.loadWeights(buffer);
console.log('Checkpoint loaded ✅');
```

### Option B — Host your own checkpoint file

If you have a `.bin` file produced by `model.exportWeights()`:

```js
const response = await fetch('/models/mamba-coder-checkpoint.bin');
if (!response.ok) throw new Error(`Failed to fetch weights: ${response.statusText}`);
const buffer   = await response.arrayBuffer();
await model.loadWeights(buffer);
```

### Option C — Load from IndexedDB (offline-first)

```js
import { openDB } from 'idb';   // or use the raw IndexedDB API

const db     = await openDB('mamba-weights', 1);
const buffer = await db.get('checkpoints', 'latest');

if (buffer) {
  await model.loadWeights(buffer);
  console.log('Loaded checkpoint from IndexedDB ✅');
} else {
  console.log('No saved checkpoint found — using random weights.');
}
```

### Verifying the checkpoint loaded correctly

Run a quick sanity-check by generating a short sequence. With valid pre-trained weights the output should be recognisable code rather than random characters:

```js
const prompt    = 'function greet(name) {';
const ids       = tokenizer.encode(prompt);
const outputIds = await model.generate(ids, 50, { temperature: 0.5 });
console.log(tokenizer.decode(outputIds));
// Expected: continuation of a JS function, not random noise
```

---

## 4. Fine-tuning on your own code

Once a pre-trained checkpoint is loaded you can fine-tune the model on your private codebase. All training runs locally — no data leaves the browser.

### Basic fine-tuning

```js
import { MambaTrainer } from 'mambacode.js';

const trainer = new MambaTrainer(model, tokenizer);

// myCodeString can be the concatenated contents of your project files
const losses = await trainer.train(myCodeString, {
  learningRate : 1e-4,
  epochs       : 5,
  onEpochEnd   : (epoch, loss) =>
    console.log(`Epoch ${epoch}: loss = ${loss.toFixed(4)}`),
});

console.log('Fine-tuning complete. Final loss:', losses.at(-1).toFixed(4));
```

### WSLA — faster fine-tuning with fewer parameters

For rapid domain adaptation (e.g. learning your API conventions), use **WSLA** mode. Only the B and C matrices of each selective scan block are updated, which is much faster and uses less memory:

```js
await trainer.train(apiUsageExamples, {
  learningRate : 1e-4,
  epochs       : 3,
  wsla         : true,   // freeze all params except B and C
});
```

### Monitoring training

```js
const losses = await trainer.train(myCode, {
  epochs     : 10,
  onEpochEnd : (epoch, loss) => {
    console.log(`Epoch ${epoch}: loss = ${loss.toFixed(4)}`);
    updateProgressBar(epoch / 10);
  },
});
```

A healthy training run will show the loss decreasing over epochs. If loss stays flat or increases, try:
- Reducing `learningRate` (e.g. `1e-5`)
- Increasing `epochs`
- Checking that the input text contains meaningful code

---

## 5. Exporting your fine-tuned weights

After training, serialise the weights so you can reload them later without re-training:

### Save via download link (simplest)

```js
const buffer = await model.exportWeights();
const blob   = new Blob([buffer], { type: 'application/octet-stream' });
const url    = URL.createObjectURL(blob);
const a      = document.createElement('a');
a.href       = url;
a.download   = 'mamba-finetuned.bin';
a.click();
URL.revokeObjectURL(url);
```

### Save to IndexedDB (offline-first / no user prompt)

```js
const buffer = await model.exportWeights();
const db     = await openDB('mamba-weights', 1, {
  upgrade(db) { db.createObjectStore('checkpoints'); }
});
await db.put('checkpoints', buffer, 'latest');
console.log('Checkpoint saved to IndexedDB ✅');
```

### Save via File System Access API (persistent file)

```js
const buffer  = await model.exportWeights();
const handle  = await window.showSaveFilePicker({
  suggestedName: 'mamba-finetuned.bin',
  types: [{ description: 'Mamba weight file', accept: { 'application/octet-stream': ['.bin'] } }],
});
const writable = await handle.createWritable();
await writable.write(buffer);
await writable.close();
console.log('Checkpoint saved ✅');
```

---

## 6. Resuming from a checkpoint

The typical workflow across multiple sessions:

```js
// --- Session 1: initial fine-tuning ---
const { device }  = await initWebGPU();
const tokenizer   = new BPETokenizer();
await tokenizer.load('/vocab.json', '/merges.txt');
const model       = new MambaModel(device, config);

// Load base checkpoint
const baseWeights = await fetch('/models/mamba-coder-base.bin').then(r => r.arrayBuffer());
await model.loadWeights(baseWeights);

// Fine-tune
const trainer = new MambaTrainer(model, tokenizer);
await trainer.train(myCodeString, { epochs: 5 });

// Save checkpoint
const finetuned = await model.exportWeights();
await saveToIndexedDB(finetuned, 'checkpoint-v1');

// --- Session 2: resume and continue fine-tuning ---
const checkpointV1 = await loadFromIndexedDB('checkpoint-v1');
await model.loadWeights(checkpointV1);

await trainer.train(moreCodeString, { epochs: 3 });

const checkpointV2 = await model.exportWeights();
await saveToIndexedDB(checkpointV2, 'checkpoint-v2');
```

> **Tip:** Tag checkpoints with a timestamp or version number so you can roll back if a fine-tuning run degrades quality.

---

## 7. Sharing weights with your team

Fine-tuned checkpoints can be shared with colleagues. Because the weights encode only the model's learned patterns — not the training data itself — sharing a checkpoint does **not** expose your private code.

### Via builderforce.ai Team Sharing

1. Export the checkpoint with `model.exportWeights()`.
2. Navigate to [builderforce.ai](https://builderforce.ai) and open the **Team Sharing** panel.
3. Upload the `.bin` file. Teammates can download it directly from the platform.
4. On the teammate's machine: fetch the download URL and call `model.loadWeights()`.

### Via your own file server or object storage

```js
// Uploader
const buffer = await model.exportWeights();
const form   = new FormData();
form.append('file', new Blob([buffer]), 'checkpoint.bin');
await fetch('/api/checkpoints', { method: 'POST', body: form });

// Downloader (teammate)
const response = await fetch('/api/checkpoints/latest');
const buffer   = await response.arrayBuffer();
await model.loadWeights(buffer);
```

---

## 8. Using builderforce.ai for weight management

[**builderforce.ai**](https://builderforce.ai) is the platform built around MambaCode.js, designed to make on-device AI development accessible without requiring a machine-learning background.

| Feature | How it helps |
|---|---|
| **Model library** | Browse and download pre-trained Mamba checkpoints |
| **Fine-tune UI** | Upload your code files and fine-tune through a web interface — no code required |
| **Prompt playground** | Experiment with code-generation prompts against any model in your library |
| **Team sharing** | Share fine-tuned model checkpoints with colleagues |
| **Integration guides** | Step-by-step recipes for VSCode extensions, CI pipelines, and web apps |

### Getting started with builderforce.ai

1. Visit [builderforce.ai](https://builderforce.ai) and create a free account.
2. Navigate to the **Model Library** and download a starter Mamba-Coder checkpoint.
3. Drop the downloaded `.bin` file into your project and load it with `model.loadWeights()`.
4. (Optional) Use the **Fine-tune UI** to upload your private code — fine-tuning runs in your browser; builderforce.ai never receives the code itself.
5. Share the resulting checkpoint with your team through the **Team Sharing** panel.

---

## 9. Weight file format reference

Weight files produced by `model.exportWeights()` use a simple binary format:

```
Offset          Size            Field
-------         -------         ------
0               4 bytes         Magic: uint32 = 0x4D424A53 ('MBJS')
4               4 bytes         Version: uint32 = 1
8               4 bytes         nParams: uint32 (number of parameter tensors)
12              4 × nParams     numel[i]: uint32 — element count of parameter i
12 + 4×nParams  4 × Σ numel[i]  Float32 data for each parameter, concatenated in order
```

The parameter order matches the output of `model.parameters()`:

```
1. embedding          (vocabSize × dModel floats)
2. block0.in_proj_w   (...)
3. block0.conv1d_w    (...)
   ...
N. final_norm         (dModel floats)
```

> ⚠️ **Compatibility note:** The parameter layout depends on the model configuration. A checkpoint saved from a model with `dModel=512, numLayers=8` **cannot** be loaded into a model with a different configuration.

---

*Back to [README](../README.md) · [Getting Started](./getting-started.md) · [API Reference](./api-reference.md)*
