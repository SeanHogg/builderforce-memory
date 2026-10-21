# ML task catalog — non-LLM learning for Evermind

| | |
|---|---|
| Status | Approved 2026-10-20 — building |
| Package | `packages/memory-engine` (new `src/ml/`) |
| Why here | Evermind's private model, facts, experience and training already live in this repo; Synapse (`builderforce-inference/crates/bf-memory`) "keeps nothing of Evermind itself". |

## 1. The goal

Make Evermind better at learning on the person's own machine, without a GPU and
without an LLM call, by giving it small, purpose-built models for bounded questions:

- **What to learn from.** Score, de-duplicate and filter the experience corpus before
  `adaptAndDiff` fits a private copy, so the locally trained model learns from good,
  varied examples instead of every transcript.
- **What happened.** Classify turns and runs (task kind, outcome, intent), detect
  anomalies (a run that cost or failed far outside its normal range), and cluster
  similar episodes into one procedure.
- **What will happen.** Predict whether a turn will succeed on a given local model, so
  the Hybrid router keeps a turn local when it can and sends it to the cloud when it
  can't (Builderforce.ai consumes this).

## 2. What already exists, and what this extracts

| Piece | Where | Role in the catalog |
|---|---|---|
| `EvermindTextEmbedder` | `lm/text_embedder.ts` | Text → vector on the CPU; the feature extractor for every text task. No GPU, no network. |
| `LimbicModel` + `LimbicTrainer` | `limbic/` | **The pattern to extract**: a small head over an experience embedding, trained locally with AdamW on the GPU when there is one and on the CPU otherwise. Affect regression becomes the catalog's first registered task — not a second copy beside it. |
| `cpu_ops`, `autograd`, AdamW | `training/` | The training math the heads reuse. |
| `experience/corpus.ts` | `experience/` | Renders episodes and runs as text — the input to every experience task. |
| `adaptAndDiff`, `meanEvalLoss` | `evermind/` | The ONE training recipe; the catalog feeds it curated data, and never adds a second recipe. |

## 3. Contract

```ts
/** One bounded question Evermind can answer without an LLM. DATA, not a class tree. */
export interface MlTask<I, O> {
  kind: MlTaskKind;                 // 'classify' | 'regress' | 'cluster' | 'anomaly' | 'rank'
  id: string;                       // 'experience.quality', 'limbic.affect', 'run.local-success', …
  features: FeatureSpec<I>;         // how an input becomes a vector (embedder, numeric fields, both)
  engines: readonly MlEngineRef[];  // in preference order, filtered by the inference mode
  train?: TrainSpec;                // absent = rule-based or pre-trained only
  output: OutputSpec<O>;            // label set / range / cluster id, + calibrated confidence
}

export type MlEngineRef =
  | { tier: 'local'; engine: 'classical'; algorithm: ClassicalAlgorithm }   // pure TS, everywhere
  | { tier: 'local'; engine: 'head'; head: 'mlp' | 'linear' }               // LimbicModel-style head
  | { tier: 'local'; engine: 'onnx'; model: string }                        // transformers.js, WASM CPU or WebGPU
  | { tier: 'cloud'; engine: 'workers-ai'; model: string };                 // e.g. @cf/baai/bge-m3

export type ClassicalAlgorithm =
  | 'knn' | 'naive-bayes' | 'logistic' | 'linear' | 'kmeans'
  | 'robust-z' | 'ewma' | 'isolation-forest' | 'gbm-small';
```

- **One registry** (`ml/registry.ts`), keyed by `id`. A new task is a new row, not a
  new module. `limbic.affect` is registered by `limbic/` itself.
- **One runner** (`ml/run.ts`): `runMlTask(id, input, ctx)` picks the first engine the
  inference mode allows (Local only: local tiers only; Hybrid: local then cloud;
  Specific: the named engine), returns `{ output, confidence, engine }`, and records
  the decision for eval.
- **One trainer seam** (`ml/train.ts`): heads train through the extracted
  `LimbicTrainer` logic (renamed `HeadTrainer`, with `LimbicTrainer` a thin
  registration over it); classical models fit in closed form or by small iteration.
  Trained state is stored with the Evermind package (`evermind/package.ts` manifest
  gains `mlHeads`), so it travels with the person's model and merges like it.
- **Pure and portable**: no `vscode`, no DOM, no network in `ml/`; a cloud engine is
  a port the host supplies (Builderforce.ai passes its gateway client).

## 4. First tasks (in order)

1. **`limbic.affect`** — register the existing affect head through the catalog (no
   behaviour change). This proves the extraction.
2. **`experience.quality`** (classify, kNN/logistic over embeddings + run outcome
   fields) and **`experience.dedupe`** (cluster, k-means/cosine threshold) — applied in
   front of `adaptAndDiff` so local training sees curated, non-redundant examples.
   Measure with `meanEvalLoss` before/after on held-out experience.
3. **`run.anomaly`** (robust-z + EWMA over cost, duration, tool-failure count) —
   flags runs to exclude from training and to surface as Insights.
4. **`run.local-success`** (logistic / small GBM over turn features + the local
   model's served context) — exported for the Hybrid router in Builderforce.ai.
5. Later, behind the ONNX engine: OCR and image labels for attachments, sentiment for
   feedback — each a row with a WASM-CPU engine first.

## 5. Not in scope

Training an LLM differently (`adaptAndDiff` stays the one recipe), server-side model
hosting, any task whose output is free text (that's the LLM's job).

## 6. Operator decisions (2026-10-20)

- **Trained heads federate.** A head's trained state is a diff like an LM delta: it rides
  the existing delta wire (`evermind/delta_wire.ts`) and merges by FedAvg
  (`evermind/merge.ts`), gated by `meanEvalLoss`-style held-out evaluation before and
  after the merge — no second merge path.
- **`run.local-success` learns from the cloud.** In Hybrid, when a turn fell back to (or
  was also served by) the cloud, that outcome is a training label for the local-success
  predictor, alongside local signals.
