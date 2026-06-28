/**
 * workflow/templates.ts — starter workflows.
 *
 * A template is just a {@link WorkflowConfig}. The 7-layer agentic stack is one;
 * the LLM-creation pipeline is another. A user clones a template or authors their
 * own; a custom workflow can produce an `.evermind` model (the "train your LLM"
 * path) via the BUILD steps.
 */

import type { WorkflowConfig } from "./types.js";

/** The full Agentic 7-step stack (diagnostic). */
export const AGENTIC_SEVEN_LAYER: WorkflowConfig = {
  id: "agentic-seven-layer",
  name: "Agentic Stack — 7 Layers",
  description:
    "Runs every layer of the agent stack end to end: Foundation model, Orchestration, Memory, RAG, Tools, Observability, Deployment.",
  steps: [
    { id: "l1", type: "foundation" },
    { id: "l3", type: "memory" },
    { id: "l4", type: "rag" },
    { id: "l5", type: "tools" },
    { id: "l2", type: "orchestration" },
    { id: "l6", type: "observability" },
    { id: "l7", type: "deployment" },
  ],
};

/**
 * Generic default corpus so "Create an LLM" runs out-of-the-box. A real run
 * passes the user's own text as the `corpus` param on the train steps — the
 * pipeline is domain-agnostic (resume tailoring, support replies, code, prose…).
 * Distinct sentences keep the dataset-quality duplicate-ratio gate happy.
 */
const DEFAULT_CORPUS = [
  "BuilderForce orchestrates many agents through a planning loop.",
  "The memory layer stores facts as SSM embeddings for fast recall.",
  "Deployment runs on Cloudflare Workers and Durable Objects.",
  "Tools are gated by a capability registry the planner consults.",
  "Write-through cognition replaces a fact instead of appending a new copy.",
  "Retrieval fuses BM25 keyword search with dense semantic scoring.",
  "The runtime distills a frontier teacher into the on-device student.",
  "A workflow compiles into runnable steps and emits an execution timeline.",
  "Observability captures a span for every step so failures are localized.",
  "A trained model packages into a portable evermind artifact for serving.",
].join(" ");

/**
 * Create your own LLM — the GENERIC foundation pipeline (domain-agnostic).
 * Train a tokenizer + model on any corpus, with the full diagnostic gates:
 * dataset quality → train → convergence → evaluate → generation/determinism →
 * deploy round-trip (packages the trained model and proves the served copy
 * regenerates identical output). Pass your own text via the `corpus` param.
 */
export const TRAIN_LLM: WorkflowConfig = {
  id: "train-llm",
  name: "Create an LLM",
  description: "Train a custom EvermindLM on your corpus, validate it end to end, and package it as a portable .evermind artifact.",
  steps: [
    { id: "tok", type: "train-tokenizer", params: { corpus: DEFAULT_CORPUS, numMerges: 120 } },
    { id: "data", type: "dataset-quality" },
    { id: "model", type: "train-model", params: { corpus: DEFAULT_CORPUS, epochs: 50, dModel: 24, numLayers: 2, hiddenDim: 32 } },
    { id: "converge", type: "convergence" },
    { id: "eval", type: "evaluate" },
    { id: "gen", type: "generate-check" },
    { id: "pkg", type: "roundtrip", params: { name: "my-llm" } },
  ],
};

export const WORKFLOW_TEMPLATES: WorkflowConfig[] = [AGENTIC_SEVEN_LAYER, TRAIN_LLM];

export function getTemplate(id: string): WorkflowConfig | undefined {
  return WORKFLOW_TEMPLATES.find((t) => t.id === id);
}

/** Deep-clone a template so a user can edit it as a custom workflow. */
export function cloneTemplate(id: string, overrides: Partial<WorkflowConfig> = {}): WorkflowConfig | undefined {
  const t = getTemplate(id);
  if (!t) return undefined;
  return {
    ...t,
    id: overrides.id ?? `${t.id}-custom`,
    name: overrides.name ?? `${t.name} (custom)`,
    ...(overrides.description ? { description: overrides.description } : {}),
    steps: (overrides.steps ?? t.steps).map((s) => ({ ...s, params: { ...s.params } })),
  };
}
