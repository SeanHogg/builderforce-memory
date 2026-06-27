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

/** Create your own LLM: train a tokenizer + model on a corpus, evaluate, package. */
export const TRAIN_LLM: WorkflowConfig = {
  id: "train-llm",
  name: "Create an LLM",
  description: "Train a custom EvermindLM on your corpus and package it as a portable .evermind artifact.",
  steps: [
    { id: "tok", type: "train-tokenizer", params: { numMerges: 80 } },
    { id: "model", type: "train-model", params: { epochs: 40, dModel: 16, numLayers: 2 } },
    { id: "eval", type: "evaluate" },
    { id: "pkg", type: "package", params: { name: "my-llm" } },
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
