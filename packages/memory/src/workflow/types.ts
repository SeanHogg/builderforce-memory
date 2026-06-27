/**
 * workflow/types.ts — the configurable workflow definition.
 *
 * A workflow is data: an ordered list of step configs, each naming a registered
 * step `type` (e.g. "foundation", "train-model") plus per-step `params`. Templates
 * (the Agentic 7-layer stack, an LLM-creation pipeline) are just `WorkflowConfig`
 * values; a user's custom workflow is one they author. Configs compile to runnable
 * `StackStep[]` via the step registry.
 */

import type { StackStep } from "../diagnostics/stack-diagnostic.js";

export interface WorkflowStepConfig {
  /** Unique id within the workflow (also the execution-output row id). */
  id: string;
  /** Registered step type, e.g. "foundation" | "train-tokenizer" | "package". */
  type: string;
  /** Display label override (defaults to the type's built-in label). */
  label?: string;
  /** Per-step parameters consumed by the step factory (e.g. corpus, epochs). */
  params?: Record<string, unknown>;
}

export interface WorkflowConfig {
  id: string;
  name: string;
  description?: string;
  steps: WorkflowStepConfig[];
}

/** Metadata a registered step type advertises (for builders/validation/UI). */
export interface StepTypeInfo {
  type: string;
  /** Grouping tag used in the execution-output timeline (e.g. "L1", "BUILD"). */
  layer: string;
  label: string;
  /** One-line description for a workflow-builder palette. */
  description: string;
}

/** Builds a runnable step from its config. Throw inside `step.run` to fail it. */
export type StepFactory = (cfg: WorkflowStepConfig) => StackStep;
