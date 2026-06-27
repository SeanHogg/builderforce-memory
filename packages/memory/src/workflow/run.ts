/**
 * workflow/run.ts — compile, validate, and run a workflow config.
 */

import {
  runStackDiagnostic,
  type RunStackOptions,
  type StackDiagnosticResult,
  type StackStep,
} from "../diagnostics/stack-diagnostic.js";
import { defaultStepRegistry, StepTypeRegistry } from "./registry.js";
import { AGENTIC_SEVEN_LAYER } from "./templates.js";
import type { WorkflowConfig } from "./types.js";

/** Compile a workflow config to runnable steps (unknown types → failing steps). */
export function compileWorkflow(config: WorkflowConfig, registry: StepTypeRegistry = defaultStepRegistry): StackStep[] {
  return config.steps.map((s) => registry.build(s));
}

export interface WorkflowValidationError {
  stepId: string;
  type: string;
  message: string;
}

/** Static checks before running: non-empty, unique ids, known types. */
export function validateWorkflow(config: WorkflowConfig, registry: StepTypeRegistry = defaultStepRegistry): WorkflowValidationError[] {
  const errors: WorkflowValidationError[] = [];
  if (!config.steps || config.steps.length === 0) {
    errors.push({ stepId: "", type: "", message: "workflow has no steps" });
    return errors;
  }
  const seen = new Set<string>();
  for (const s of config.steps) {
    if (seen.has(s.id)) errors.push({ stepId: s.id, type: s.type, message: `duplicate step id "${s.id}"` });
    seen.add(s.id);
    if (!registry.has(s.type)) errors.push({ stepId: s.id, type: s.type, message: `unknown step type "${s.type}"` });
  }
  return errors;
}

export interface RunWorkflowOptions extends RunStackOptions {
  registry?: StepTypeRegistry;
}

/**
 * Run a workflow config and return its execution-output timeline + artifacts
 * (e.g. `result.artifacts.evermind` is the trained model for a build workflow).
 */
export function runWorkflow(config: WorkflowConfig, opts: RunWorkflowOptions = {}): Promise<StackDiagnosticResult> {
  const { registry = defaultStepRegistry, ...runOpts } = opts;
  return runStackDiagnostic(compileWorkflow(config, registry), runOpts);
}

/** Back-compat: the 7-layer template compiled to steps (consumed by the e2e test). */
export function buildEvermindStackSteps(): StackStep[] {
  return compileWorkflow(AGENTIC_SEVEN_LAYER);
}
