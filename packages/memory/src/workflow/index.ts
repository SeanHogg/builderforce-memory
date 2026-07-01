/**
 * Configurable agent/LLM workflows — templates, a step-type registry, and a
 * runner that produces an execution-output timeline (+ a trained `.evermind`
 * artifact for build workflows).
 */

export type { WorkflowConfig, WorkflowStepConfig, StepTypeInfo, StepFactory } from "./types.js";
export { StepTypeRegistry, createDefaultRegistry, defaultStepRegistry } from "./registry.js";
export { BUILTIN_STEPS } from "./steps.js";
export {
  AGENTIC_SEVEN_LAYER,
  TRAIN_LLM,
  TEACH_CODE,
  WORKFLOW_TEMPLATES,
  getTemplate,
  cloneTemplate,
} from "./templates.js";
export {
  analyzeCode,
  runJsCases,
  type CodeAnalysis,
  type CodeCase,
  type CodeEvalResult,
} from "./code-eval.js";
export {
  compileWorkflow,
  validateWorkflow,
  runWorkflow,
  buildEvermindStackSteps,
  type WorkflowValidationError,
  type RunWorkflowOptions,
} from "./run.js";
