/**
 * Diagnostics — runnable health checks shaped as execution-output timelines.
 */

export { runStackDiagnostic, buildEvermindStackSteps } from "./stack-diagnostic.js";
export type {
  StackStep,
  StackStepResult,
  StackContext,
  StackDiagnosticResult,
  RunStackOptions,
  StepStatus,
} from "./stack-diagnostic.js";
