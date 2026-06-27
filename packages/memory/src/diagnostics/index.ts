/**
 * Diagnostics — the generic workflow runner + execution-output types.
 * The configurable workflow / template layer lives in `../workflow/`.
 */

export { runStackDiagnostic } from "./stack-diagnostic.js";
export type {
  StackStep,
  StackStepResult,
  StackContext,
  StackDiagnosticResult,
  RunStackOptions,
  StepStatus,
} from "./stack-diagnostic.js";
