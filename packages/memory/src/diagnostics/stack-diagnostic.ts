/**
 * stack-diagnostic.ts — the generic workflow runner + execution-output types.
 *
 * A workflow is an ordered list of {@link StackStep}s. `runStackDiagnostic`
 * executes them, capturing per-step pass/fail + timing + error and streaming a
 * callback — so the result is an "execution output" timeline a UI renders: run a
 * workflow, watch each step, see exactly which one breaks.
 *
 * The concrete steps (the 7 agent-stack layers, the LLM-creation pipeline) and
 * the configurable workflow/template layer live in `../workflow/` and compile
 * down to `StackStep[]` runnable here. Steps share a {@link StackContext}: `bag`
 * is scratch threaded across steps; `artifacts` is the workflow's OUTPUT (e.g. a
 * trained `.evermind` blob).
 */

export type StepStatus = "pass" | "fail" | "skip";

/** One step's result — the shape an execution-output timeline renders per row. */
export interface StackStepResult {
  id: string;
  /** Grouping tag, e.g. "L1".."L7" for the agent-stack layers, or "BUILD". */
  layer: string;
  label: string;
  status: StepStatus;
  ms: number;
  /** Short human detail on success (e.g. "retrieved 3 chunks"). */
  detail?: string;
  /** Failure message when status === "fail". */
  error?: string;
}

/** Shared state threaded across steps. */
export interface StackContext {
  /** Scratch populated/consumed by steps (e.g. the model L1 builds, L7 deploys). */
  bag: Record<string, unknown>;
  /** Workflow OUTPUTS — what the run produces (e.g. `artifacts.evermind` = blob). */
  artifacts: Record<string, unknown>;
  /** Prior step results so far — this IS the observability trace. */
  trace: StackStepResult[];
  /** Optional IDBFactory (Node/fake-indexeddb); browser uses global indexedDB. */
  idbFactory?: unknown;
  now: () => number;
}

export interface StackStep {
  id: string;
  layer: string;
  label: string;
  /** Throw to fail the step; optionally return a human detail string. */
  run: (ctx: StackContext) => Promise<string | void>;
}

export interface RunStackOptions {
  /** Streamed as each step settles — for live execution-output rendering. */
  onStep?: (r: StackStepResult) => void;
  /** Injectable clock for deterministic timing in tests. Default Date.now. */
  now?: () => number;
  /** IDBFactory for Node; omit in the browser. */
  idbFactory?: unknown;
}

export interface StackDiagnosticResult {
  ok: boolean;
  steps: StackStepResult[];
  /** The first step that failed (the breaking point), if any. */
  firstFailure?: StackStepResult;
  /** Outputs produced by the run (e.g. `artifacts.evermind` for a build workflow). */
  artifacts: Record<string, unknown>;
  totalMs: number;
}

/**
 * Run the steps in order, capturing each result. Never throws — a step that
 * throws becomes a `fail` row, and the run continues so the caller sees the full
 * timeline (and exactly where it broke). Pass `onStep` to stream rows live.
 */
export async function runStackDiagnostic(
  steps: StackStep[],
  opts: RunStackOptions = {},
): Promise<StackDiagnosticResult> {
  const now = opts.now ?? (() => Date.now());
  const ctx: StackContext = { bag: {}, artifacts: {}, trace: [], idbFactory: opts.idbFactory, now };
  const results: StackStepResult[] = [];
  let firstFailure: StackStepResult | undefined;
  const start = now();

  for (const step of steps) {
    const t0 = now();
    let result: StackStepResult;
    try {
      const detail = await step.run(ctx);
      result = { id: step.id, layer: step.layer, label: step.label, status: "pass", ms: now() - t0, ...(detail ? { detail } : {}) };
    } catch (e) {
      result = { id: step.id, layer: step.layer, label: step.label, status: "fail", ms: now() - t0, error: e instanceof Error ? e.message : String(e) };
      firstFailure ??= result;
    }
    results.push(result);
    ctx.trace.push(result);
    opts.onStep?.(result);
  }

  return { ok: !firstFailure, steps: results, ...(firstFailure ? { firstFailure } : {}), artifacts: ctx.artifacts, totalMs: now() - start };
}
