/**
 * ml/run.ts — the ONE runner: `runMlTask(id, input, ctx)`.
 *
 * Picks the first engine the inference mode allows that can answer (Local only:
 * local tiers; Hybrid: local first, then the host's cloud port; Specific: exactly
 * the named engine), returns `{ output, confidence, engine }`, and records the
 * decision so routing can be evaluated later. The cloud is a PORT the host
 * supplies — nothing here touches the network.
 */
import { extractFeatures, type TextEmbedder } from "./features.js";
import type { MlTaskRegistry } from "./registry.js";
import { engineKey, type InferenceMode, type MlEngineRef, type MlPrediction, type MlPredictor, type MlTask } from "./types.js";

/** Fitted local models, per (task id, engine key). */
export interface MlModelStore {
  get(taskId: string, engine: string): MlPredictor | undefined;
  set(taskId: string, engine: string, predictor: MlPredictor): void;
}

/** The plain in-process store. */
export class InMemoryMlModels implements MlModelStore {
  private readonly _m = new Map<string, MlPredictor>();
  get(taskId: string, engine: string): MlPredictor | undefined {
    return this._m.get(`${taskId}|${engine}`);
  }
  set(taskId: string, engine: string, predictor: MlPredictor): void {
    this._m.set(`${taskId}|${engine}`, predictor);
  }
}

/** A cloud engine, supplied by the host (Builderforce.ai passes its gateway client). */
export interface MlCloudPort {
  run(request: { taskId: string; engine: MlEngineRef; input: unknown }): Promise<MlPrediction>;
}

/** Why the runner moved past an engine. */
export type MlSkipReason = "unavailable" | "low-confidence" | "error";

/** One routing decision, recorded for eval. */
export interface MlDecision {
  taskId: string;
  mode: InferenceMode;
  /** The engine that answered (key), or null when none could. */
  engine: string | null;
  tier: "local" | "cloud" | null;
  confidence: number | null;
  /** Engines passed over before the answer, with why. */
  skipped: { engine: string; reason: MlSkipReason; detail?: string }[];
}

export interface MlDecisionSink {
  record(decision: MlDecision): void;
}

/** A bounded in-memory decision log with a per-task summary. */
export class MlDecisionLog implements MlDecisionSink {
  private readonly _d: MlDecision[] = [];
  constructor(readonly capacity = 500) {}

  record(decision: MlDecision): void {
    this._d.push(decision);
    if (this._d.length > this.capacity) this._d.splice(0, this._d.length - this.capacity);
  }

  list(): readonly MlDecision[] {
    return this._d;
  }

  /** Per task: how many runs, which engines answered, mean confidence, unanswered count. */
  summary(): Record<string, { runs: number; byEngine: Record<string, number>; unanswered: number; meanConfidence: number | null }> {
    const out: Record<string, { runs: number; byEngine: Record<string, number>; unanswered: number; meanConfidence: number | null }> = {};
    const sums: Record<string, { s: number; n: number }> = {};
    for (const d of this._d) {
      const t = (out[d.taskId] ??= { runs: 0, byEngine: {}, unanswered: 0, meanConfidence: null });
      t.runs++;
      if (d.engine == null) {
        t.unanswered++;
        continue;
      }
      t.byEngine[d.engine] = (t.byEngine[d.engine] ?? 0) + 1;
      const c = (sums[d.taskId] ??= { s: 0, n: 0 });
      c.s += d.confidence ?? 0;
      c.n++;
      t.meanConfidence = c.s / c.n;
    }
    return out;
  }
}

export interface MlRunContext {
  registry: MlTaskRegistry;
  mode: InferenceMode;
  /** For `specific`: the engine (ref or key) to use. */
  engine?: MlEngineRef | string;
  models?: MlModelStore;
  embedder?: TextEmbedder;
  cloud?: MlCloudPort;
  /** Hybrid only: a local answer below this confidence defers to the cloud when it can answer. */
  minConfidence?: number;
  decisions?: MlDecisionSink;
}

export interface MlResult<O> {
  output: O;
  confidence: number;
  engine: MlEngineRef;
}

/** No engine allowed by the mode could answer. */
export class MlEngineUnavailableError extends Error {
  constructor(
    readonly taskId: string,
    readonly decision: MlDecision,
  ) {
    super(`runMlTask(${taskId}): no engine could answer in '${decision.mode}' mode (${decision.skipped.map((s) => `${s.engine}: ${s.reason}`).join(", ") || "none allowed"})`);
    this.name = "MlEngineUnavailableError";
  }
}

/** The task's engines the mode allows, in the order the runner tries them. */
export function candidateEngines(task: MlTask<any, any>, mode: InferenceMode, specific?: MlEngineRef | string): MlEngineRef[] {
  if (mode === "local-only") return task.engines.filter((e) => e.tier === "local");
  if (mode === "hybrid") return [...task.engines.filter((e) => e.tier === "local"), ...task.engines.filter((e) => e.tier === "cloud")];
  if (specific == null) throw new Error(`runMlTask(${task.id}): 'specific' mode needs ctx.engine`);
  const key = typeof specific === "string" ? specific : engineKey(specific);
  const match = task.engines.filter((e) => engineKey(e) === key);
  if (match.length === 0) throw new Error(`runMlTask(${task.id}): engine '${key}' is not one of this task's engines`);
  return match;
}

/** Answer task `id` for `input` with the first engine the mode allows that can. */
export async function runMlTask<O = unknown, I = unknown>(id: string, input: I, ctx: MlRunContext): Promise<MlResult<O>> {
  const task = ctx.registry.require<I, O>(id);
  const engines = candidateEngines(task, ctx.mode, ctx.engine);
  const skipped: MlDecision["skipped"] = [];
  let features: Float32Array | null = null;
  let deferred: { ref: MlEngineRef; p: MlPrediction } | null = null;

  const finish = (ref: MlEngineRef, p: MlPrediction): MlResult<O> => {
    ctx.decisions?.record({ taskId: id, mode: ctx.mode, engine: engineKey(ref), tier: ref.tier, confidence: p.confidence, skipped });
    return { output: task.output.decode(p), confidence: p.confidence, engine: ref };
  };

  for (let i = 0; i < engines.length; i++) {
    const ref = engines[i]!;
    const key = engineKey(ref);
    if (ref.tier === "local") {
      const predictor = ctx.models?.get(id, key);
      if (!predictor) {
        skipped.push({ engine: key, reason: "unavailable" });
        continue;
      }
      features ??= await extractFeatures(task.features, input, ctx.embedder);
      const p = predictor.predict(features);
      const cloudLater = ctx.mode === "hybrid" && !!ctx.cloud && engines.slice(i + 1).some((e) => e.tier === "cloud");
      if (ctx.minConfidence != null && p.confidence < ctx.minConfidence && cloudLater) {
        skipped.push({ engine: key, reason: "low-confidence", detail: p.confidence.toFixed(3) });
        deferred ??= { ref, p };
        continue;
      }
      return finish(ref, p);
    }
    if (!ctx.cloud) {
      skipped.push({ engine: key, reason: "unavailable" });
      continue;
    }
    try {
      return finish(ref, await ctx.cloud.run({ taskId: id, engine: ref, input }));
    } catch (error) {
      skipped.push({ engine: key, reason: "error", detail: error instanceof Error ? error.message : String(error) });
    }
  }
  // The cloud could not answer after a low-confidence local one: the local answer stands.
  if (deferred) return finish(deferred.ref, deferred.p);
  const decision: MlDecision = { taskId: id, mode: ctx.mode, engine: null, tier: null, confidence: null, skipped };
  ctx.decisions?.record(decision);
  throw new MlEngineUnavailableError(id, decision);
}
