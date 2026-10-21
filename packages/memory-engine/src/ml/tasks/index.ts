/**
 * ml/tasks — the built-in catalog rows (docs/ml-task-catalog.md §4), in order.
 * `limbic.affect` is defined by limbic/ itself; this only lists it.
 */
import { LIMBIC_AFFECT_TASK } from "../../limbic/limbic_task.js";
import { MlTaskRegistry } from "../registry.js";
import type { MlTask } from "../types.js";
import { EXPERIENCE_DEDUPE_TASK, EXPERIENCE_QUALITY_TASK } from "./experience.js";
import { LOCAL_SUCCESS_TASK } from "./local_success.js";
import { RUN_ANOMALY_TASK } from "./run_anomaly.js";

/** The built-in rows, in catalog order. */
export const BUILTIN_ML_TASKS: readonly MlTask<any, any>[] = [
  LIMBIC_AFFECT_TASK,
  EXPERIENCE_QUALITY_TASK,
  EXPERIENCE_DEDUPE_TASK,
  RUN_ANOMALY_TASK,
  LOCAL_SUCCESS_TASK,
];

/** A registry holding every built-in row (add host rows with `register`). */
export function createMlRegistry(): MlTaskRegistry {
  const r = new MlTaskRegistry();
  for (const t of BUILTIN_ML_TASKS) r.register(t);
  return r;
}

export * from "./experience.js";
export * from "./run_anomaly.js";
export * from "./local_success.js";
