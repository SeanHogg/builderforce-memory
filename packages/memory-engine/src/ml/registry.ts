/**
 * ml/registry.ts — the ONE registry of ML tasks, keyed by task id.
 *
 * A new task is a new row, not a new module. The registry only stores and looks
 * up rows; running lives in `run.ts`, training in `train.ts`.
 */
import { ML_TASK_ID_RE, type MlTask } from "./types.js";

export class MlTaskRegistry {
  private readonly _tasks = new Map<string, MlTask<any, any>>();

  /** Register a row. Throws on a malformed id, a duplicate id, or a row with no engine. */
  register<I, O>(task: MlTask<I, O>): MlTask<I, O> {
    if (!ML_TASK_ID_RE.test(task.id)) throw new Error(`MlTaskRegistry: malformed task id '${task.id}'`);
    if (this._tasks.has(task.id)) throw new Error(`MlTaskRegistry: task '${task.id}' is already registered`);
    if (task.engines.length === 0) throw new Error(`MlTaskRegistry: task '${task.id}' lists no engine`);
    if (task.kind === "classify" && !(task.output.labels && task.output.labels.length >= 2)) {
      throw new Error(`MlTaskRegistry: classify task '${task.id}' needs at least two output labels`);
    }
    this._tasks.set(task.id, task);
    return task;
  }

  /** The row for `id`, or undefined. */
  get<I = unknown, O = unknown>(id: string): MlTask<I, O> | undefined {
    return this._tasks.get(id) as MlTask<I, O> | undefined;
  }

  /** The row for `id`; throws when it is not registered. */
  require<I = unknown, O = unknown>(id: string): MlTask<I, O> {
    const t = this.get<I, O>(id);
    if (!t) throw new Error(`MlTaskRegistry: unknown task '${id}'`);
    return t;
  }

  has(id: string): boolean {
    return this._tasks.has(id);
  }

  /** Every row, in registration order. */
  list(): MlTask<any, any>[] {
    return [...this._tasks.values()];
  }
}
