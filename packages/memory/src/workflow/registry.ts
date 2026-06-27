/**
 * workflow/registry.ts — the step-type registry.
 *
 * Maps a step `type` → its factory + metadata. Built-ins are pre-registered;
 * callers (e.g. the agent runtime) can `register` real implementations for the
 * layers they own. Unknown types compile to a failing step so a bad workflow is
 * localized in the timeline rather than crashing the run.
 */

import { BUILTIN_STEPS } from "./steps.js";
import type { StepFactory, StepTypeInfo, WorkflowStepConfig } from "./types.js";
import type { StackStep } from "../diagnostics/stack-diagnostic.js";

export class StepTypeRegistry {
  private readonly map = new Map<string, { info: StepTypeInfo; factory: StepFactory }>();

  register(info: StepTypeInfo, factory: StepFactory): this {
    this.map.set(info.type, { info, factory });
    return this;
  }

  has(type: string): boolean {
    return this.map.has(type);
  }

  /** Compile one step config to a runnable step (failing step for unknown types). */
  build(cfg: WorkflowStepConfig): StackStep {
    const entry = this.map.get(cfg.type);
    if (!entry) {
      return {
        id: cfg.id,
        layer: "?",
        label: cfg.label ?? cfg.type,
        run: async () => {
          throw new Error(`unknown step type "${cfg.type}"`);
        },
      };
    }
    return entry.factory(cfg);
  }

  /** All registered step types — a workflow-builder palette reads this. */
  types(): StepTypeInfo[] {
    return [...this.map.values()].map((e) => e.info);
  }
}

export function createDefaultRegistry(): StepTypeRegistry {
  const r = new StepTypeRegistry();
  for (const { info, factory } of Object.values(BUILTIN_STEPS)) r.register(info, factory);
  return r;
}

/** Shared registry seeded with the built-in step types. */
export const defaultStepRegistry = createDefaultRegistry();
