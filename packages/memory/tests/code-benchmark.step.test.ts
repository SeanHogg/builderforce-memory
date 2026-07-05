/**
 * tests/code-benchmark.step.test.ts
 *
 * The `code-benchmark` step scores held-out coding pass@1: it prompts the
 * TRAINED model on unseen tasks, executes each generated solution against its
 * cases, and passes a task only when ALL its cases pass. We drive it with a fake
 * model whose output is deterministic per prompt, so the test verifies the
 * scoring + gate logic (not a tiny model's luck).
 */

import { defaultStepRegistry } from "../src/workflow/registry.js";
import type { StackContext } from "../src/diagnostics/stack-diagnostic.js";

/** A model that returns a canned completion keyed by the prompt. */
function fakeModel(byPrompt: Record<string, string>) {
  return {
    generateText: (prompt: string) => byPrompt[prompt] ?? "",
  };
}

function ctxWith(bag: Record<string, unknown>): StackContext {
  return { bag, artifacts: {}, trace: [], now: () => 0 };
}

const add = "function add(a, b) { return a + b; }";
const brokenAdd = "function add(a, b) { return a - b; }";

test("scores pass@1 — a task passes only when ALL its cases pass", async () => {
  const model = fakeModel({
    "write add": add, // both cases pass → task passes
    "write sub": brokenAdd, // one case fails → task fails
  });
  const step = defaultStepRegistry.build({
    id: "cb",
    type: "code-benchmark",
    params: {
      tasks: [
        { prompt: "write add", cases: [{ call: "add(2,3)", expect: 5 }, { call: "add(0,0)", expect: 0 }] },
        { prompt: "write sub", cases: [{ call: "add(5,2)", expect: 7 }] },
      ],
    },
  });
  const ctx = ctxWith({ model, tokenizer: {} });
  const detail = await step.run(ctx);

  const report = ctx.bag.codeBenchmark as { pass1: number; passedTasks: number; totalTasks: number };
  expect(report.totalTasks).toBe(2);
  expect(report.passedTasks).toBe(1);
  expect(report.pass1).toBeCloseTo(0.5);
  expect(detail).toContain("pass@1 1/2");
});

test("minPass1 gate fails a model that can't code the held-out tasks", async () => {
  const model = fakeModel({ "write add": brokenAdd });
  const step = defaultStepRegistry.build({
    id: "cb",
    type: "code-benchmark",
    params: {
      minPass1: 0.9,
      tasks: [{ prompt: "write add", cases: [{ call: "add(2,3)", expect: 5 }] }],
    },
  });
  await expect(step.run(ctxWith({ model, tokenizer: {} }))).rejects.toThrow(/pass@1 0% below min 90%/);
});

test("requires a trained model and at least one held-out task", async () => {
  const noModel = defaultStepRegistry.build({ id: "cb", type: "code-benchmark", params: { tasks: [{ prompt: "x" }] } });
  await expect(noModel.run(ctxWith({}))).rejects.toThrow(/no trained model/);

  const noTasks = defaultStepRegistry.build({ id: "cb", type: "code-benchmark", params: {} });
  await expect(noTasks.run(ctxWith({ model: fakeModel({}), tokenizer: {} }))).rejects.toThrow(/no held-out tasks/);
});
