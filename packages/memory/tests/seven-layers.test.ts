/**
 * tests/seven-layers.test.ts
 *
 * End-to-end test of the seven-layer agent stack from the BuilderForce blog
 * ("The Agent Tech Stack: All Seven Layers").
 *
 * This consumes the SAME diagnostic the product surfaces behind a "Run Stack
 * Check" button (`runStackDiagnostic` / `buildEvermindStackSteps`) — the test and
 * the live execution-output timeline share one set of evaluators, so a passing
 * test means the button is green too, and a failing layer is localized identically.
 *
 *   L1 Foundation (EvermindLM+tokenizer) · L2 Orchestration · L3 Memory (cognition)
 *   L4 RAG (chunk+hybrid) · L5 Tools · L6 Observability · L7 Deployment (.evermind)
 *
 * L1/L3/L4/L7 use the real shipped components; L2/L5/L6 use the in-process
 * harness (their production forms live in the agent runtime).
 */

import "fake-indexeddb/auto";

import { runStackDiagnostic, buildEvermindStackSteps } from "../src/diagnostics/stack-diagnostic.js";

describe("seven-layer agent stack — end to end", () => {
  test("every layer passes in one run (and the order is L1→L3→L4→L5→L2→L6→L7)", async () => {
    const streamed: string[] = [];
    const result = await runStackDiagnostic(buildEvermindStackSteps(), {
      onStep: (s) => streamed.push(`${s.layer}:${s.status}`),
    });

    // Surface the breaking step in the failure message, exactly like the UI would.
    if (!result.ok) {
      throw new Error(`stack broke at ${result.firstFailure?.layer} (${result.firstFailure?.label}): ${result.firstFailure?.error}`);
    }

    expect(result.ok).toBe(true);
    expect(result.steps).toHaveLength(7);
    expect(result.steps.map((s) => s.layer)).toEqual(["L1", "L3", "L4", "L5", "L2", "L6", "L7"]);
    for (const s of result.steps) {
      expect(s.status).toBe("pass");
      expect(s.ms).toBeGreaterThanOrEqual(0);
    }
    // The onStep stream is the live execution-output the UI renders.
    expect(streamed).toEqual(["L1:pass", "L3:pass", "L4:pass", "L5:pass", "L2:pass", "L6:pass", "L7:pass"]);
  });

  test("a broken layer is localized (which step is breaking)", async () => {
    const steps = buildEvermindStackSteps();
    // Inject a failure after L1 to prove the diagnostic pinpoints the breaking step
    // and keeps reporting the rest (so the timeline is complete).
    steps.splice(1, 0, {
      id: "inject-fail",
      layer: "LX",
      label: "injected failure",
      run: async () => {
        throw new Error("boom");
      },
    });

    const result = await runStackDiagnostic(steps);
    expect(result.ok).toBe(false);
    expect(result.firstFailure?.id).toBe("inject-fail");
    expect(result.firstFailure?.error).toMatch(/boom/);
    expect(result.steps.find((s) => s.id === "inject-fail")?.status).toBe("fail");
    // The run continued past the failure — the timeline is complete, not truncated.
    expect(result.steps).toHaveLength(8);
  });
});
