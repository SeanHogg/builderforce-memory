/**
 * tests/workflow.test.ts
 * Configurable workflows: templates, validation, custom workflows, and the
 * LLM-creation pipeline that outputs a deployable .evermind model.
 */

import "fake-indexeddb/auto";

import { EvermindModelPackage, BPETokenizer } from "@seanhogg/builderforce-memory-engine";
import {
  runWorkflow,
  validateWorkflow,
  defaultStepRegistry,
  WORKFLOW_TEMPLATES,
  getTemplate,
  cloneTemplate,
  AGENTIC_SEVEN_LAYER,
  TRAIN_LLM,
  type WorkflowConfig,
} from "../src/workflow/index.js";

describe("workflow — templates + registry", () => {
  test("ships the Agentic 7-layer and Create-an-LLM templates", () => {
    expect(WORKFLOW_TEMPLATES.map((t) => t.id)).toEqual(["agentic-seven-layer", "train-llm"]);
    expect(getTemplate("agentic-seven-layer")?.steps).toHaveLength(7);
    expect(getTemplate("nope")).toBeUndefined();
  });

  test("the registry exposes a step-type palette for a builder", () => {
    const types = defaultStepRegistry.types().map((t) => t.type);
    expect(types).toEqual(expect.arrayContaining(["foundation", "rag", "train-tokenizer", "train-model", "package"]));
  });

  test("validateWorkflow flags unknown types, duplicate ids, and empty workflows", () => {
    expect(validateWorkflow(AGENTIC_SEVEN_LAYER)).toEqual([]);
    const bad: WorkflowConfig = {
      id: "x",
      name: "x",
      steps: [
        { id: "a", type: "foundation" },
        { id: "a", type: "nope" },
      ],
    };
    const errs = validateWorkflow(bad);
    expect(errs.some((e) => /duplicate/.test(e.message))).toBe(true);
    expect(errs.some((e) => /unknown step type/.test(e.message))).toBe(true);
    expect(validateWorkflow({ id: "e", name: "e", steps: [] })[0]!.message).toMatch(/no steps/);
  });
});

describe("workflow — running", () => {
  test("the Agentic 7-layer template runs green", async () => {
    const r = await runWorkflow(AGENTIC_SEVEN_LAYER);
    if (!r.ok) throw new Error(`broke at ${r.firstFailure?.layer}: ${r.firstFailure?.error}`);
    expect(r.steps).toHaveLength(7);
    expect(r.steps.every((s) => s.status === "pass")).toBe(true);
  });

  test("an unknown step type fails its row (localized) without crashing the run", async () => {
    const r = await runWorkflow({ id: "c", name: "c", steps: [{ id: "s1", type: "does-not-exist" }] });
    expect(r.ok).toBe(false);
    expect(r.firstFailure?.id).toBe("s1");
    expect(r.firstFailure?.error).toMatch(/unknown step type/);
  });

  test("Create-an-LLM workflow trains + packages a deployable model (text I/O)", async () => {
    const r = await runWorkflow(TRAIN_LLM, {
      // keep the test quick
      registry: defaultStepRegistry,
    });
    if (!r.ok) throw new Error(`broke at step ${r.firstFailure?.id}: ${r.firstFailure?.error}`);
    expect(r.steps.map((s) => s.id)).toEqual(["tok", "model", "eval", "pkg"]);

    // The workflow OUTPUT is a portable .evermind artifact + its tokenizer.
    const blob = r.artifacts.evermind as ArrayBuffer;
    const tokDesc = r.artifacts.tokenizer as { vocab: Record<string, number>; merges: string[] };
    expect(blob.byteLength).toBeGreaterThan(0);
    expect(tokDesc.vocab).toBeDefined();

    // "Deploy" the created LLM: reconstruct model + tokenizer and generate text.
    const model = EvermindModelPackage.fromBlob(blob).loadLM();
    const tok = new BPETokenizer();
    tok.loadFromObjects(tokDesc.vocab, tokDesc.merges);
    const text = model.generateText("The", tok, { maxNewTokens: 4, temperature: 0 });
    expect(typeof text).toBe("string");
  }, 20000);

  test("a CUSTOM workflow (cloned + edited) trains a model with custom params", async () => {
    const custom = cloneTemplate("train-llm", {
      id: "my-custom",
      steps: [
        { id: "t", type: "train-tokenizer", params: { corpus: "alpha beta gamma. beta gamma delta. gamma delta alpha.", numMerges: 30 } },
        { id: "m", type: "train-model", params: { epochs: 15, dModel: 12, numLayers: 1 } },
        { id: "p", type: "package", params: { name: "mini" } },
      ],
    })!;
    expect(validateWorkflow(custom)).toEqual([]);
    const r = await runWorkflow(custom);
    if (!r.ok) throw new Error(`broke at ${r.firstFailure?.id}: ${r.firstFailure?.error}`);
    expect(r.steps.map((s) => s.id)).toEqual(["t", "m", "p"]);
    expect((r.artifacts.evermind as ArrayBuffer).byteLength).toBeGreaterThan(0);
  }, 20000);
});
