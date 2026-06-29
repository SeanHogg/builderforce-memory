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

  test("the registry exposes a step-type palette for a builder (incl. the new diagnostics)", () => {
    const types = defaultStepRegistry.types().map((t) => t.type);
    expect(types).toEqual(
      expect.arrayContaining([
        "foundation", "rag", "train-tokenizer", "train-model", "package",
        "dataset-quality", "convergence", "generate-check", "roundtrip",
      ]),
    );
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

  test("Create-an-LLM runs the full generic diagnostic pipeline and ships a deployable model", async () => {
    const r = await runWorkflow(TRAIN_LLM, { registry: defaultStepRegistry });
    if (!r.ok) throw new Error(`broke at step ${r.firstFailure?.id}: ${r.firstFailure?.error}`);
    // Generic foundation: dataset gate → train → convergence → eval → generation → deploy round-trip.
    expect(r.steps.map((s) => s.id)).toEqual(["tok", "data", "model", "converge", "eval", "gen", "bench", "pkg", "export"]);
    expect(r.steps.every((s) => s.status === "pass")).toBe(true);

    // The new diagnostics actually asserted something meaningful:
    const byId = Object.fromEntries(r.steps.map((s) => [s.id, s]));
    expect(byId.data!.detail).toMatch(/words/);          // dataset quality measured the corpus
    expect(byId.converge!.detail).toMatch(/→/);          // convergence saw loss drop
    expect(byId.gen!.detail).toMatch(/deterministic/);   // generation reproducible
    expect(byId.bench!.detail).toMatch(/ppl .*bits\/tok.*top1/); // benchmark scored held-out text
    expect(byId.pkg!.detail).toMatch(/matches/);         // served == trained (deploy round-trip)

    // The workflow OUTPUT is a portable .evermind artifact + its tokenizer.
    const blob = r.artifacts.evermind as ArrayBuffer;
    const tokDesc = r.artifacts.tokenizer as { vocab: Record<string, number>; merges: string[] };
    expect(blob.byteLength).toBeGreaterThan(0);
    expect(tokDesc.vocab).toBeDefined();

    // "Deploy" the created LLM: reconstruct model + tokenizer and generate text.
    const served = EvermindModelPackage.fromBlob(blob);
    expect(served.validate().ok).toBe(true);
    const model = served.loadLM();
    const tok = new BPETokenizer();
    tok.loadFromObjects(tokDesc.vocab, tokDesc.merges);
    expect(typeof model.generateText("The", tok, { maxNewTokens: 4, temperature: 0 })).toBe("string");
  }, 30000);

  test("dataset-quality GATE rejects a degenerate corpus before wasting epochs", async () => {
    const r = await runWorkflow({
      id: "tiny", name: "tiny",
      steps: [
        { id: "tok", type: "train-tokenizer", params: { corpus: "a. a.", numMerges: 5 } },
        { id: "data", type: "dataset-quality" },
        { id: "model", type: "train-model" },
      ],
    });
    expect(r.ok).toBe(false);
    expect(r.firstFailure?.id).toBe("data");
    expect(r.firstFailure?.error).toMatch(/too small|too few|repetitive/);
  });

  test("convergence diagnostic guards a missing training history", async () => {
    const r = await runWorkflow({
      id: "noconv", name: "noconv",
      steps: [{ id: "converge", type: "convergence" }],
    });
    expect(r.ok).toBe(false);
    expect(r.firstFailure?.error).toMatch(/no training history/);
  });

  test("benchmark step requires a trained model", async () => {
    const r = await runWorkflow({
      id: "nobench", name: "nobench",
      steps: [{ id: "bench", type: "benchmark" }],
    });
    expect(r.ok).toBe(false);
    expect(r.firstFailure?.error).toMatch(/no trained model/);
  });

  test("benchmark step scores a trained model and its perplexity gate can fail a bad model", async () => {
    const steps = [
      { id: "tok", type: "train-tokenizer", params: { corpus: "alpha beta gamma. beta gamma delta. gamma delta alpha. delta alpha beta.", numMerges: 30 } },
      { id: "model", type: "train-model", params: { epochs: 15, dModel: 12, numLayers: 1 } },
    ];
    // No gate → passes and reports a scorecard.
    const ok = await runWorkflow({ id: "b1", name: "b1", steps: [...steps, { id: "bench", type: "benchmark" }] });
    if (!ok.ok) throw new Error(`broke at ${ok.firstFailure?.id}: ${ok.firstFailure?.error}`);
    const bench = ok.steps.find((s) => s.id === "bench")!;
    expect(bench.status).toBe("pass");
    expect(bench.detail).toMatch(/ppl .*bits\/tok.*top1.*top5/);

    // An impossible perplexity gate fails the step.
    const gated = await runWorkflow({
      id: "b2", name: "b2",
      steps: [...steps, { id: "bench", type: "benchmark", params: { maxPerplexity: 1.0 } }],
    });
    expect(gated.ok).toBe(false);
    expect(gated.firstFailure?.id).toBe("bench");
    expect(gated.firstFailure?.error).toMatch(/perplexity .* exceeds max/);

    // An impossible top-1 accuracy gate also fails the step.
    const top1 = await runWorkflow({
      id: "b3", name: "b3",
      steps: [...steps, { id: "bench", type: "benchmark", params: { minTop1: 0.999 } }],
    });
    expect(top1.ok).toBe(false);
    expect(top1.firstFailure?.id).toBe("bench");
    expect(top1.firstFailure?.error).toMatch(/top-1 accuracy .* below min/);

    // An eval corpus that yields no scorable tokens fails clearly.
    const empty = await runWorkflow({
      id: "b4", name: "b4",
      steps: [...steps, { id: "bench", type: "benchmark", params: { evalCorpus: "" } }],
    });
    expect(empty.ok).toBe(false);
    expect(empty.firstFailure?.id).toBe("bench");
    expect(empty.firstFailure?.error).toMatch(/no scorable tokens/);
  }, 30000);

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
