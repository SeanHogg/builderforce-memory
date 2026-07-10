/**
 * tests/workflow.test.ts
 * Configurable workflows: templates, validation, custom workflows, and the
 * LLM-creation pipeline that outputs a deployable .evermind model.
 */

import "fake-indexeddb/auto";

import { EvermindModelPackage, BPETokenizer, EvermindLM, exportSafetensors } from "@seanhogg/builderforce-memory-engine";
import {
  runWorkflow,
  validateWorkflow,
  defaultStepRegistry,
  WORKFLOW_TEMPLATES,
  getTemplate,
  cloneTemplate,
  selectBestCandidate,
  stripCodeFences,
  AGENTIC_SEVEN_LAYER,
  TRAIN_LLM,
  type WorkflowConfig,
} from "../src/workflow/index.js";

describe("workflow — templates + registry", () => {
  test("ships the Agentic 7-layer, Create-an-LLM, and Teach-Code templates", () => {
    expect(WORKFLOW_TEMPLATES.map((t) => t.id)).toEqual(["agentic-seven-layer", "train-llm", "teach-code"]);
    expect(getTemplate("agentic-seven-layer")?.steps).toHaveLength(7);
    expect(getTemplate("teach-code")?.steps.some((s) => s.type === "distill-corpus")).toBe(true);
    expect(getTemplate("nope")).toBeUndefined();
  });

  test("the registry exposes a step-type palette for a builder (incl. the new diagnostics)", () => {
    const types = defaultStepRegistry.types().map((t) => t.type);
    expect(types).toEqual(
      expect.arrayContaining([
        "foundation", "rag", "train-tokenizer", "train-model", "package",
        "dataset-quality", "convergence", "generate-check", "roundtrip",
        "distill-corpus", "code-parse-check", "code-eval", "import-model",
        "video-train", "video-roundtrip",
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

  test("video-train + video-roundtrip: fit codec, train EvermindLM, gate reconstruction", async () => {
    // Small everything so it runs fast; synthetic clips are generated by the step.
    const params = { height: 8, width: 8, channels: 3, patch: 4, levels: 2, codebookSize: 16, clips: 3, frames: 3 };
    const r = await runWorkflow({
      id: "vid", name: "vid",
      steps: [
        { id: "vtrain", type: "video-train", params: { ...params, epochs: 20, dModel: 24 } },
        { id: "vround", type: "video-roundtrip", params },
      ],
    });
    if (!r.ok) throw new Error(`broke at ${r.firstFailure?.id}: ${r.firstFailure?.error}`);
    const byId = Object.fromEntries(r.steps.map((s) => [s.id, s]));
    expect(byId.vtrain!.detail).toMatch(/codec MSE .*trained .*epochs/);
    expect(byId.vround!.detail).toMatch(/recon MSE .*frames round-trip clean/);
  }, 30000);

  test("video-roundtrip MSE gate fails an under-fit codec", async () => {
    const r = await runWorkflow({
      id: "vidgate", name: "vidgate",
      steps: [
        // Impossible gate: 1 level / tiny codebook can't hit MSE ≤ 1e-9.
        { id: "vround", type: "video-roundtrip", params: { levels: 1, codebookSize: 2, maxMSE: 1e-9, clips: 2, frames: 2 } },
      ],
    });
    expect(r.ok).toBe(false);
    expect(r.firstFailure?.id).toBe("vround");
    expect(r.firstFailure?.error).toMatch(/reconstruction MSE .* exceeds max/);
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

describe("workflow — teaching Evermind to code", () => {
  test("train-tokenizer imports merges from a Hugging Face tokenizer.json (import-merges path)", async () => {
    // A minimal byte-level BPE tokenizer.json: vocab + merges, no training corpus.
    const hfTokenizer = {
      model: {
        vocab: { "<unk>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3, a: 4, b: 5, c: 6, ab: 7, abc: 8 },
        merges: ["a b", "ab c"],
      },
    };
    const r = await runWorkflow({
      id: "import", name: "import",
      steps: [{ id: "tok", type: "train-tokenizer", params: { hfTokenizer } }],
    });
    if (!r.ok) throw new Error(`broke: ${r.firstFailure?.error}`);
    expect(r.steps[0]!.detail).toMatch(/imported vocab .*Hugging Face/);
  });

  test("distill-corpus assembles offline (prompt → code) exemplars into a trainable corpus", async () => {
    const r = await runWorkflow({
      id: "distill", name: "distill",
      steps: [
        {
          id: "distill", type: "distill-corpus",
          params: { pairs: [
            { prompt: "add", completion: "function add(a, b) { return a + b; }" },
            { prompt: "sub", completion: "function sub(a, b) { return a - b; }" },
          ] },
        },
        { id: "tok", type: "train-tokenizer", params: { numMerges: 40 } },
        { id: "data", type: "dataset-quality", params: { minSequences: 2, minWords: 6 } },
      ],
    });
    if (!r.ok) throw new Error(`broke at ${r.firstFailure?.id}: ${r.firstFailure?.error}`);
    expect(r.steps[0]!.detail).toMatch(/distilled 2 exemplar.*2 offline/);
    // The distilled code became multiple trainable sequences (blank-line split).
    expect(r.steps[2]!.detail).toMatch(/2 seqs|3 seqs/);
  });

  test("distill-corpus drops an offline exemplar whose code fails its execution cases", async () => {
    const r = await runWorkflow({
      id: "filter", name: "filter",
      steps: [
        {
          id: "distill", type: "distill-corpus",
          params: { pairs: [
            { prompt: "add", completion: "function add(a, b) { return a + b; }", cases: [{ call: "add(2, 3)", expect: 5 }] },
            // Wrong implementation — must be dropped, never trained on.
            { prompt: "sub", completion: "function sub(a, b) { return a + b; }", cases: [{ call: "sub(5, 2)", expect: 3 }] },
          ] },
        },
      ],
    });
    if (!r.ok) throw new Error(`broke at ${r.firstFailure?.id}: ${r.firstFailure?.error}`);
    expect(r.steps[0]!.detail).toMatch(/distilled 1 exemplar.*1 offline.*1 dropped/);
  });

  test("distill-corpus strips a markdown code fence a teacher wrapped around the code", async () => {
    const r = await runWorkflow({
      id: "fence", name: "fence",
      steps: [
        {
          id: "distill", type: "distill-corpus",
          params: { pairs: [
            { prompt: "add", completion: "```js\nfunction add(a, b) { return a + b; }\n```", cases: [{ call: "add(2, 3)", expect: 5 }] },
          ] },
        },
      ],
    });
    // The fence would fail the execution case if not stripped → the exemplar survives.
    if (!r.ok) throw new Error(`broke at ${r.firstFailure?.id}: ${r.firstFailure?.error}`);
    expect(r.steps[0]!.detail).toMatch(/distilled 1 exemplar.*1 offline/);
  });

  test("selectBestCandidate keeps the first passing candidate (strongest-teacher-first) and drops all-failing", () => {
    const cases = [{ call: "f(2)", expect: 4 }];
    const opus = { teacher: "anthropic:claude-opus-4-8", completion: "function f(x){return x*x;}" };
    const cheap = { teacher: "openai:coder", completion: "function f(x){return x+x;}" };
    // Both listed; the strongest (first) correct answer wins.
    expect(selectBestCandidate([opus, cheap], cases)?.teacher).toBe("anthropic:claude-opus-4-8");
    // Opus wrong, cheap right → the verifiably-correct cheaper answer is kept.
    const opusWrong = { teacher: "anthropic:claude-opus-4-8", completion: "function f(x){return x+1;}" };
    expect(selectBestCandidate([opusWrong, cheap], cases)?.teacher).toBe("openai:coder");
    // None pass → null (nothing trained on).
    expect(selectBestCandidate([opusWrong, opusWrong], cases)).toBeNull();
    // No cases → first non-empty candidate (teacher-order priority).
    expect(selectBestCandidate([opus, cheap], [])?.teacher).toBe("anthropic:claude-opus-4-8");
    expect(selectBestCandidate([], cases)).toBeNull();
  });

  test("stripCodeFences unwraps a fenced block but leaves bare code untouched", () => {
    expect(stripCodeFences("```js\nfunction f(){}\n```")).toBe("function f(){}");
    expect(stripCodeFences("```\nx = 1\n```")).toBe("x = 1");
    expect(stripCodeFences("function f(){}")).toBe("function f(){}");
  });

  test("distill-corpus fails clearly when given neither pairs nor a live teacher", async () => {
    const r = await runWorkflow({ id: "empty", name: "empty", steps: [{ id: "d", type: "distill-corpus" }] });
    expect(r.ok).toBe(false);
    expect(r.firstFailure?.error).toMatch(/no exemplars/);
  });

  test("code-parse-check + code-eval grade real code, and their gates can fail bad output", async () => {
    const good = "function add(a, b) { return a + b; }";
    // Report-only (no thresholds) — both gates pass and surface metrics.
    const ok = await runWorkflow({
      id: "cg", name: "cg",
      steps: [
        { id: "parse", type: "code-parse-check", params: { code: good, language: "js" } },
        { id: "reward", type: "code-eval", params: { code: good, cases: [{ call: "add(2, 3)", expect: 5 }, { call: "add(-1, 1)", expect: 0 }] } },
      ],
    });
    if (!ok.ok) throw new Error(`broke at ${ok.firstFailure?.id}: ${ok.firstFailure?.error}`);
    expect(ok.steps.find((s) => s.id === "parse")!.detail).toMatch(/js-parse ok/);
    expect(ok.steps.find((s) => s.id === "reward")!.detail).toMatch(/2\/2 cases passed/);

    // A structure gate fails on unbalanced code.
    const broken = await runWorkflow({
      id: "cb", name: "cb",
      steps: [{ id: "parse", type: "code-parse-check", params: { code: "function add(a, b) { return a + b;", language: "js", minScore: 1 } }],
    });
    expect(broken.ok).toBe(false);
    expect(broken.firstFailure?.error).toMatch(/structure score|not valid/);

    // A pass-rate gate fails when the code is wrong.
    const wrong = await runWorkflow({
      id: "cw", name: "cw",
      steps: [{ id: "reward", type: "code-eval", params: { code: "function add(a, b) { return a - b; }", cases: [{ call: "add(2, 3)", expect: 5 }], minPassRate: 1 } }],
    });
    expect(wrong.ok).toBe(false);
    expect(wrong.firstFailure?.error).toMatch(/pass-rate/);
  });

  test("the Teach-Code template runs green end-to-end and ships an evermind-coder artifact", async () => {
    const r = await runWorkflow(getTemplate("teach-code")!, { registry: defaultStepRegistry });
    if (!r.ok) throw new Error(`broke at step ${r.firstFailure?.id}: ${r.firstFailure?.error}`);
    expect(r.steps.map((s) => s.id)).toEqual(
      ["distill", "tok", "data", "model", "converge", "eval", "gen", "parse", "reward", "bench", "codebench", "pkg", "export"],
    );
    expect(r.steps.every((s) => s.status === "pass")).toBe(true);
    expect((r.artifacts.evermind as ArrayBuffer).byteLength).toBeGreaterThan(0);
  }, 30000);
});

describe("workflow — warm-start (weight-port import)", () => {
  test("import-model builds a live EvermindLM from a .safetensors checkpoint, then packages it", async () => {
    const lm = new EvermindLM({ vocabSize: 40, dModel: 16, numLayers: 2, hiddenDim: 24, numExperts: 4, topK: 2, seed: 5 });
    const bytes = exportSafetensors(lm); // Uint8Array

    const r = await runWorkflow({
      id: "warm", name: "warm",
      steps: [
        { id: "imp", type: "import-model", params: { safetensors: bytes } },
        { id: "pkg", type: "package", params: { name: "warm" } },
      ],
    });
    if (!r.ok) throw new Error(`broke at ${r.firstFailure?.id}: ${r.firstFailure?.error}`);
    expect(r.steps[0]!.detail).toMatch(/imported .* params .*dModel 16, 2L, 4E/);
    // The imported model is real and serveable → package produces a valid artifact.
    expect((r.artifacts.evermind as ArrayBuffer).byteLength).toBeGreaterThan(0);
  });

  test("import-model fails clearly when no checkpoint bytes are provided", async () => {
    const r = await runWorkflow({ id: "noimp", name: "noimp", steps: [{ id: "imp", type: "import-model" }] });
    expect(r.ok).toBe(false);
    expect(r.firstFailure?.error).toMatch(/safetensors/);
  });
});
