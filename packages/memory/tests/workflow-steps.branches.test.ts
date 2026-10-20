/**
 * tests/workflow-steps.branches.test.ts — the built-in steps' guard rails and
 * less-travelled paths, driven step-by-step with fake models and tokenizers so
 * each test checks the step's own logic (not a tiny model's luck):
 *
 *   - every step that needs an earlier step's output fails with a message
 *     naming the step to add, instead of crashing on undefined;
 *   - the dataset / convergence / generation gates reject what they exist to
 *     reject;
 *   - the distill panel dedupes calls, skips blank answers, drops tasks no
 *     teacher solved and skips teachers without credentials;
 *   - import-model accepts every documented checkpoint encoding;
 *   - the code gates report the details they promise.
 */

import { jest } from "@jest/globals";
import { EvermindLM, exportSafetensors } from "@seanhogg/builderforce-memory-engine";
import { defaultStepRegistry } from "../src/workflow/registry.js";
import type { StackContext } from "../src/diagnostics/stack-diagnostic.js";

function ctxWith(bag: Record<string, unknown> = {}, trace: StackContext["trace"] = []): StackContext {
  return { bag, artifacts: {}, trace, now: () => 0 };
}

const step = (type: string, params?: Record<string, unknown>) => defaultStepRegistry.build({ id: type, type, ...(params ? { params } : {}) });

/** A tokenizer stand-in: the steps under test only call `encode` (and read `vocabSize`). */
const fakeTokenizer = (encode: (s: string) => number[], vocabSize = 8) => ({ encode, vocabSize });

const add = "function add(a, b) { return a + b; }";
const wrongAdd = "function add(a, b) { return a - b; }";

describe("steps — missing prerequisites fail with the step to add", () => {
  test.each([
    ["orchestration", /prerequisite layer/],
    ["deployment", /no model from L1/],
    ["train-model", /add a 'train-tokenizer' step first/],
    ["evaluate", /add 'train-model' first/],
    ["dataset-quality", /add a 'train-tokenizer' step first/],
    ["generate-check", /add 'train-model' first/],
    ["roundtrip", /add 'train-model' first/],
    ["package", /add 'train-model' first/],
    ["export", /add 'train-model' first/],
    ["convergence", /run 'train-model' first/],
    ["code-parse-check", /no code to check/],
    ["code-eval", /no code to evaluate/],
  ])("%s", async (type, message) => {
    await expect(step(type).run(ctxWith())).rejects.toThrow(message);
  });

  test("an empty convergence history is the same as none", async () => {
    await expect(step("convergence").run(ctxWith({ trainingHistory: [] }))).rejects.toThrow(/no training history/);
  });

  test("huggingface export needs the tokenizer, even with a model", async () => {
    await expect(step("export", { format: "huggingface" }).run(ctxWith({ model: {} }))).rejects.toThrow(/needs a tokenizer/);
  });

  test("video-train with zero clips has nothing to train on", async () => {
    await expect(step("video-train", { clips: 0 }).run(ctxWith())).rejects.toThrow(/no clips to train on/);
  });
});

describe("steps — observability and orchestration", () => {
  test("observability rejects an empty trace and a malformed span", async () => {
    await expect(step("observability").run(ctxWith())).rejects.toThrow(/no spans captured/);
    const bad = [{ id: "s1", layer: 1 as unknown as string, label: "x", status: "pass" as const, ms: 0 }];
    await expect(step("observability").run(ctxWith({}, bad))).rejects.toThrow(/malformed span: s1/);
    const negative = [{ id: "s2", layer: "L1", label: "x", status: "pass" as const, ms: -1 }];
    await expect(step("observability").run(ctxWith({}, negative))).rejects.toThrow(/malformed span: s2/);
  });

  test("orchestration still runs the tool when retrieval finds nothing", async () => {
    const shout = jest.fn((a: string) => a.toUpperCase());
    const bag = {
      model: { generateText: () => "ok" },
      tokenizer: {},
      cognition: { recall: async () => [] },
      candidates: [],
      tools: new Map([["shout", shout]]),
    };
    expect(await step("orchestration").run(ctxWith(bag))).toBe("retrieve → recall → tool → model");
    expect(shout).toHaveBeenCalledWith("deploy");
  });
});

describe("steps — corpus and training gates", () => {
  test("train-model rejects a corpus with no trainable sequence, from params or the bag", async () => {
    const tokenizer = fakeTokenizer(() => [1]); // every sentence encodes too short to train on
    await expect(step("train-model", { corpus: "One. Two." }).run(ctxWith({ tokenizer }))).rejects.toThrow(/no trainable sequences/);
    await expect(step("train-model").run(ctxWith({ tokenizer, corpus: "From the bag." }))).rejects.toThrow(/no trainable sequences/);
    await expect(step("train-model").run(ctxWith({ tokenizer }))).rejects.toThrow(/no trainable sequences/);
  });

  test("train-model with zero epochs reports a zero loss rather than crashing", async () => {
    const tokenizer = fakeTokenizer(() => [1, 2, 3]);
    const ctx = ctxWith({ tokenizer, corpus: "Alpha beta. Gamma delta." });
    const detail = await step("train-model", { epochs: 0 }).run(ctx);
    expect(detail).toBe("trained 0 epochs over 2 seqs, loss 0.000");
    expect(ctx.bag.trainingHistory).toEqual([]);
    expect(ctx.bag.model).toBeInstanceOf(EvermindLM);
  });

  const sentences = (n: number) => Array.from({ length: n }, (_, i) => `sentence number ${i} has six words.`).join(" ");

  test("dataset-quality rejects too few sequences, then too much repetition", async () => {
    // Plenty of words, but every sentence encodes too short to be a sequence.
    const tooShort = fakeTokenizer(() => [1]);
    const ctx = ctxWith({ tokenizer: tooShort });
    await expect(step("dataset-quality", { corpus: sentences(6) }).run(ctx)).rejects.toThrow(/too few trainable sequences: 0 < 3/);
    expect(ctx.bag.datasetMetrics).toEqual({ words: 36, sequences: 0, avgSeqLen: 0, duplicateRatio: 1 });

    // Every sentence encodes identically: 6 sequences, 1 unique.
    const same = fakeTokenizer(() => [1, 2]);
    await expect(step("dataset-quality", { corpus: sentences(6) }).run(ctxWith({ tokenizer: same }))).rejects.toThrow(/too repetitive: 83% duplicate/);

    // Distinct sequences pass, with the corpus taken from the bag.
    let next = 0;
    const distinct = fakeTokenizer(() => [next++, next++]);
    const detail = await step("dataset-quality").run(ctxWith({ tokenizer: distinct, corpus: sentences(6) }));
    expect(detail).toBe("36 words, 6 seqs (avg 2 tok), 0% dup");
  });

  test("convergence rejects a diverged or flat loss curve and accepts a falling one", async () => {
    await expect(step("convergence").run(ctxWith({ trainingHistory: [Number.NaN, 1] }))).rejects.toThrow(/not finite/);
    await expect(step("convergence").run(ctxWith({ trainingHistory: [2, Number.POSITIVE_INFINITY] }))).rejects.toThrow(/not finite/);
    await expect(step("convergence").run(ctxWith({ trainingHistory: [1, 1] }))).rejects.toThrow(/did not decrease/);
    await expect(step("convergence", { minDrop: 0.5 }).run(ctxWith({ trainingHistory: [1, 0.8] }))).rejects.toThrow(/did not decrease/);
    const ctx = ctxWith({ trainingHistory: [2, 1.5, 1] });
    expect(await step("convergence").run(ctx)).toBe("loss 2.000 → 1.000 (−1.000)");
    expect(ctx.bag.converged).toBe(true);
  });
});

describe("steps — generation gates", () => {
  test("evaluate rejects a model that returns no text", async () => {
    await expect(step("evaluate").run(ctxWith({ model: { generateText: () => 42 }, tokenizer: {} }))).rejects.toThrow(/evaluation generation failed/);
  });

  test("generate-check rejects non-deterministic sampling and empty output", async () => {
    let calls = 0;
    const flaky = { generateText: () => `sample ${calls++}` };
    await expect(step("generate-check").run(ctxWith({ model: flaky, tokenizer: {} }))).rejects.toThrow(/non-deterministic/);
    const silent = { generateText: () => "   " };
    await expect(step("generate-check").run(ctxWith({ model: silent, tokenizer: {} }))).rejects.toThrow(/empty output/);
    const steady = { generateText: () => "steady text" };
    const ctx = ctxWith({ model: steady, tokenizer: {} });
    expect(await step("generate-check", { seed: 9 }).run(ctx)).toBe('deterministic@seed 9; sample "steady text"');
    expect(ctx.bag.sample).toBe("steady text");
  });
});

describe("steps — code gates", () => {
  test("code-parse-check names js-parse only for JavaScript", async () => {
    expect(await step("code-parse-check", { code: add }).run(ctxWith())).toMatch(/js-parse ok$/);
    expect(await step("code-parse-check", { code: "def add(a, b):\n    return a + b", language: "python" }).run(ctxWith())).not.toMatch(/js-parse/);
    // The code comes from the bag's sample when no param is given.
    expect(await step("code-parse-check").run(ctxWith({ sample: add }))).toMatch(/^structure/);
  });

  test("code-eval skips without cases, ignores malformed cases, and quotes a compile error at its gate", async () => {
    expect(await step("code-eval", { code: add }).run(ctxWith())).toBe("no cases — execution reward skipped");
    expect(await step("code-eval", { code: add, cases: "not a list" }).run(ctxWith())).toBe("no cases — execution reward skipped");
    expect(await step("code-eval", { code: add, cases: [{ call: 1 }, { call: "add(1, 2)", expect: 3 }] }).run(ctxWith())).toBe("1/1 cases passed (100%)");
    await expect(
      step("code-eval", { code: "function add(a, b) {", cases: [{ call: "add(1, 2)", expect: 3 }], minPassRate: 1 }).run(ctxWith()),
    ).rejects.toThrow(/pass-rate 0% < min 100% \(.+\)/);
    await expect(
      step("code-eval", { code: wrongAdd, cases: [{ call: "add(1, 2)", expect: 3 }], minPassRate: 1 }).run(ctxWith()),
    ).rejects.toThrow(/pass-rate 0% < min 100%$/);
  });

  test("code-benchmark honours a per-task token budget, tasks without cases and non-text output", async () => {
    const budgets: number[] = [];
    const model = {
      generateText: (prompt: string, _tok: unknown, opts: { maxNewTokens: number }) => {
        budgets.push(opts.maxNewTokens);
        return prompt === "silent" ? undefined : add;
      },
    };
    const ctx = ctxWith({ model, tokenizer: {} });
    const detail = await step("code-benchmark", {
      maxNewTokens: 30,
      tasks: [
        { prompt: "write add", cases: [{ call: "add(2, 3)", expect: 5 }], maxNewTokens: 12 },
        { prompt: "no cases", cases: "none" },
        { prompt: "silent", cases: [{ call: "add(1, 1)", expect: 2 }] },
        { notAPrompt: true },
      ],
    }).run(ctx);
    expect(budgets).toEqual([12, 30, 30]);
    expect(detail).toBe("pass@1 1/3 (33%) over held-out coding tasks");
    const report = ctx.bag.codeBenchmark as { perTask: Array<{ prompt: string; passed: boolean; cases: number }> };
    expect(report.perTask.map((t) => [t.prompt, t.passed, t.cases])).toEqual([["write add", true, 1], ["no cases", false, 0], ["silent", false, 1]]);
  });
});

describe("steps — import-model checkpoint encodings", () => {
  const lm = new EvermindLM({ vocabSize: 40, dModel: 16, numLayers: 2, hiddenDim: 24, numExperts: 4, topK: 2, seed: 5 });
  const bytes = exportSafetensors(lm);
  const base64 = Buffer.from(bytes).toString("base64");

  test.each([
    ["an ArrayBuffer", { safetensors: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) }],
    ["a number[]", { safetensors: Array.from(bytes) }],
    ["base64", { safetensorsBase64: base64 }],
    ["base64 when the bytes param is not numeric", { safetensors: ["x"], safetensorsBase64: base64 }],
  ])("imports from %s", async (_label, params) => {
    const ctx = ctxWith();
    const detail = await step("import-model", params).run(ctx);
    expect(detail).toContain(`from ${bytes.length} bytes`);
    expect((ctx.bag.model as EvermindLM).config.dModel).toBe(16);
  });

  test("applies topK and a rename map (named tensors renamed, the rest kept)", async () => {
    const ctx = ctxWith();
    await step("import-model", { safetensors: bytes, topK: 1, renameMap: { "token_embedding.weight": "token_embedding.weight", "not.a.tensor": "ignored" } }).run(ctx);
    expect((ctx.bag.model as EvermindLM).config.topK).toBe(1);
  });

  test("an empty base64 string is no checkpoint", async () => {
    await expect(step("import-model", { safetensorsBase64: "" }).run(ctxWith())).rejects.toThrow(/provide 'safetensors'/);
  });
});

describe("steps — distill-corpus teacher panel", () => {
  /** Answers per teacher host, keyed by the user prompt. */
  const answers: Record<string, Record<string, string>> = {
    "strong.test": { "write add": "```js\n" + add + "\n```", "explain": "Adds two numbers.", "unsolvable": wrongAdd, "blank": "   " },
    "weak.test": { "write add": wrongAdd, "explain": "It adds.", "unsolvable": wrongAdd, "blank": "   " },
  };

  let fetchSpy: ReturnType<typeof jest.spyOn>;
  const calls: string[] = [];
  beforeEach(() => {
    calls.length = 0;
    fetchSpy = jest.spyOn(globalThis, "fetch").mockImplementation(async (input: unknown, init?: RequestInit) => {
      const url = new URL(String(input));
      const body = JSON.parse(String(init?.body)) as { messages: Array<{ role: string; content: string }> };
      const prompt = body.messages[body.messages.length - 1]!.content;
      calls.push(`${url.host} ${prompt}`);
      const content = answers[url.host]?.[prompt] ?? "";
      return new Response(JSON.stringify({ choices: [{ message: { content } }] }), { status: 200, headers: { "Content-Type": "application/json" } });
    });
  });
  afterEach(() => fetchSpy.mockRestore());

  test("keeps the strongest passing answer, dedupes repeated prompts, and drops what nobody solved", async () => {
    const ctx = ctxWith();
    const detail = await step("distill-corpus", {
      teachers: [
        { provider: "OpenAI", baseUrl: "https://strong.test/v1", apiKey: "k1", model: "strong-1", systemPrompt: "code only", maxTokens: 64 },
        { baseUrl: "https://weak.test/v1" },
        { provider: "anthropic" }, // no apiKey: skipped
        { provider: "openai", apiKey: "k3" }, // no baseUrl: skipped
        null,
      ],
      tasks: [
        { prompt: "write add", cases: [{ call: "add(2, 3)", expect: 5 }, { bad: true }] },
        { prompt: "write add", cases: [{ call: "add(2, 3)", expect: 5 }] }, // repeated: answered from the memo
        { prompt: "explain" }, // no cases: the first teacher's non-blank answer
        { prompt: "unsolvable", cases: [{ call: "add(2, 3)", expect: 5 }] },
        { prompt: "blank" }, // every teacher blank: dropped
        { prompt: 7 }, // not a prompt: ignored
      ],
      pairs: [{ prompt: "offline", completion: add }, { prompt: 1, completion: "x" }],
    }).run(ctx);

    // Two teachers × four distinct prompts — the repeat cost nothing.
    expect(calls.sort()).toEqual([
      "strong.test blank", "strong.test explain", "strong.test unsolvable", "strong.test write add",
      "weak.test blank", "weak.test explain", "weak.test unsolvable", "weak.test write add",
    ]);
    const pairs = ctx.bag.distillPairs as Array<{ prompt: string; completion: string; teacher: string; cases?: unknown[] }>;
    expect(pairs.map((p) => [p.prompt, p.teacher])).toEqual([
      ["offline", "offline"],
      ["write add", "openai:strong-1"],
      ["write add", "openai:strong-1"],
      ["explain", "openai:strong-1"],
    ]);
    expect(pairs[1]!.completion).toBe(add); // the fence was stripped
    expect(pairs[1]!.cases).toEqual([{ call: "add(2, 3)", expect: 5 }]);
    expect(pairs[3]!.cases).toBeUndefined();
    expect(detail).toMatch(/^distilled 4 exemplar\(s\) \(4 total, \d+ chars\) — 1 offline, 3 verified via 2 teacher\(s\): openai:strong-1, openai:default, 2 dropped$/);
  });

  test("plain prompts[] and the legacy single-teacher params form a one-teacher panel that accumulates", async () => {
    const ctx = ctxWith({ distillPairs: [{ prompt: "earlier", completion: add, teacher: "offline" }] });
    const detail = await step("distill-corpus", {
      baseUrl: "https://weak.test/v1",
      model: "weak-1",
      prompts: ["explain", 5],
    }).run(ctx);
    expect(calls).toEqual(["weak.test explain"]);
    expect((ctx.bag.distillPairs as unknown[]).length).toBe(2);
    expect(detail).toMatch(/^distilled 1 exemplar\(s\) \(2 total, \d+ chars\) — 1 verified via 1 teacher\(s\): openai:weak-1$/);
  });
});
