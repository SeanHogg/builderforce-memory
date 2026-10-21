/**
 * tests/ml_runner.test.ts — the ML catalog's ONE registry, ONE runner (engine
 * selection per inference mode, decision recording) and ONE trainer seam.
 */
import { MlTaskRegistry } from "../src/ml/registry.js";
import {
  runMlTask,
  candidateEngines,
  InMemoryMlModels,
  MlDecisionLog,
  MlEngineUnavailableError,
  type MlCloudPort,
} from "../src/ml/run.js";
import { trainMlTask, mlEvalLoss } from "../src/ml/train.js";
import { DenseHead } from "../src/ml/dense_head.js";
import { HeadTrainer } from "../src/ml/head_trainer.js";
import { extractFeatures, logScale, clampRatio } from "../src/ml/features.js";
import { engineKey, type InferenceMode, type MlPrediction, type MlPredictor, type MlTask } from "../src/ml/types.js";
import type { TextEmbedder } from "../src/lm/text_embedder.js";
import { WEIGHT_UPDATE_WGSL } from "../src/kernels/weight_update.js";
import { fakeGpu } from "./fakeGpu";

interface In {
  a: number;
  b: number;
}

const LOGISTIC = { tier: "local", engine: "classical", algorithm: "logistic" } as const;
const KNN = { tier: "local", engine: "classical", algorithm: "knn" } as const;
const CLOUD = { tier: "cloud", engine: "workers-ai", model: "@cf/test" } as const;

function task(over: Partial<MlTask<In, string>> = {}): MlTask<In, string> {
  return {
    kind: "classify",
    id: "test.sign",
    description: "is a + b positive",
    features: { numeric: (i) => [i.a, i.b] },
    engines: [LOGISTIC, CLOUD, KNN],
    train: { minSamples: 4 },
    output: { labels: ["neg", "pos"], decode: (p) => ((p.scores[1] ?? 0) >= 0.5 ? "pos" : "neg") },
    ...over,
  };
}

const fixed = (scores: number[], confidence: number): MlPredictor => ({ predict: () => ({ scores, confidence }) });

function cloud(answer: MlPrediction | Error): MlCloudPort & { calls: number } {
  const port = {
    calls: 0,
    async run() {
      port.calls++;
      if (answer instanceof Error) throw answer;
      return answer;
    },
  };
  return port;
}

/** Bag-of-words hashing embedder: same words → same vector. */
const hashEmbedder: TextEmbedder = {
  dimensions: 64,
  fingerprint: "hash/64",
  async embed(text: string) {
    const out = new Float32Array(64);
    for (const w of text.toLowerCase().split(/\W+/).filter(Boolean)) {
      let h = 0;
      for (const c of w) h = (h * 31 + c.charCodeAt(0)) >>> 0;
      out[h % 64]! += 1;
    }
    const n = Math.hypot(...out) || 1;
    return out.map((x) => x / n);
  },
};

describe("MlTaskRegistry", () => {
  test("registers rows and refuses malformed ones", () => {
    const r = new MlTaskRegistry();
    r.register(task());
    expect(r.has("test.sign")).toBe(true);
    expect(r.require("test.sign").kind).toBe("classify");
    expect(r.list()).toHaveLength(1);
    expect(() => r.register(task())).toThrow(/already registered/);
    expect(() => r.register(task({ id: "NoDots" }))).toThrow(/malformed/);
    expect(() => r.register(task({ id: "test.empty", engines: [] }))).toThrow(/no engine/);
    expect(() => r.register(task({ id: "test.nolabels", output: { decode: () => "x" } }))).toThrow(/labels/);
    expect(() => r.require("test.missing")).toThrow(/unknown task/);
  });
});

describe("candidateEngines", () => {
  test("filters and orders by mode", () => {
    const t = task();
    expect(candidateEngines(t, "local-only").map(engineKey)).toEqual([engineKey(LOGISTIC), engineKey(KNN)]);
    expect(candidateEngines(t, "hybrid").map(engineKey)).toEqual([engineKey(LOGISTIC), engineKey(KNN), engineKey(CLOUD)]);
    expect(candidateEngines(t, "specific", CLOUD).map(engineKey)).toEqual([engineKey(CLOUD)]);
    expect(candidateEngines(t, "specific", "local:classical:knn").map(engineKey)).toEqual([engineKey(KNN)]);
    expect(() => candidateEngines(t, "specific")).toThrow(/needs ctx.engine/);
    expect(() => candidateEngines(t, "specific", "local:classical:gbm-small")).toThrow(/not one of/);
  });

  test("an unknown mode string throws instead of being read as 'specific'", async () => {
    const t = task();
    const bogus = "cloud-only" as unknown as InferenceMode;
    expect(() => candidateEngines(t, bogus, KNN)).toThrow(/unknown inference mode 'cloud-only'/);
    const registry = new MlTaskRegistry();
    registry.register(t);
    await expect(runMlTask("test.sign", { a: 1, b: 1 }, { registry, mode: bogus, engine: KNN })).rejects.toThrow(/unknown inference mode/);
  });
});

describe("runMlTask", () => {
  const registry = new MlTaskRegistry();
  registry.register(task());

  test("local-only answers locally and never calls the cloud", async () => {
    const models = new InMemoryMlModels();
    models.set("test.sign", engineKey(KNN), fixed([0.2, 0.8], 0.8));
    const port = cloud({ scores: [1, 0], confidence: 1 });
    const decisions = new MlDecisionLog();
    const r = await runMlTask<string>("test.sign", { a: 1, b: 1 }, { registry, mode: "local-only", models, cloud: port, decisions });
    expect(r.output).toBe("pos");
    expect(engineKey(r.engine)).toBe(engineKey(KNN));
    expect(port.calls).toBe(0);
    expect(decisions.list()[0]).toMatchObject({ engine: engineKey(KNN), tier: "local", skipped: [{ engine: engineKey(LOGISTIC), reason: "unavailable" }] });
  });

  test("local-only with nothing trained throws and records an unanswered decision", async () => {
    const decisions = new MlDecisionLog();
    await expect(runMlTask("test.sign", { a: 1, b: 1 }, { registry, mode: "local-only", cloud: cloud({ scores: [0, 1], confidence: 1 }), decisions })).rejects.toBeInstanceOf(
      MlEngineUnavailableError,
    );
    expect(decisions.list()[0]!.engine).toBeNull();
    expect(decisions.summary()["test.sign"]).toMatchObject({ runs: 1, unanswered: 1 });
  });

  test("hybrid falls through to the cloud when no local engine is trained", async () => {
    const port = cloud({ scores: [0.1, 0.9], confidence: 0.9 });
    const r = await runMlTask<string>("test.sign", { a: 1, b: 1 }, { registry, mode: "hybrid", cloud: port });
    expect(r.engine.tier).toBe("cloud");
    expect(r.output).toBe("pos");
    expect(port.calls).toBe(1);
  });

  test("hybrid defers a low-confidence local answer to the cloud, and keeps it when the cloud fails", async () => {
    const models = new InMemoryMlModels();
    models.set("test.sign", engineKey(LOGISTIC), fixed([0.55, 0.45], 0.55));
    const ok = cloud({ scores: [0.05, 0.95], confidence: 0.95 });
    const viaCloud = await runMlTask<string>("test.sign", { a: 1, b: 1 }, { registry, mode: "hybrid", models, cloud: ok, minConfidence: 0.8 });
    expect(viaCloud.engine.tier).toBe("cloud");
    expect(viaCloud.output).toBe("pos");

    const decisions = new MlDecisionLog();
    const down = cloud(new Error("gateway down"));
    const kept = await runMlTask<string>("test.sign", { a: 1, b: 1 }, { registry, mode: "hybrid", models, cloud: down, minConfidence: 0.8, decisions });
    expect(engineKey(kept.engine)).toBe(engineKey(LOGISTIC));
    expect(kept.output).toBe("neg");
    expect(decisions.list()[0]!.skipped.map((s) => s.reason)).toEqual(["low-confidence", "unavailable", "error"]);

    // Without a cloud port the low-confidence local answer stands immediately.
    const noCloud = await runMlTask<string>("test.sign", { a: 1, b: 1 }, { registry, mode: "hybrid", models, minConfidence: 0.8 });
    expect(engineKey(noCloud.engine)).toBe(engineKey(LOGISTIC));
  });

  test("an engine that throws is an 'error' skip; hybrid falls through to the cloud, and only all-fail throws", async () => {
    const models = new InMemoryMlModels();
    models.set("test.sign", engineKey(LOGISTIC), {
      predict: () => {
        throw new Error("corrupt head");
      },
    });
    const decisions = new MlDecisionLog();
    const r = await runMlTask<string>("test.sign", { a: 1, b: 1 }, { registry, mode: "hybrid", models, cloud: cloud({ scores: [0.1, 0.9], confidence: 0.9 }), decisions });
    expect(r.engine.tier).toBe("cloud");
    expect(r.output).toBe("pos");
    expect(decisions.list()[0]!.skipped[0]).toEqual({ engine: engineKey(LOGISTIC), reason: "error", detail: "corrupt head" });

    // No engine answers: the error names every engine and why.
    await expect(runMlTask("test.sign", { a: 1, b: 1 }, { registry, mode: "hybrid", models, cloud: cloud(new Error("gateway down")) })).rejects.toThrow(
      /local:classical:logistic: error \(corrupt head\).*local:classical:knn: unavailable.*cloud:workers-ai:@cf\/test: error \(gateway down\)/,
    );
  });

  test("a feature-extraction failure is an 'error' skip for each local engine, and the cloud still answers", async () => {
    const t = task({ id: "test.text", features: { text: (i) => `a=${i.a}` } });
    const reg = new MlTaskRegistry();
    reg.register(t);
    const models = new InMemoryMlModels();
    models.set("test.text", engineKey(LOGISTIC), fixed([0, 1], 1));
    models.set("test.text", engineKey(KNN), fixed([0, 1], 1));
    const decisions = new MlDecisionLog();
    // No embedder in the context: the text features cannot be extracted.
    const r = await runMlTask<string>("test.text", { a: 1, b: 1 }, { registry: reg, mode: "hybrid", models, cloud: cloud({ scores: [1, 0], confidence: 1 }), decisions });
    expect(r.engine.tier).toBe("cloud");
    const skipped = decisions.list()[0]!.skipped;
    expect(skipped.map((s) => s.reason)).toEqual(["error", "error"]);
    expect(skipped[0]!.detail).toMatch(/^features: .*no embedder/);
    await expect(runMlTask("test.text", { a: 1, b: 1 }, { registry: reg, mode: "local-only", models })).rejects.toBeInstanceOf(MlEngineUnavailableError);
  });

  test("specific uses exactly the named engine", async () => {
    const models = new InMemoryMlModels();
    models.set("test.sign", engineKey(LOGISTIC), fixed([0, 1], 1));
    models.set("test.sign", engineKey(KNN), fixed([1, 0], 1));
    const r = await runMlTask<string>("test.sign", { a: 1, b: 1 }, { registry, mode: "specific", engine: KNN, models });
    expect(engineKey(r.engine)).toBe(engineKey(KNN));
    expect(r.output).toBe("neg");
  });

  test("MlDecisionLog is bounded and summarises per engine", () => {
    const log = new MlDecisionLog(2);
    for (const c of [0.2, 0.4, 0.6]) log.record({ taskId: "t.x", mode: "hybrid", engine: "local:classical:knn", tier: "local", confidence: c, skipped: [] });
    expect(log.list()).toHaveLength(2);
    expect(log.summary()["t.x"]).toEqual({ runs: 2, byEngine: { "local:classical:knn": 2 }, unanswered: 0, meanConfidence: 0.5 });
  });
});

describe("features", () => {
  test("concatenates the embedded text and numeric fields", async () => {
    const spec = { text: (i: { t: string; n: number }) => i.t, numeric: (i: { t: string; n: number }) => [i.n] };
    const x = await extractFeatures(spec, { t: "hello world", n: 3 }, hashEmbedder);
    expect(x.length).toBe(65);
    expect(x[64]).toBe(3);
    await expect(extractFeatures(spec, { t: "x", n: 1 })).rejects.toThrow(/no embedder/);
    await expect(extractFeatures({ numeric: () => [Number.NaN] }, {})).rejects.toThrow(/non-finite/);
  });

  test("fixed scales", () => {
    expect(logScale(0, 100)).toBe(0);
    expect(logScale(100, 100)).toBeCloseTo(1, 9);
    expect(logScale(-5, 100)).toBe(0);
    expect(clampRatio(5, 10)).toBe(0.5);
    expect(clampRatio(5, 0)).toBe(0);
    expect(clampRatio(100, 1, 4)).toBe(4);
  });
});

describe("trainMlTask (the ONE trainer seam)", () => {
  const examples = [-4, -3, -2, -1.5, -1, 1, 1.5, 2, 3, 4].map((a) => ({ input: { a, b: a / 2 }, label: a > 0 ? 1 : 0 }));

  test("fits a classical engine, measures the held-out tail, stores the predictor, and the runner answers", async () => {
    const t = task();
    const registry = new MlTaskRegistry();
    registry.register(t);
    const models = new InMemoryMlModels();
    const shuffled = [examples[0]!, examples[9]!, examples[1]!, examples[8]!, examples[2]!, examples[7]!, examples[3]!, examples[6]!, examples[4]!, examples[5]!];
    const r = (await trainMlTask(t, shuffled, { models }))!;
    expect(engineKey(r.engine)).toBe(engineKey(LOGISTIC));
    expect(r.holdout).toBe(2);
    expect(r.holdoutLoss).not.toBeNull();
    expect(r.samples).toBe(10);
    const answer = await runMlTask<string>("test.sign", { a: 2.5, b: 1 }, { registry, mode: "local-only", models });
    expect(answer.output).toBe("pos");
    expect(answer.confidence).toBeGreaterThan(0.8);
  });

  test("stays untrained below minSamples; refuses rule-only and externally-trained tasks", async () => {
    expect(await trainMlTask(task(), examples.slice(0, 3))).toBeNull();
    const noTrain = task();
    delete noTrain.train;
    await expect(trainMlTask(noTrain, examples)).rejects.toThrow(/rule-based/);
    await expect(trainMlTask(task({ train: { minSamples: 1, trainedBy: "SomethingElse" } }), examples)).rejects.toThrow(/SomethingElse/);
  });

  test("a head engine trains a DenseHead through HeadTrainer", async () => {
    const t = task({ engines: [{ tier: "local", engine: "head", head: "mlp" }] });
    const r = (await trainMlTask(t, examples, { holdoutFraction: 0, head: { hiddenDim: 4, train: { epochs: 300, learningRate: 0.05 } } }))!;
    expect(r.predictor).toBeInstanceOf(DenseHead);
    expect(r.predictor.predict(Float32Array.of(3, 1.5)).scores[1]).toBeGreaterThan(0.9);
    expect(r.predictor.predict(Float32Array.of(-3, -1.5)).scores[0]).toBeGreaterThan(0.9);
  });

  test("text features train with the supplied embedder", async () => {
    const t: MlTask<{ text: string }, string> = {
      kind: "classify",
      id: "test.topic",
      description: "billing vs deploy",
      features: { text: (i) => i.text },
      engines: [KNN],
      train: { minSamples: 2 },
      output: { labels: ["billing", "deploy"], decode: (p) => ((p.scores[1] ?? 0) >= 0.5 ? "deploy" : "billing") },
    };
    const ex = [
      { input: { text: "invoice payment refund" }, label: 0 },
      { input: { text: "refund the invoice" }, label: 0 },
      { input: { text: "deploy the worker" }, label: 1 },
      { input: { text: "worker deploy failed" }, label: 1 },
    ];
    await expect(trainMlTask(t, ex)).rejects.toThrow(/no embedder/);
    const r = (await trainMlTask(t, ex, { embedder: hashEmbedder, holdoutFraction: 0, classical: { knn: { k: 2 } } }))!;
    const registry = new MlTaskRegistry();
    registry.register(t);
    const models = new InMemoryMlModels();
    models.set(t.id, engineKey(KNN), r.predictor);
    expect((await runMlTask<string>(t.id, { text: "invoice refund" }, { registry, mode: "local-only", models, embedder: hashEmbedder })).output).toBe("billing");
    expect((await runMlTask<string>(t.id, { text: "deploy worker" }, { registry, mode: "local-only", models, embedder: hashEmbedder })).output).toBe("deploy");
  });

  test("mlEvalLoss: cross-entropy for classify, ½·sq error for regress, null otherwise", () => {
    const p = fixed([0.25, 0.75], 0.75);
    expect(mlEvalLoss("classify", p, [{ x: new Float32Array(1), y: 1 }])).toBeCloseTo(-Math.log(0.75), 9);
    expect(mlEvalLoss("regress", fixed([3], 1), [{ x: new Float32Array(1), y: 1 }])).toBeCloseTo(2, 9);
    expect(mlEvalLoss("anomaly", p, [{ x: new Float32Array(1), y: 0 }])).toBeNull();
  });
});

describe("DenseHead + HeadTrainer", () => {
  test("analytic gradients match finite differences", () => {
    for (const objective of ["softmax", "mse"] as const) {
      const head = new DenseHead({ inputDim: 3, hiddenDim: 4, outputDim: objective === "softmax" ? 3 : 1, objective, seed: 11 });
      const sample = { x: Float32Array.of(0.3, -0.7, 0.5), y: objective === "softmax" ? 2 : 0.4, weight: 1.5 };
      head.zeroGrad();
      head.accumulate(sample);
      const eps = 1e-3;
      head.parameters().forEach((p, pi) => {
        for (let i = 0; i < p.numel; i += 3) {
          const orig = p.data[i]!;
          p.data[i] = orig + eps;
          const up = head.loss(sample);
          p.data[i] = orig - eps;
          const down = head.loss(sample);
          p.data[i] = orig;
          expect(head.gradients()[pi]!.data[i]).toBeCloseTo((up - down) / (2 * eps), 2);
        }
      });
    }
  });

  test("trains on the CPU path, params round-trip, empty set refused", async () => {
    const head = new DenseHead({ inputDim: 1, hiddenDim: 0, outputDim: 1, objective: "mse", seed: 2 });
    const data = [0, 1, 2, 3].map((x) => ({ x: Float32Array.of(x), y: 2 * x + 1 }));
    const trainer = new HeadTrainer(head);
    expect(trainer.gpuTraining).toBe(false);
    const losses = await trainer.train(data, { epochs: 400, learningRate: 0.05, maxGradNorm: 100 });
    expect(losses[losses.length - 1]!).toBeLessThan(losses[0]! * 0.01);
    expect(head.predict(Float32Array.of(5)).scores[0]).toBeCloseTo(11, 0);
    const copy = new DenseHead({ inputDim: 1, hiddenDim: 0, outputDim: 1, objective: "mse", seed: 99 });
    copy.importParams(head.exportParams());
    expect(copy.predict(Float32Array.of(5)).scores).toEqual(head.predict(Float32Array.of(5)).scores);
    await expect(trainer.train([])).rejects.toThrow(/HeadTrainer.train: no samples/);
  });

  test("the CPU step is the shared, guarded AdamW: a NaN gradient never reaches a weight", async () => {
    const head = new DenseHead({ inputDim: 1, hiddenDim: 0, outputDim: 1, objective: "mse", seed: 4 });
    const before = Array.from(head.exportParams());
    await new HeadTrainer(head).train([{ x: Float32Array.of(Number.NaN), y: 1 }], { epochs: 3 });
    expect(Array.from(head.exportParams())).toEqual(before);
  });

  test("the GPU step packs AdamParams exactly as WEIGHT_UPDATE_WGSL declares it", async () => {
    // Field order and count straight from the kernel source, so the two cannot drift.
    const body = /struct AdamParams\s*\{([^}]*)\}/.exec(WEIGHT_UPDATE_WGSL)![1]!;
    const fields = body
      .split("\n")
      .map((l) => l.replace(/\/\/.*$/, "").trim())
      .filter((l) => /^\w+\s*:/.test(l))
      .map((l) => l.split(":")[0]!.trim());
    expect(fields).toEqual(["num_elements", "lr", "beta1", "beta2", "eps", "weight_decay", "beta1_t", "beta2_t", "max_delta", "_pad0", "_pad1", "_pad2"]);

    const gpu = fakeGpu();
    const head = new DenseHead({ inputDim: 2, hiddenDim: 0, outputDim: 1, objective: "mse", seed: 1 });
    const trainer = new HeadTrainer(head, gpu.device);
    expect(trainer.gpuTraining).toBe(true);
    await trainer.train([{ x: Float32Array.of(1, 2), y: 3 }], { epochs: 1, learningRate: 0.01, weightDecay: 0.1 });

    const steps = gpu.dispatches.filter((d) => d.entryPoint === "adamw_update");
    expect(steps).toHaveLength(head.parameters().length);
    steps.forEach((d, i) => {
      const uni = d.buffers[0]!;
      expect(uni.size).toBe(fields.length * 4); // 48 — the old packer wrote 32
      const dv = new DataView(uni.bytes.buffer, uni.bytes.byteOffset, uni.size);
      const f32 = (name: string) => dv.getFloat32(fields.indexOf(name) * 4, true);
      expect(dv.getUint32(0, true)).toBe(head.parameters()[i]!.numel);
      expect(f32("lr")).toBeCloseTo(0.01, 7);
      expect(f32("beta1")).toBeCloseTo(0.9, 7);
      expect(f32("beta2")).toBeCloseTo(0.999, 7);
      expect(f32("eps")).toBeCloseTo(1e-8, 12);
      expect(f32("weight_decay")).toBeCloseTo(0.1, 7);
      expect(f32("beta1_t")).toBeCloseTo(0.9, 7);
      expect(f32("beta2_t")).toBeCloseTo(0.999, 7);
      expect(f32("max_delta")).toBe(0);
    });
  });
});
