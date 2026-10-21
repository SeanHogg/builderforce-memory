/**
 * tests/ml_tasks.test.ts — the built-in catalog rows (docs/ml-task-catalog.md §4):
 * limbic.affect (the extraction), experience.quality/dedupe + the opt-in curation
 * step in front of adaptation, run.anomaly, run.local-success (learning from cloud
 * outcomes).
 */
import { LimbicModel } from "../src/limbic/limbic_model.js";
import { LimbicTrainer, limbicHead, type LimbicSample } from "../src/limbic/limbic_trainer.js";
import { LIMBIC_AFFECT_TASK, limbicAffectPredictor, type LimbicAffectOutput } from "../src/limbic/limbic_task.js";
import { HeadTrainer } from "../src/ml/head_trainer.js";
import { InMemoryMlModels, runMlTask } from "../src/ml/run.js";
import { trainMlTask } from "../src/ml/train.js";
import { engineKey, type MlPredictor } from "../src/ml/types.js";
import {
  BUILTIN_ML_TASKS,
  createMlRegistry,
  curateExperience,
  experienceQualityInputs,
  experienceQualityExamples,
  experienceQualityLabel,
  DedupeIndex,
  EXPERIENCE_QUALITY_TASK,
  RUN_ANOMALY_TASK,
  RunAnomalyDetector,
  experienceRunMetrics,
  LOCAL_SUCCESS_TASK,
  localSuccessLabel,
  localSuccessExamples,
  CLOUD_ONLY_FAILURE_WEIGHT,
  type ExperienceQualityInput,
  type LocalSuccessOutcome,
  type LocalTurnFeatures,
  type RunAnomaly,
  type RunMetrics,
  type LocalSuccess,
} from "../src/ml/tasks/index.js";
import { EvermindLM } from "../src/lm/evermind_lm.js";
import { EvermindTextEmbedder } from "../src/lm/text_embedder.js";
import { BPETokenizer } from "../src/tokenizer/bpe.js";
import type { ExperienceDocument } from "../src/experience/corpus.js";
import type { Episode, RecordedStep, Run, Skill } from "../src/experience/types.js";
import type { TextEmbedder } from "../src/lm/text_embedder.js";

const registry = createMlRegistry();

describe("built-in registry", () => {
  test("holds the catalog rows in order", () => {
    expect(registry.list().map((t) => t.id)).toEqual(["limbic.affect", "experience.quality", "experience.dedupe", "run.anomaly", "run.local-success"]);
    expect(BUILTIN_ML_TASKS).toHaveLength(5);
  });
});

describe("limbic.affect (the extracted head)", () => {
  const CFG = { inputDim: 4, hiddenDim: 6, stateDim: 8, seed: 77 };

  function data(n: number): LimbicSample[] {
    return Array.from({ length: n }, (_, i) => {
      const input = Float32Array.from({ length: 4 }, (_, j) => Math.sin(i * 1.3 + j));
      return {
        input,
        state: new Float32Array(8),
        deltaTarget: Float32Array.from({ length: 8 }, (_, k) => 0.3 * Math.tanh(input[k % 4]!)),
        reward: 0.2 * input[0]!,
      };
    });
  }

  test("LimbicTrainer is exactly HeadTrainer over limbicHead (no behaviour change)", async () => {
    const a = new LimbicModel(CFG);
    const b = new LimbicModel(CFG);
    const d = data(12);
    const la = await new LimbicTrainer(a, null).train(d, { epochs: 25, learningRate: 0.05 });
    const lb = await new HeadTrainer(limbicHead(b), null).train(d, { epochs: 25, learningRate: 0.05 });
    expect(la).toEqual(lb);
    expect(Array.from(a.win)).toEqual(Array.from(b.win));
    expect(new LimbicTrainer(a).evaluate(d)).toBe(new HeadTrainer(limbicHead(a)).evaluate(d));
  });

  test("the runner answers with the model's own forward pass", async () => {
    const model = new LimbicModel(CFG);
    const models = new InMemoryMlModels();
    models.set("limbic.affect", engineKey(LIMBIC_AFFECT_TASK.engines[0]!), limbicAffectPredictor(model));
    const input = { embedding: [0.1, -0.2, 0.3, 0.4], state: [0.5, 0, 0, 0, 0, 0, 0, 0.2] };
    const r = await runMlTask<LimbicAffectOutput>("limbic.affect", input, { registry, mode: "local-only", models });
    const f = model.forward(input.embedding, model.initHidden(), input.state);
    expect(Array.from(r.output.delta)).toEqual(Array.from(f.delta));
    expect(r.output.reward).toBeCloseTo(f.reward, 6);
  });

  test("its params federate via the flat vector, and the catalog seam defers training to LimbicTrainer", async () => {
    const a = limbicAffectPredictor(new LimbicModel(CFG));
    const bModel = new LimbicModel({ ...CFG, seed: 1 });
    limbicAffectPredictor(bModel).importParams!(a.exportParams!());
    const x = Float32Array.from({ length: 12 }, (_, i) => i / 12);
    expect(limbicAffectPredictor(bModel).predict(x).scores).toEqual(a.predict(x).scores);
    expect(() => a.predict(new Float32Array(3))).toThrow(/expected 12 features/);
    await expect(trainMlTask(LIMBIC_AFFECT_TASK, [{ input: { embedding: [0, 0, 0, 0], state: new Array(8).fill(0) }, label: 0 }])).rejects.toThrow(/LimbicTrainer/);
  });
});

// ── experience ───────────────────────────────────────────────────────────────

const el = (name: string) => ({ name, automationId: "", controlType: "Button", className: "", windowTitle: "App", processName: "app", relX: 0, relY: 0 });

function skill(id: string, name: string, src: string): Skill {
  return {
    id,
    name,
    program: "C:/apps/app.exe",
    args: [],
    params: [],
    steps: [{ id: `${id}-1`, action: { kind: "click", target: el("Save") }, requiresApproval: false }],
    sourceEpisode: src,
    createdAt: 0,
  };
}

function episode(id: string, name: string, clicks: string[]): Episode {
  return { id, name, program: "C:/apps/app.exe", args: [], startedAt: 0, steps: clicks.map((c, i): RecordedStep => ({ id: `${id}-${i}`, atMs: i, action: { kind: "click", target: el(c) } })) };
}

function run(skillId: string, status: Run["status"], i = 0, failedSteps = 0): Run {
  return {
    id: `r${skillId}${i}`,
    skillId,
    skillName: skillId,
    trigger: "manual",
    status,
    startedAt: 1000 * i,
    endedAt: 1000 * i + 500,
    steps: Array.from({ length: failedSteps }, (_, k) => ({ idx: k, stepId: "x", outcome: "failed" as const, at: 0 })),
  };
}

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

describe("experience.quality", () => {
  test("joins run outcomes onto documents and labels only those with runs", () => {
    const s1 = skill("s1", "Pay invoice", "e1");
    const inputs = experienceQualityInputs([episode("e1", "Pay invoice", ["Save"]), episode("e2", "Other", ["Open", "Close"])], [s1], [
      run("s1", "succeeded", 0),
      run("s1", "failed", 1),
      run("s1", "succeeded", 2),
      run("s1", "running", 3),
    ]);
    // e1 is compiled into s1, so it is superseded.
    expect(inputs.map((i) => i.doc.id)).toEqual(["s1", "e2"]);
    expect(inputs[0]).toMatchObject({ steps: 1, runs: 3, succeeded: 2, failed: 1 });
    expect(inputs[1]).toMatchObject({ steps: 2, runs: 0 });
    expect(experienceQualityLabel(inputs[0]!)).toBe(1);
    expect(experienceQualityLabel(inputs[1]!)).toBeNull();
    expect(experienceQualityExamples(inputs)).toHaveLength(1);
    expect(experienceQualityLabel({ ...inputs[0]!, succeeded: 0, failed: 3 })).toBe(0);
  });

  test("trains on run outcomes and scores a new procedure", async () => {
    const doc = (id: string, text: string): ExperienceDocument => ({ id, text });
    const good = (i: number): ExperienceQualityInput => ({ doc: doc(`g${i}`, "Task: save report. 1. Click Save."), steps: 3, runs: 5, succeeded: 5, failed: 0 });
    const bad = (i: number): ExperienceQualityInput => ({ doc: doc(`b${i}`, "Task: wiggle mouse. 1. Click nowhere."), steps: 40, runs: 5, succeeded: 0, failed: 5 });
    const ex = experienceQualityExamples([good(1), bad(1), good(2), bad(2), good(3), bad(3)]);
    const models = new InMemoryMlModels();
    const r = (await trainMlTask(EXPERIENCE_QUALITY_TASK, ex, { embedder: hashEmbedder, models, holdoutFraction: 0 }))!;
    expect(engineKey(r.engine)).toBe("local:classical:knn");
    const q = await runMlTask<{ label: string; pHigh: number }>("experience.quality", good(9), { registry, mode: "local-only", models, embedder: hashEmbedder });
    expect(q.output.label).toBe("high");
  });
});

describe("experience.dedupe + curation (opt-in, in front of adaptAndDiff)", () => {
  const docs: ExperienceDocument[] = [
    { id: "s1", text: "Task: Pay invoice (app.exe)\n1. Click “Save”." },
    { id: "e9", text: "Task: Pay invoice (app.exe)\n1. Click “Save”." },
    { id: "e3", text: "Task: Deploy worker (cli.exe)\n1. Press Enter." },
    { id: "e4", text: "Task: Book travel (web.exe)\n1. Click “Search”." },
  ];

  test("drops a near-duplicate of an earlier kept document — the first copy wins", async () => {
    const r = await curateExperience(docs, { embedder: hashEmbedder });
    expect(r.kept.map((d) => d.id)).toEqual(["s1", "e3", "e4"]);
    expect(r.dropped).toEqual([{ id: "e9", reason: "duplicate", of: "s1", score: expect.closeTo(1, 6) }]);
  });

  test("works with the person's real CPU embedder (EvermindTextEmbedder)", async () => {
    const tok = new BPETokenizer();
    tok.train(docs.map((d) => d.text).join(" "), { numMerges: 30 });
    const lm = new EvermindLM({ vocabSize: tok.vocabSize, dModel: 16, numLayers: 1, hiddenDim: 24, seed: 7 });
    const r = await curateExperience(docs, { embedder: new EvermindTextEmbedder(lm, tok) });
    expect(r.dropped.find((d) => d.id === "e9")?.reason).toBe("duplicate");
    expect(r.kept[0]!.id).toBe("s1");
  });

  test("quality filter and exclusions; an untrained quality task keeps everything", async () => {
    const input = (d: ExperienceDocument): ExperienceQualityInput => ({ doc: d, steps: 1, runs: 0, succeeded: 0, failed: 0 });
    const untrained = await curateExperience(docs, { embedder: hashEmbedder, dedupeThreshold: 1.01, quality: { ctx: { registry, mode: "local-only" }, input } });
    expect(untrained.kept).toHaveLength(4);

    const models = new InMemoryMlModels();
    // A stub quality model: "Book travel" is low quality.
    const stub: MlPredictor = { predict: (x) => (x[0] === -1 ? { scores: [0.9, 0.1], confidence: 0.9 } : { scores: [0.1, 0.9], confidence: 0.9 }) };
    models.set("experience.quality", "local:classical:knn", stub);
    const flagged = await curateExperience(docs, {
      embedder: { ...hashEmbedder, embed: async (t: string) => (t.includes("Book travel") ? Float32Array.of(-1, ...new Array(63).fill(0)) : hashEmbedder.embed(t)) },
      quality: { ctx: { registry, mode: "local-only", models }, input },
      exclude: new Set(["e3"]),
    });
    expect(flagged.kept.map((d) => d.id)).toEqual(["s1"]);
    expect(flagged.dropped.map((d) => [d.id, d.reason])).toEqual([
      ["e9", "duplicate"],
      ["e3", "excluded"],
      ["e4", "low-quality"],
    ]);
  });

  test("DedupeIndex reports nearest, similarity and the duplicate flag", () => {
    const idx = new DedupeIndex(0.9);
    expect(idx.predict(Float32Array.of(1, 0)).scores).toEqual([-1, 0, 0]);
    idx.add(Float32Array.of(1, 0));
    expect(idx.predict(Float32Array.of(1, 0.01)).scores[2]).toBe(1);
    expect(idx.predict(Float32Array.of(0, 1)).scores[2]).toBe(0);
    expect(idx.size).toBe(1);
  });
});

// ── run.anomaly ──────────────────────────────────────────────────────────────

describe("run.anomaly", () => {
  const history: RunMetrics[] = Array.from({ length: 20 }, (_, i) => ({
    costUsd: 0.1 + 0.01 * (i % 5),
    durationMs: 60_000 + 1000 * (i % 7),
    toolFailures: i % 3,
  }));

  test("robust-z + EWMA flag a run far outside its range, and say which metric", async () => {
    const models = new InMemoryMlModels();
    const r = (await trainMlTask(RUN_ANOMALY_TASK, history.map((input) => ({ input, label: 0 })), { models }))!;
    expect(r.predictor).toBeInstanceOf(RunAnomalyDetector);
    const ctx = { registry, mode: "local-only" as const, models };
    const normal = await runMlTask<RunAnomaly>("run.anomaly", { costUsd: 0.12, durationMs: 62_000, toolFailures: 1 }, ctx);
    expect(normal.output.anomalous).toBe(false);
    const costly = await runMlTask<RunAnomaly>("run.anomaly", { costUsd: 5, durationMs: 62_000, toolFailures: 1 }, ctx);
    expect(costly.output.anomalous).toBe(true);
    const det = r.predictor as RunAnomalyDetector;
    const why = det.explain(RUN_ANOMALY_TASK.features.numeric!({ costUsd: 5, durationMs: 62_000, toolFailures: 1 }));
    expect(why.find((w) => w.metric === "cost")!.population).toBeGreaterThan(1);
    expect(why.find((w) => w.metric === "duration")!.population).toBeLessThan(1);
    expect(await trainMlTask(RUN_ANOMALY_TASK, history.slice(0, 4).map((input) => ({ input, label: 0 })))).toBeNull();
  });

  test("experience run metrics come from the run log", () => {
    expect(experienceRunMetrics(run("s1", "failed", 2, 3), 0.4)).toEqual({ costUsd: 0.4, durationMs: 500, toolFailures: 3 });
  });
});

// ── run.local-success ────────────────────────────────────────────────────────

describe("run.local-success (learns from cloud outcomes)", () => {
  const turn = (fill: number, i: number): LocalTurnFeatures => ({
    promptTokens: 500 + 10 * i,
    contextTokens: Math.round(fill * 8192) - 500,
    servedContextTokens: 8192,
    tools: 4,
    turnIndex: i % 10,
  });

  test("labels: local result, fallback = local failure, cloud-only failure is weak, cloud-only success is no evidence", () => {
    const t = turn(0.3, 0);
    expect(localSuccessLabel({ turn: t, local: "succeeded" })).toEqual({ label: 1, weight: 1 });
    expect(localSuccessLabel({ turn: t, local: "failed", cloud: "succeeded", fellBack: true })).toEqual({ label: 0, weight: 1 });
    expect(localSuccessLabel({ turn: t, cloud: "succeeded", fellBack: true })).toEqual({ label: 0, weight: 1 });
    expect(localSuccessLabel({ turn: t, cloud: "failed" })).toEqual({ label: 0, weight: CLOUD_ONLY_FAILURE_WEIGHT });
    expect(localSuccessLabel({ turn: t, cloud: "succeeded" })).toBeNull();
  });

  test("trains on local AND cloud outcomes and predicts for the Hybrid router", async () => {
    const outcomes: LocalSuccessOutcome[] = [];
    for (let i = 0; i < 30; i++) {
      outcomes.push({ turn: turn(0.2 + 0.01 * (i % 10), i), local: "succeeded" });
      outcomes.push({ turn: turn(1.3 + 0.02 * (i % 10), i), cloud: "succeeded", fellBack: true }); // overflowed locally
      outcomes.push({ turn: turn(0.5, i), cloud: "succeeded" }); // cloud only: no evidence
    }
    const ex = localSuccessExamples(outcomes);
    expect(ex).toHaveLength(60);
    const models = new InMemoryMlModels();
    const r = (await trainMlTask(LOCAL_SUCCESS_TASK, ex, { models }))!;
    expect(engineKey(r.engine)).toBe("local:classical:logistic");
    const ctx = { registry, mode: "local-only" as const, models };
    const small = await runMlTask<LocalSuccess>("run.local-success", turn(0.25, 3), ctx);
    const huge = await runMlTask<LocalSuccess>("run.local-success", turn(1.4, 3), ctx);
    expect(small.output.succeeds).toBe(true);
    expect(huge.output.succeeds).toBe(false);
    expect(huge.output.probability).toBeLessThan(small.output.probability);
    // The GBM engine trains on the same rows.
    const g = (await trainMlTask(LOCAL_SUCCESS_TASK, ex, { engine: LOCAL_SUCCESS_TASK.engines[1]! }))!;
    expect(g.predictor.predict(Float32Array.from(LOCAL_SUCCESS_TASK.features.numeric!(turn(1.4, 3)))).scores[1]).toBeLessThan(0.5);
  });
});
