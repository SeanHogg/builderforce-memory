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
import { AnomalyCompositeModel, fitClassical } from "../src/ml/classical/index.js";
import { extractFeatureMatrix } from "../src/ml/features.js";
import {
  BUILTIN_ML_TASKS,
  createMlRegistry,
  curateExperience,
  curatePendingExperience,
  curationRecords,
  curationStands,
  CURATION_KNOWN_LIMIT,
  QUALITY_OBSERVED_MIN_RUNS,
  RUN_ANOMALY_MIN_SCALE,
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
import { experienceDocuments, type ExperienceDocument } from "../src/experience/corpus.js";
import type { CuratedOut, Episode, RecordedStep, Run, Skill } from "../src/experience/types.js";
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
    // 12 = the task's minSamples: below it the task stays untrained (see the next test).
    const ex = experienceQualityExamples(Array.from({ length: 6 }, (_, i) => [good(i), bad(i)]).flat());
    const models = new InMemoryMlModels();
    const r = (await trainMlTask(EXPERIENCE_QUALITY_TASK, ex, { embedder: hashEmbedder, models, holdoutFraction: 0 }))!;
    expect(engineKey(r.engine)).toBe("local:classical:knn");
    const q = await runMlTask<{ label: string; pHigh: number }>("experience.quality", good(9), { registry, mode: "local-only", models, embedder: hashEmbedder });
    expect(q.output.label).toBe("high");
  });

  test("the label is not a feature: no success / failure rate in the numeric fields", () => {
    expect(EXPERIENCE_QUALITY_TASK.features.numericNames).toEqual(["steps", "runs"]);
    const i: ExperienceQualityInput = { doc: { id: "x", text: "t" }, steps: 3, runs: 4, succeeded: 4, failed: 0 };
    expect(EXPERIENCE_QUALITY_TASK.features.numeric!(i)).toEqual(EXPERIENCE_QUALITY_TASK.features.numeric!({ ...i, succeeded: 0, failed: 4 }));
  });

  describe("small support never drops a good procedure (no stub predictor)", () => {
    const doc = (id: string, text: string): ExperienceDocument => ({ id, text });
    const high: ExperienceQualityInput = { doc: doc("h1", "Task: save report (app.exe)\n1. Click Save."), steps: 3, runs: 5, succeeded: 5, failed: 0 };
    const lows: ExperienceQualityInput[] = [
      { doc: doc("l1", "Task: wiggle mouse (app.exe)\n1. Click nowhere."), steps: 40, runs: 5, succeeded: 0, failed: 5 },
      { doc: doc("l2", "Task: open twice (app.exe)\n1. Click Open. 2. Click Open."), steps: 30, runs: 4, succeeded: 1, failed: 3 },
      { doc: doc("l3", "Task: retry forever (cli.exe)\n1. Press Enter."), steps: 25, runs: 6, succeeded: 0, failed: 6 },
    ];
    const train = [...lows, high];
    // 4 runs, so the model judges it (from QUALITY_OBSERVED_MIN_RUNS the observed rate does).
    const copy: ExperienceQualityInput = { ...high, doc: doc("h2", high.doc.text), runs: 4, succeeded: 4 };
    const byId = new Map([...train, copy].map((i) => [i.doc.id, i] as const));
    const input = (d: ExperienceDocument) => byId.get(d.id)!;

    /** The knn engine fitted on the real features of the 3 low + 1 high set. */
    async function knnOn(k?: number) {
      const X = await extractFeatureMatrix(EXPERIENCE_QUALITY_TASK.features, train, hashEmbedder);
      const samples = train.map((t, i) => ({ x: X[i]!, y: t === high ? 1 : 0 }));
      const models = new InMemoryMlModels();
      models.set("experience.quality", "local:classical:knn", fitClassical("knn", samples, k != null ? { knn: { k } } : {}));
      return models;
    }

    test("below minSamples the task stays untrained, so an exact copy of the high doc is kept", async () => {
      const models = new InMemoryMlModels();
      expect(await trainMlTask(EXPERIENCE_QUALITY_TASK, experienceQualityExamples(train), { embedder: hashEmbedder, models })).toBeNull();
      const r = await curateExperience([copy.doc], { embedder: hashEmbedder, dedupeThreshold: 1.01, quality: { ctx: { registry, mode: "local-only", models }, input } });
      expect(r.kept.map((d) => d.id)).toEqual(["h2"]);
    });

    test("fitted anyway, k = ⌊√4⌋ = 2 keeps the exact copy; the old k = n (every point votes) dropped it", async () => {
      const opts = (models: InMemoryMlModels) => ({ embedder: hashEmbedder, dedupeThreshold: 1.01, quality: { ctx: { registry, mode: "local-only" as const, models }, input } });
      const kept = await curateExperience([copy.doc], opts(await knnOn()));
      expect(kept.kept.map((d) => d.id)).toEqual(["h2"]);
      const everyPointVotes = await curateExperience([copy.doc], opts(await knnOn(4)));
      expect(everyPointVotes.dropped).toEqual([{ id: "h2", reason: "low-quality", score: expect.any(Number) }]);
    });

    test("an unscored procedure (runs = 0) is kept even by a model that would score it low", async () => {
      const neverRun: ExperienceQualityInput = { ...lows[0]!, doc: doc("u1", lows[0]!.doc.text), runs: 0, succeeded: 0, failed: 0 };
      const models = await knnOn(4);
      const r = await curateExperience([neverRun.doc], {
        embedder: hashEmbedder,
        dedupeThreshold: 1.01,
        quality: { ctx: { registry, mode: "local-only", models }, input: () => neverRun },
      });
      expect(r.kept.map((d) => d.id)).toEqual(["u1"]);
      expect(r.dropped).toEqual([]);
    });
  });
});

describe("experience.quality in curation: leave-one-out, the observed rate, standing verdicts", () => {
  // Twelve learned skills with the same procedure, each 0/2 — the neighbours.
  const failing = Array.from({ length: 12 }, (_, i) => skill(`f${i}`, "Wiggle mouse", `ef${i}`));
  const failRuns = failing.flatMap((s) => [run(s.id, "failed", 0), run(s.id, "failed", 1)]);
  const known = experienceDocuments([], failing);
  const opts = { embedder: hashEmbedder, known, dedupeThreshold: 1.01 };
  const succeeded = (id: string, n: number) => Array.from({ length: n }, (_, i) => run(id, "succeeded", i));

  test("a 1/1 skill among failing neighbours is scored by its neighbours alone — it never votes for itself", async () => {
    const t1 = skill("t1", "Wiggle mouse", "et1");
    const r = await curatePendingExperience(experienceDocuments([], [t1]), { episodes: [], skills: [...failing, t1], runs: [...failRuns, ...succeeded("t1", 1)] }, opts);
    // t1 is in the training set (pending + learned, scored). With its own example voting
    // at similarity 1, k = 3 gave it ≈ 1/3; left out, only the failing neighbours vote.
    expect(r.kept).toEqual([]);
    expect(r.dropped).toEqual([{ id: "t1", reason: "low-quality", score: expect.any(Number) }]);
    expect(r.dropped[0]!.score!).toBeLessThan(0.01);
  });

  test("from QUALITY_OBSERVED_MIN_RUNS runs the observed rate decides: 5/5 is kept, 0/5 dropped, whatever the neighbours", async () => {
    expect(QUALITY_OBSERVED_MIN_RUNS).toBe(5);
    const t5 = skill("t5", "Wiggle mouse", "et5");
    const good = await curatePendingExperience(experienceDocuments([], [t5]), { episodes: [], skills: [...failing, t5], runs: [...failRuns, ...succeeded("t5", 5)] }, opts);
    expect(good.kept.map((d) => d.id)).toEqual(["t5"]);
    expect(good.dropped).toEqual([]);

    const okSkills = Array.from({ length: 12 }, (_, i) => skill(`g${i}`, "Wiggle mouse", `eg${i}`));
    const b5 = skill("b5", "Wiggle mouse", "eb5");
    const bad = await curatePendingExperience(
      experienceDocuments([], [b5]),
      { episodes: [], skills: [...okSkills, b5], runs: [...okSkills.flatMap((s) => succeeded(s.id, 2)), ...[0, 1, 2, 3, 4].map((i) => run("b5", "failed", i))] },
      { ...opts, known: experienceDocuments([], okSkills) },
    );
    expect(bad.dropped).toEqual([{ id: "b5", reason: "low-quality", score: 0 }]);
  });

  test("an earlier curated_out verdict that still stands is skipped, not judged again", async () => {
    const t1 = skill("t1", "Wiggle mouse", "et1");
    const runs = [...failRuns, ...succeeded("t1", 1)];
    const first = await curatePendingExperience(experienceDocuments([], [t1]), { episodes: [], skills: [...failing, t1], runs }, opts);
    const records = curationRecords(first, { runs }, 42);
    expect(records).toEqual({ dropped: { t1: { status: "curated_out", reason: "low-quality", score: expect.any(Number), runs: 1, at: 42 } }, kept: [] });

    const again = await curatePendingExperience(experienceDocuments([], [t1]), { episodes: [], skills: [...failing, t1], runs, curated: records.dropped }, opts);
    expect(again.dropped).toEqual([]);
    expect(again.kept).toEqual([]);
    expect(again.alreadyDropped).toEqual([{ id: "t1", reason: "low-quality", score: records.dropped.t1!.score }]);

    // A new run is new evidence: the verdict no longer stands, so t1 is judged again.
    const moreRuns = [...runs, run("t1", "succeeded", 7)];
    const rejudged = await curatePendingExperience(experienceDocuments([], [t1]), { episodes: [], skills: [...failing, t1], runs: moreRuns, curated: records.dropped }, opts);
    expect(rejudged.alreadyDropped).toEqual([]);
    expect(rejudged.dropped.map((d) => d.id)).toEqual(["t1"]);
  });

  test("curationStands: a duplicate while what it repeats is learned; low-quality while its run count holds; exclusion never", () => {
    const v = (over: Partial<CuratedOut>): CuratedOut => ({ status: "curated_out", reason: "duplicate", of: "a", runs: 0, at: 0, ...over });
    expect(curationStands(v({}), 0, new Set(["a"]))).toBe(true);
    expect(curationStands(v({}), 0, new Set())).toBe(false);
    expect(curationStands(v({ reason: "low-quality", runs: 3 }), 3, new Set())).toBe(true);
    expect(curationStands(v({ reason: "low-quality", runs: 3 }), 4, new Set())).toBe(false);
    expect(curationStands(v({ reason: "excluded" }), 0, new Set(["a"]))).toBe(false);
    expect(curationStands(undefined, 0, new Set())).toBe(false);
  });

  test("the quality-training set is bounded at CURATION_KNOWN_LIMIT scored procedures, newest first", async () => {
    const many = Array.from({ length: CURATION_KNOWN_LIMIT + 20 }, (_, i) => skill(`m${i}`, `Thing ${i}`, `em${i}`));
    const runs = many.flatMap((s, i) => [run(s.id, i % 2 ? "failed" : "succeeded", 0)]);
    const seen = new Set<string>();
    const counting: TextEmbedder = { ...hashEmbedder, embed: async (t: string) => (seen.add(t), hashEmbedder.embed(t)) };
    // Learned most recently first; nothing pending, no known window beyond the limit embedded either.
    await curatePendingExperience([], { episodes: [], skills: many, runs }, { embedder: counting, known: experienceDocuments([], many) });
    expect(seen.size).toBe(CURATION_KNOWN_LIMIT);
    expect([...seen].some((t) => t.includes(`Thing ${CURATION_KNOWN_LIMIT + 1}`))).toBe(false);
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

  test("a pending document that repeats an ALREADY-LEARNED one is dropped as its duplicate", async () => {
    const r = await curateExperience([docs[1]!, docs[2]!], { embedder: hashEmbedder, known: [docs[0]!] });
    expect(r.kept.map((d) => d.id)).toEqual(["e3"]);
    expect(r.dropped).toEqual([{ id: "e9", reason: "duplicate", of: "s1", score: expect.closeTo(1, 6) }]);
  });

  test("curatePendingExperience: the host's one call — quality trained from runs, dedupe vs a bounded known window", async () => {
    const sk = skill("s1", "Pay invoice", "e1");
    const source = { episodes: [episode("e2", "Pay invoice", ["Save"])], skills: [sk], runs: [] as Run[] };
    const known: ExperienceDocument[] = [{ id: "s1", text: "Task: Pay invoice (app.exe)\n1. Click “Save”." }];
    const pending: ExperienceDocument[] = [{ id: "e2", text: "Task: Pay invoice (app.exe)\n1. Click “Save”." }, docs[3]!];
    const r = await curatePendingExperience(pending, source, { embedder: hashEmbedder, known });
    expect(r.kept.map((d) => d.id)).toEqual(["e4"]);
    expect(r.dropped).toEqual([{ id: "e2", reason: "duplicate", of: "s1", score: expect.closeTo(1, 6) }]);

    // Only the most recent CURATION_KNOWN_LIMIT known documents are embedded.
    const seen: string[] = [];
    const counting: TextEmbedder = { ...hashEmbedder, embed: async (t: string) => (seen.push(t), hashEmbedder.embed(t)) };
    const many = Array.from({ length: CURATION_KNOWN_LIMIT + 5 }, (_, i): ExperienceDocument => ({ id: `k${i}`, text: `Task: thing ${i}` }));
    await curatePendingExperience([], source, { embedder: counting, known: many });
    expect(new Set(seen).size).toBe(CURATION_KNOWN_LIMIT);
    expect(seen).not.toContain(`Task: thing ${CURATION_KNOWN_LIMIT}`);
  });

  test("quality filter and exclusions; an untrained quality task keeps everything", async () => {
    // Every document has run outcomes here, so the quality filter applies to each.
    const input = (d: ExperienceDocument): ExperienceQualityInput => ({ doc: d, steps: 1, runs: 2, succeeded: 1, failed: 1 });
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

  test("ONE composite engine: trained, stored and answered under local:classical:anomaly-composite", async () => {
    expect(RUN_ANOMALY_TASK.engines.map(engineKey)).toEqual(["local:classical:anomaly-composite"]);
    const models = new InMemoryMlModels();
    const r = (await trainMlTask(RUN_ANOMALY_TASK, history.map((input) => ({ input, label: 0 })), { models }))!;
    expect(engineKey(r.engine)).toBe("local:classical:anomaly-composite");
    expect(models.get("run.anomaly", "local:classical:anomaly-composite")).toBe(r.predictor);
    expect(r.predictor).toBeInstanceOf(AnomalyCompositeModel);
    const det = r.predictor as RunAnomalyDetector;
    expect(Array.from(det.population.scale).every((sc, i) => sc >= RUN_ANOMALY_MIN_SCALE[i]!)).toBe(true);
    expect(det.trend.minScale).toEqual(RUN_ANOMALY_MIN_SCALE);
  });

  test("after 20 clean runs, one tool failure is NOT an anomaly — a large cost spike still is", async () => {
    const clean: RunMetrics[] = Array.from({ length: 20 }, (_, i) => ({ costUsd: 0.1, durationMs: 60_000 + 1000 * (i % 3), toolFailures: 0 }));
    const models = new InMemoryMlModels();
    await trainMlTask(RUN_ANOMALY_TASK, clean.map((input) => ({ input, label: 0 })), { models });
    const ctx = { registry, mode: "local-only" as const, models };
    const oneFailure = await runMlTask<RunAnomaly>("run.anomaly", { costUsd: 0.1, durationMs: 61_000, toolFailures: 1 }, ctx);
    expect(oneFailure.output.anomalous).toBe(false);
    const spike = await runMlTask<RunAnomaly>("run.anomaly", { costUsd: 5, durationMs: 61_000, toolFailures: 0 }, ctx);
    expect(spike.output.anomalous).toBe(true);
  });

  test("floors are per metric: a 10 s → 15 min run flags; a 1.5× duration wobble does not", async () => {
    // Each metric has its own log reference, so one scalar floor was ~2× on cost but
    // ~4.5× on duration — 10 s → 15 min scored z ≈ 3.0 and slipped under EWMA's 3.
    expect(RUN_ANOMALY_MIN_SCALE[0]).toBeCloseTo(Math.log(3) / (3 * Math.log1p(1000)), 6);
    expect(RUN_ANOMALY_MIN_SCALE[1]).toBeCloseTo(Math.log(3) / (3 * Math.log1p(3_600_000)), 6);
    expect(RUN_ANOMALY_MIN_SCALE[2]).toBe(0.1);
    const steady: RunMetrics[] = Array.from({ length: 20 }, () => ({ costUsd: 0.1, durationMs: 10_000, toolFailures: 0 }));
    const models = new InMemoryMlModels();
    await trainMlTask(RUN_ANOMALY_TASK, steady.map((input) => ({ input, label: 0 })), { models });
    const ctx = { registry, mode: "local-only" as const, models };
    const long = await runMlTask<RunAnomaly>("run.anomaly", { costUsd: 0.1, durationMs: 15 * 60_000, toolFailures: 0 }, ctx);
    expect(long.output.anomalous).toBe(true);
    const wobble = await runMlTask<RunAnomaly>("run.anomaly", { costUsd: 0.1, durationMs: 15_000, toolFailures: 0 }, ctx);
    expect(wobble.output.anomalous).toBe(false);
    const costWobble = await runMlTask<RunAnomaly>("run.anomaly", { costUsd: 0.15, durationMs: 10_000, toolFailures: 0 }, ctx);
    expect(costWobble.output.anomalous).toBe(false);
    // Failures keep 0.1: 1–2 after a clean history are normal, 3 flag.
    const two = await runMlTask<RunAnomaly>("run.anomaly", { costUsd: 0.1, durationMs: 10_000, toolFailures: 2 }, ctx);
    expect(two.output.anomalous).toBe(false);
    const three = await runMlTask<RunAnomaly>("run.anomaly", { costUsd: 0.1, durationMs: 10_000, toolFailures: 3 }, ctx);
    expect(three.output.anomalous).toBe(true);
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
