/**
 * ml/tasks/experience.ts — `experience.quality` and `experience.dedupe`, and the
 * opt-in curation step that applies them in front of the ONE adaptation recipe
 * (`evermind/adapt.ts`), so the private model learns from good, varied examples
 * instead of every transcript. Curation only drops documents; it never rewrites one.
 */
import { experienceDocuments, type ExperienceDocument } from "../../experience/corpus.js";
import type { CurationRecord } from "../../experience/store.js";
import type { CuratedOut, Episode, Run, Skill } from "../../experience/types.js";
import { KnnModel } from "../classical/knn.js";
import { logScale, type TextEmbedder } from "../features.js";
import { MlTaskRegistry } from "../registry.js";
import { InMemoryMlModels, MlEngineUnavailableError, runMlTask, type MlModelStore, type MlRunContext } from "../run.js";
import { trainMlTask, type MlTrainExample } from "../train.js";
import type { MlPrediction, MlPredictor, MlTask } from "../types.js";

// ── experience.quality ────────────────────────────────────────────────────────

/** One learnable procedure plus the outcome fields that say how well it went. */
export interface ExperienceQualityInput {
  doc: ExperienceDocument;
  /** Steps in the procedure. */
  steps: number;
  /** Runs of the skill (0 for an uncompiled demonstration). */
  runs: number;
  succeeded: number;
  failed: number;
}

export interface ExperienceQuality {
  label: "low" | "high";
  /** P(high). */
  pHigh: number;
}

export const EXPERIENCE_QUALITY_TASK: MlTask<ExperienceQualityInput, ExperienceQuality> = {
  kind: "classify",
  id: "experience.quality",
  description: "Is this procedure worth learning from (does it run, and run well)?",
  features: {
    text: (i) => i.doc.text,
    // No success/failure rate: the label IS `succeeded / runs >= 0.5`, so a rate
    // feature would leak it — the model would learn the label, not the procedure.
    numeric: (i) => [logScale(i.steps, 50), logScale(i.runs, 20)],
    numericNames: ["steps", "runs"],
  },
  engines: [
    { tier: "local", engine: "classical", algorithm: "knn" },
    { tier: "local", engine: "classical", algorithm: "logistic" },
  ],
  // Below 12 labelled procedures the task stays untrained and curation keeps
  // everything. At 12, after the 20% holdout ≈10 remain to fit, both classes can have
  // several members, and kNN votes with k = ⌊√n⌋ = 3 neighbours rather than the whole
  // set — with fewer, every point votes and the answer collapses to the class prior.
  train: { minSamples: 12 },
  output: {
    labels: ["low", "high"],
    decode: (p: MlPrediction) => ({ label: (p.scores[1] ?? 0) >= 0.5 ? "high" : "low", pHigh: p.scores[1] ?? 0 }),
  },
};

/** Quality inputs for every learnable document, with run outcomes joined by skill id. */
export function experienceQualityInputs(episodes: Episode[], skills: Skill[], runs: readonly Run[] = []): ExperienceQualityInput[] {
  const steps = new Map<string, number>([...skills.map((s) => [s.id, s.steps.length] as const), ...episodes.map((e) => [e.id, e.steps.length] as const)]);
  const byItem = new Map<string, { runs: number; succeeded: number; failed: number }>();
  for (const r of runs) {
    if (r.status === "running") continue;
    const t = byItem.get(r.skillId) ?? { runs: 0, succeeded: 0, failed: 0 };
    t.runs++;
    if (r.status === "succeeded") t.succeeded++;
    if (r.status === "failed") t.failed++;
    byItem.set(r.skillId, t);
  }
  return experienceDocuments(episodes, skills).map((doc) => ({
    doc,
    steps: steps.get(doc.id) ?? 0,
    ...(byItem.get(doc.id) ?? { runs: 0, succeeded: 0, failed: 0 }),
  }));
}

/** The training label of a quality input: 1 (high) when most finished runs succeeded, 0 when not, null with no runs. */
export function experienceQualityLabel(i: ExperienceQualityInput): number | null {
  if (i.runs === 0) return null;
  return i.succeeded / i.runs >= 0.5 ? 1 : 0;
}

/**
 * From this many finished runs, curation judges a procedure by its OBSERVED success
 * rate (`succeeded / runs`) instead of the model: the label is then known, and better
 * than any guess from neighbours. The model only scores procedures with 1–4 runs;
 * never-run ones (0) are kept.
 */
export const QUALITY_OBSERVED_MIN_RUNS = 5;

/** Labelled examples for `experience.quality` — only the inputs that have run outcomes. */
export function experienceQualityExamples(inputs: readonly ExperienceQualityInput[]): MlTrainExample<ExperienceQualityInput>[] {
  return inputs.flatMap((input) => {
    const label = experienceQualityLabel(input);
    return label == null ? [] : [{ input, label }];
  });
}

// ── experience.dedupe ─────────────────────────────────────────────────────────

export interface ExperienceDuplicate {
  /** Index (in keep order) of the nearest already-kept document, or -1 when none. */
  nearest: number;
  similarity: number;
  duplicate: boolean;
}

/** Cosine similarity at or above which two procedures count as one. */
export const DEFAULT_DEDUPE_THRESHOLD = 0.97;

export const EXPERIENCE_DEDUPE_TASK: MlTask<ExperienceDocument, ExperienceDuplicate> = {
  kind: "cluster",
  id: "experience.dedupe",
  description: "Is this procedure a near-copy of one already kept?",
  features: { text: (d) => d.text },
  engines: [{ tier: "local", engine: "classical", algorithm: "knn" }],
  output: {
    decode: (p: MlPrediction) => ({ nearest: p.scores[0] ?? -1, similarity: p.scores[1] ?? 0, duplicate: (p.scores[2] ?? 0) === 1 }),
  },
};

/**
 * The `experience.dedupe` predictor: a cosine nearest-neighbour index of the
 * documents kept so far. `scores = [nearestIndex, similarity, duplicate ? 1 : 0]`.
 */
export class DedupeIndex implements MlPredictor {
  private readonly _knn = new KnnModel(1, "cosine", 2);

  constructor(readonly threshold: number = DEFAULT_DEDUPE_THRESHOLD) {}

  get size(): number {
    return this._knn.size;
  }

  add(x: Float32Array): void {
    this._knn.add(x);
  }

  predict(x: Float32Array): MlPrediction {
    const n = this._knn.nearest(x);
    if (!n) return { scores: [-1, 0, 0], confidence: 1 };
    const dup = n.similarity >= this.threshold;
    return { scores: [n.index, n.similarity, dup ? 1 : 0], confidence: Math.abs(n.similarity) };
  }
}

// ── curation (opt-in, in front of adaptAndDiff) ─────────────────────────────

export interface CurationOptions {
  /** Embeds each document once (the person's CPU Evermind: `EvermindTextEmbedder`). */
  embedder: TextEmbedder;
  /** Cosine similarity at or above which a document duplicates a kept one. Default {@link DEFAULT_DEDUPE_THRESHOLD}. */
  dedupeThreshold?: number;
  /**
   * Quality filter. A document whose P(high) is below `minPHigh` (default 0.5) is
   * dropped. P(high) is the OBSERVED success rate once it has
   * {@link QUALITY_OBSERVED_MIN_RUNS} finished runs; with 1–4 it is the runner's
   * answer, and one the runner cannot score (no trained engine) is kept. An UNSCORED
   * document (`runs === 0` — never run is unknown, not bad) is always kept.
   */
  quality?: {
    ctx: MlRunContext;
    input: (doc: ExperienceDocument) => ExperienceQualityInput;
    minPHigh?: number;
    /**
     * The ids of the documents the quality model was trained on, in training order. A
     * document among them is scored LEAVE-ONE-OUT — its own example does not vote, or
     * a kNN would count it at similarity 1. Exact for the kNN engine (the one
     * {@link curatePendingExperience} trains); another engine answers as fitted.
     */
    trainedIds?: readonly string[];
  };
  /** Document ids to leave out (e.g. those whose runs `run.anomaly` flagged). */
  exclude?: ReadonlySet<string>;
  /**
   * Documents the model has ALREADY learned. They seed the de-duplication index, so a
   * pending document that only repeats one of them is dropped (reported `of` that
   * one); they are never in `kept`. Each is embedded once per call, so this is
   * bounded — {@link curatePendingExperience} keeps the {@link CURATION_KNOWN_LIMIT}
   * most recently learned.
   */
  known?: readonly ExperienceDocument[];
}

/**
 * How many already-learned documents curation de-duplicates against: the most
 * recently learned ones. Each costs one CPU embed per curation call, and a
 * repeat of something learned long ago is cheap to relearn, so a recent window is the
 * trade — 200 embeds of a small Evermind take seconds on a laptop CPU.
 */
export const CURATION_KNOWN_LIMIT = 200;

/** One document curation left out, and why. */
export interface CurationDrop {
  id: string;
  reason: CuratedOut["reason"];
  of?: string;
  score?: number;
}

export interface CurationResult {
  kept: ExperienceDocument[];
  dropped: CurationDrop[];
  /**
   * Documents an EARLIER curation left out whose verdict still stands (see
   * {@link CuratedOut}) — skipped without being embedded or judged again. Only
   * {@link curatePendingExperience} fills it, from the snapshot's `curated` records.
   */
  alreadyDropped?: CurationDrop[];
}

/** A model store whose kNN for `taskId` leaves the point at `index` out of every vote. */
function leaveOneOut(models: MlModelStore, taskId: string, index: number): MlModelStore {
  return {
    get: (t, e) => {
      const p = models.get(t, e);
      return t === taskId && p instanceof KnnModel && index < p.size ? p.excluding(index) : p;
    },
    set: (t, e, p) => models.set(t, e, p),
  };
}

/** Memoise an embedder for one call, so curation embeds each text once. */
function memoEmbedder(e: TextEmbedder): TextEmbedder {
  const cache = new Map<string, Float32Array>();
  return {
    dimensions: e.dimensions,
    fingerprint: e.fingerprint,
    async embed(text: string) {
      let v = cache.get(text);
      if (!v) cache.set(text, (v = await e.embed(text)));
      return v;
    },
  };
}

/**
 * P(high) of one quality input, or null when it cannot be judged (never run, or no
 * trained engine). Observed rate from {@link QUALITY_OBSERVED_MIN_RUNS} runs; the
 * runner's answer below that, leave-one-out when the document is training example
 * `trainedIndex`.
 */
async function qualityScore(input: ExperienceQualityInput, ctx: MlRunContext, trainedIndex: number | undefined): Promise<number | null> {
  // Never run ⇒ no outcome to judge it by: keep it rather than let a guess drop it.
  if (input.runs <= 0) return null;
  if (input.runs >= QUALITY_OBSERVED_MIN_RUNS) return input.succeeded / input.runs;
  const runCtx = trainedIndex == null || !ctx.models ? ctx : { ...ctx, models: leaveOneOut(ctx.models, EXPERIENCE_QUALITY_TASK.id, trainedIndex) };
  try {
    return (await runMlTask<ExperienceQuality>(EXPERIENCE_QUALITY_TASK.id, input, runCtx)).output.pHigh;
  } catch (error) {
    if (!(error instanceof MlEngineUnavailableError)) throw error;
    return null;
  }
}

/**
 * Curate documents before they are packed into adaptation passes: drop excluded
 * ones, low-quality ones (when a quality filter is given), then near-duplicates of
 * an already-learned one (`known`) or an earlier kept one. Order is preserved, and
 * the FIRST copy wins — compiled skills come before raw demonstrations in
 * `experienceDocuments`, so the reviewed version is the one kept.
 */
export async function curateExperience(docs: readonly ExperienceDocument[], opts: CurationOptions): Promise<CurationResult> {
  const embedder = memoEmbedder(opts.embedder);
  const index = new DedupeIndex(opts.dedupeThreshold ?? DEFAULT_DEDUPE_THRESHOLD);
  /** Index position → document id: the known documents first, then each kept one. */
  const indexed: string[] = [];
  const kept: ExperienceDocument[] = [];
  const dropped: CurationResult["dropped"] = [];
  const qctx = opts.quality ? { ...opts.quality.ctx, embedder } : null;
  const trainedAt = new Map((opts.quality?.trainedIds ?? []).map((id, i) => [id, i] as const));
  const embedDoc = (doc: ExperienceDocument) => embedder.embed(EXPERIENCE_DEDUPE_TASK.features.text!(doc));

  for (const doc of opts.known ?? []) {
    index.add(await embedDoc(doc));
    indexed.push(doc.id);
  }

  for (const doc of docs) {
    if (opts.exclude?.has(doc.id)) {
      dropped.push({ id: doc.id, reason: "excluded" });
      continue;
    }
    if (opts.quality && qctx) {
      const pHigh = await qualityScore(opts.quality.input(doc), qctx, trainedAt.get(doc.id));
      if (pHigh != null && pHigh < (opts.quality.minPHigh ?? 0.5)) {
        dropped.push({ id: doc.id, reason: "low-quality", score: pHigh });
        continue;
      }
    }
    const x = await embedDoc(doc);
    const d = EXPERIENCE_DEDUPE_TASK.output.decode(index.predict(x));
    if (d.duplicate) {
      dropped.push({ id: doc.id, reason: "duplicate", of: indexed[d.nearest]!, score: d.similarity });
      continue;
    }
    index.add(x);
    indexed.push(doc.id);
    kept.push(doc);
  }
  return { kept, dropped };
}

/** What {@link curatePendingExperience} reads: the experience snapshot's raw records. */
export interface ExperienceCurationSource {
  episodes: Episode[];
  skills: Skill[];
  runs: readonly Run[];
  /** Earlier curations' verdicts (the snapshot's `curated`); a standing one is skipped. */
  curated?: Readonly<Record<string, CuratedOut>>;
}

/**
 * Whether an earlier `curated_out` verdict still stands (see {@link CuratedOut}): a
 * duplicate while what it repeats is learned (in `learnedIds`), a low-quality one
 * while its finished-run count is unchanged. An exclusion is always reconsidered.
 */
export function curationStands(verdict: CuratedOut | undefined, finishedRuns: number, learnedIds: ReadonlySet<string>): boolean {
  if (!verdict || verdict.status !== "curated_out") return false;
  if (verdict.reason === "duplicate") return verdict.of != null && learnedIds.has(verdict.of);
  if (verdict.reason === "low-quality") return verdict.runs === finishedRuns;
  return false;
}

/**
 * The whole curation step a host runs in front of the adaptation passes, so every
 * host wires it the same way:
 *
 *   1. skip each pending document whose earlier `curated_out` verdict still stands
 *      ({@link curationStands}) — reported in `alreadyDropped`, never re-embedded;
 *   2. train `experience.quality` on the run outcomes of at most
 *      {@link CURATION_KNOWN_LIMIT} scored procedures — the pending ones, then the
 *      learned ones most recently learned first (it stays untrained below its
 *      `minSamples`, and then only the observed-rate rule drops for quality);
 *   3. {@link curateExperience} the rest with it — leave-one-out for a document in
 *      that training set — de-duplicating against each other and against `known`:
 *      the already-learned documents, MOST RECENTLY LEARNED FIRST; only the first
 *      {@link CURATION_KNOWN_LIMIT} are embedded, so a host can pass its whole ledger.
 */
export async function curatePendingExperience(
  pending: readonly ExperienceDocument[],
  source: ExperienceCurationSource,
  opts: { embedder: TextEmbedder; known?: readonly ExperienceDocument[]; dedupeThreshold?: number; minPHigh?: number },
): Promise<CurationResult> {
  const known = opts.known?.slice(0, CURATION_KNOWN_LIMIT);
  const embedder = memoEmbedder(opts.embedder);
  const inputs = experienceQualityInputs(source.episodes, source.skills, source.runs);
  const byId = new Map(inputs.map((i) => [i.doc.id, i] as const));
  // A document with no joined input (not in the snapshot) counts as never run.
  const inputOf = (doc: ExperienceDocument): ExperienceQualityInput => byId.get(doc.id) ?? { doc, steps: 0, runs: 0, succeeded: 0, failed: 0 };

  const learnedIds = new Set((opts.known ?? []).map((d) => d.id));
  const alreadyDropped: CurationDrop[] = [];
  const toCurate: ExperienceDocument[] = [];
  for (const doc of pending) {
    const verdict = source.curated?.[doc.id];
    if (verdict && curationStands(verdict, inputOf(doc).runs, learnedIds)) {
      alreadyDropped.push({ id: doc.id, reason: verdict.reason, ...(verdict.of != null ? { of: verdict.of } : {}), ...(verdict.score != null ? { score: verdict.score } : {}) });
    } else toCurate.push(doc);
  }

  // The bounded training set: newest first (pending, then most recently learned),
  // scored only, then reversed — trainMlTask holds out the chronological tail.
  const recentFirst = [...pending, ...(opts.known ?? [])]
    .map((d) => byId.get(d.id))
    .filter((i): i is ExperienceQualityInput => i != null && experienceQualityLabel(i) != null)
    .slice(0, CURATION_KNOWN_LIMIT);
  const train = experienceQualityExamples(recentFirst.reverse());

  const registry = new MlTaskRegistry();
  registry.register(EXPERIENCE_QUALITY_TASK);
  const models = new InMemoryMlModels();
  const fitted = await trainMlTask(EXPERIENCE_QUALITY_TASK, train, { embedder, models });
  const r = await curateExperience(toCurate, {
    embedder,
    ...(known ? { known } : {}),
    ...(opts.dedupeThreshold != null ? { dedupeThreshold: opts.dedupeThreshold } : {}),
    quality: {
      ctx: { registry, mode: "local-only", models },
      input: inputOf,
      ...(opts.minPHigh != null ? { minPHigh: opts.minPHigh } : {}),
      // The refit on all examples keeps their order, so kNN point i is train[i].
      ...(fitted ? { trainedIds: train.map((e) => e.input.doc.id) } : {}),
    },
  });
  return { ...r, alreadyDropped };
}

/**
 * The `curated_out` records a host persists (`ExperienceStore.recordCuration`) for one
 * curation result: each dropped document with its verdict and finished-run count, and
 * the kept ids (whose earlier verdicts clear).
 */
export function curationRecords(result: CurationResult, source: Pick<ExperienceCurationSource, "runs">, at: number): CurationRecord {
  const finished = new Map<string, number>();
  for (const run of source.runs) if (run.status !== "running") finished.set(run.skillId, (finished.get(run.skillId) ?? 0) + 1);
  const dropped: Record<string, CuratedOut> = {};
  for (const d of result.dropped) {
    dropped[d.id] = {
      status: "curated_out",
      reason: d.reason,
      ...(d.of != null ? { of: d.of } : {}),
      ...(d.score != null ? { score: d.score } : {}),
      runs: finished.get(d.id) ?? 0,
      at,
    };
  }
  return { dropped, kept: result.kept.map((d) => d.id) };
}
