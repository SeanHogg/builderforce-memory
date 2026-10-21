/**
 * ml/tasks/experience.ts — `experience.quality` and `experience.dedupe`, and the
 * opt-in curation step that applies them in front of the ONE adaptation recipe
 * (`evermind/adapt.ts`), so the private model learns from good, varied examples
 * instead of every transcript. Curation only drops documents; it never rewrites one.
 */
import { experienceDocuments, type ExperienceDocument } from "../../experience/corpus.js";
import type { Episode, Run, Skill } from "../../experience/types.js";
import { KnnModel } from "../classical/knn.js";
import { logScale, type TextEmbedder } from "../features.js";
import { MlTaskRegistry } from "../registry.js";
import { InMemoryMlModels, MlEngineUnavailableError, runMlTask, type MlRunContext } from "../run.js";
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
   * Quality filter through the runner. A document whose P(high) is below `minPHigh`
   * (default 0.5) is dropped; one the runner cannot score (no trained engine) is kept,
   * and so is an UNSCORED one (`runs === 0` — never run is unknown, not bad).
   */
  quality?: {
    ctx: MlRunContext;
    input: (doc: ExperienceDocument) => ExperienceQualityInput;
    minPHigh?: number;
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

export interface CurationResult {
  kept: ExperienceDocument[];
  dropped: { id: string; reason: "duplicate" | "low-quality" | "excluded"; of?: string; score?: number }[];
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
      const input = opts.quality.input(doc);
      // Never run ⇒ no outcome to judge it by: keep it rather than let a guess drop it.
      if (input.runs > 0) {
        try {
          const r = await runMlTask<ExperienceQuality>(EXPERIENCE_QUALITY_TASK.id, input, qctx);
          if (r.output.pHigh < (opts.quality.minPHigh ?? 0.5)) {
            dropped.push({ id: doc.id, reason: "low-quality", score: r.output.pHigh });
            continue;
          }
        } catch (error) {
          if (!(error instanceof MlEngineUnavailableError)) throw error;
        }
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
}

/**
 * The whole curation step a host runs in front of the adaptation passes, so every
 * host wires it the same way: train `experience.quality` on the run outcomes there
 * are (it stays untrained below its `minSamples`, and then nothing is dropped for
 * quality), then {@link curateExperience} the pending documents with it,
 * de-duplicating against each other and against `known` — the already-learned
 * documents, MOST RECENTLY LEARNED FIRST; only the first {@link CURATION_KNOWN_LIMIT}
 * are embedded, so a host can pass its whole ledger.
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
  const registry = new MlTaskRegistry();
  registry.register(EXPERIENCE_QUALITY_TASK);
  const models = new InMemoryMlModels();
  await trainMlTask(EXPERIENCE_QUALITY_TASK, experienceQualityExamples(inputs), { embedder, models });
  return curateExperience(pending, {
    embedder,
    ...(known ? { known } : {}),
    ...(opts.dedupeThreshold != null ? { dedupeThreshold: opts.dedupeThreshold } : {}),
    quality: {
      ctx: { registry, mode: "local-only", models },
      // A document with no joined input (not in the snapshot) counts as never run.
      input: (doc) => byId.get(doc.id) ?? { doc, steps: 0, runs: 0, succeeded: 0, failed: 0 },
      ...(opts.minPHigh != null ? { minPHigh: opts.minPHigh } : {}),
    },
  });
}
