/**
 * ml/tasks/experience.ts — `experience.quality` and `experience.dedupe`, and the
 * opt-in curation step that applies them in front of the ONE adaptation recipe
 * (`evermind/adapt.ts`), so the private model learns from good, varied examples
 * instead of every transcript. Curation only drops documents; it never rewrites one.
 */
import { experienceDocuments, type ExperienceDocument } from "../../experience/corpus.js";
import type { Episode, Run, Skill } from "../../experience/types.js";
import { KnnModel } from "../classical/knn.js";
import { clampRatio, logScale, type TextEmbedder } from "../features.js";
import { MlEngineUnavailableError, runMlTask, type MlRunContext } from "../run.js";
import type { MlTrainExample } from "../train.js";
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
    numeric: (i) => [logScale(i.steps, 50), logScale(i.runs, 20), clampRatio(i.succeeded, i.runs, 1), clampRatio(i.failed, i.runs, 1)],
    numericNames: ["steps", "runs", "successRate", "failureRate"],
  },
  engines: [
    { tier: "local", engine: "classical", algorithm: "knn" },
    { tier: "local", engine: "classical", algorithm: "logistic" },
  ],
  train: { minSamples: 4 },
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
   * (default 0.5) is dropped; one the runner cannot score (no trained engine) is kept.
   */
  quality?: {
    ctx: MlRunContext;
    input: (doc: ExperienceDocument) => ExperienceQualityInput;
    minPHigh?: number;
  };
  /** Document ids to leave out (e.g. those whose runs `run.anomaly` flagged). */
  exclude?: ReadonlySet<string>;
}

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
 * an earlier kept one. Order is preserved, and the FIRST copy wins — compiled
 * skills come before raw demonstrations in `experienceDocuments`, so the reviewed
 * version is the one kept.
 */
export async function curateExperience(docs: readonly ExperienceDocument[], opts: CurationOptions): Promise<CurationResult> {
  const embedder = memoEmbedder(opts.embedder);
  const index = new DedupeIndex(opts.dedupeThreshold ?? DEFAULT_DEDUPE_THRESHOLD);
  const kept: ExperienceDocument[] = [];
  const dropped: CurationResult["dropped"] = [];
  const qctx = opts.quality ? { ...opts.quality.ctx, embedder } : null;

  for (const doc of docs) {
    if (opts.exclude?.has(doc.id)) {
      dropped.push({ id: doc.id, reason: "excluded" });
      continue;
    }
    if (opts.quality && qctx) {
      try {
        const r = await runMlTask<ExperienceQuality>(EXPERIENCE_QUALITY_TASK.id, opts.quality.input(doc), qctx);
        if (r.output.pHigh < (opts.quality.minPHigh ?? 0.5)) {
          dropped.push({ id: doc.id, reason: "low-quality", score: r.output.pHigh });
          continue;
        }
      } catch (error) {
        if (!(error instanceof MlEngineUnavailableError)) throw error;
      }
    }
    const x = await embedder.embed(EXPERIENCE_DEDUPE_TASK.features.text!(doc));
    const d = EXPERIENCE_DEDUPE_TASK.output.decode(index.predict(x));
    if (d.duplicate) {
      dropped.push({ id: doc.id, reason: "duplicate", of: kept[d.nearest]!.id, score: d.similarity });
      continue;
    }
    index.add(x);
    kept.push(doc);
  }
  return { kept, dropped };
}
