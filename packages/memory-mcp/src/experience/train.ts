/**
 * Teach the person's private Evermind what they have done.
 *
 * Experience becomes text through the engine (`experienceDocuments`: each skill or
 * uncompiled demonstration as a written procedure), and the text becomes weights
 * through the engine's ONE adaptation recipe (`adaptPackage`). This module only
 * decides what is new, runs the passes over the local model file and writes the
 * result back, keeping the previous model beside it.
 *
 * "What is new" is the store's learned ledger. Every adaptation writes `<base>+expN`
 * and records N against each procedure it learned, so a loaded model trusts exactly
 * the entries at or below its own N: rolling back to `<model>.prev` (+exp N-1)
 * re-teaches what the discarded version had learned, and a fresh base (no suffix,
 * N = 0) — a newly downloaded Evermind — gets every procedure again.
 */

import type {
    adaptPackage,
    curatePendingExperience,
    experienceDocuments,
    packDocuments,
    CurationResult,
    EvermindTextEmbedder,
    ExperienceDocument,
    ExperienceSnapshot,
    ExperienceStore,
} from "@seanhogg/builderforce-memory-engine";
import type { LoadedEvermind } from "../model/evermind-package.js";

/** The engine functions training needs, passed in so the peer stays optional. */
export interface TrainEngine {
    experienceDocuments: typeof experienceDocuments;
    packDocuments: typeof packDocuments;
    adaptPackage: typeof adaptPackage;
    EVERMIND_ADAPT_MAX_CHARS: number;
    /** The opt-in curation step and the CPU embedder it scores with (engine ≥ 2026.10.2). */
    curatePendingExperience?: typeof curatePendingExperience;
    EvermindTextEmbedder?: typeof EvermindTextEmbedder;
}

/** What curation left out of a pass, when it ran. */
export type CurationReport = { dropped: CurationResult["dropped"] };

export type TrainOutcome =
    | { status: "nothing_new"; pending: 0; version: string }
    | { status: "no_window"; pending: number; version: string; curation?: CurationReport }
    | { status: "curated_out"; pending: number; version: string; curation: CurationReport }
    | {
          status: "trained";
          learned: number;
          passes: number;
          loss: number;
          version: string;
          previousModel: string;
          curation?: CurationReport;
      };

const EXP_SUFFIX = /\+exp(\d+)$/;

/** The version an adaptation writes: `<base>+exp1`, then `+exp2`, … */
export function nextExperienceVersion(version: string): string {
    const m = EXP_SUFFIX.exec(version);
    return m ? version.replace(EXP_SUFFIX, `+exp${Number(m[1]) + 1}`) : `${version}+exp1`;
}

/** Which adaptation produced this model: N for `+expN`, 0 for a base that has none. */
export function experienceIndex(version: string): number {
    const m = EXP_SUFFIX.exec(version);
    return m ? Number(m[1]) : 0;
}

/**
 * Split the procedures by what the model at `modelVersion` has learned: `pending` in
 * corpus order, `known` most recently learned first (what curation de-duplicates
 * against — the engine embeds only the most recent `CURATION_KNOWN_LIMIT` of them).
 */
function splitDocuments(snap: ExperienceSnapshot, engine: TrainEngine, modelVersion: string) {
    const current = experienceIndex(modelVersion);
    const learnedAt = (id: string): number | null => {
        const n = snap.learned[id];
        return typeof n === "number" && n >= 1 && n <= current ? n : null;
    };
    const pending: ExperienceDocument[] = [];
    const known: { doc: ExperienceDocument; at: number }[] = [];
    for (const doc of engine.experienceDocuments(snap.episodes, snap.skills)) {
        const at = learnedAt(doc.id);
        if (at == null) pending.push(doc);
        else known.push({ doc, at });
    }
    known.sort((a, b) => b.at - a.at);
    return { pending, known: known.map((k) => k.doc) };
}

/** Procedures the model has not learned yet. */
export async function pendingDocuments(store: ExperienceStore, engine: TrainEngine, modelVersion: string) {
    return splitDocuments(await store.snapshot(), engine, modelVersion).pending;
}

export interface TrainExperienceOptions {
    /**
     * Opt-in curation between "what is new" and the adaptation passes: the engine's
     * `curatePendingExperience` (the `experience.quality` filter, trained on this
     * store's run outcomes, + near-duplicate removal against the batch AND the
     * recently learned procedures — docs/ml-task-catalog.md §4.2), scoring with the
     * model's own CPU embedder. Default false ⇒ every pending procedure is learned,
     * exactly as before. A procedure curation drops is NOT recorded as learned, so a
     * later pass reconsiders it.
     */
    curate?: boolean;
}

/** The message when curation is asked of an engine that predates it. */
export const CURATION_UNSUPPORTED = "Curation needs @seanhogg/builderforce-memory-engine 2026.10.2 or newer.";

export async function trainExperience(
    store: ExperienceStore,
    engine: TrainEngine,
    model: LoadedEvermind,
    modelFile: string,
    opts: TrainExperienceOptions = {},
): Promise<TrainOutcome> {
    const version = model.pkg.manifest.version ?? "0";
    const snap = await store.snapshot();
    const { pending, known } = splitDocuments(snap, engine, version);
    if (pending.length === 0) return { status: "nothing_new", pending: 0, version };

    let docs = pending;
    let curation: CurationReport | undefined;
    if (opts.curate) {
        if (!engine.curatePendingExperience || !engine.EvermindTextEmbedder) throw new Error(CURATION_UNSUPPORTED);
        // The model being trained embeds for curation: its own notion of "the same procedure".
        const lm = model.pkg.loadLM() as ConstructorParameters<typeof EvermindTextEmbedder>[0];
        const codec = model.codec as ConstructorParameters<typeof EvermindTextEmbedder>[1];
        const embedder = new engine.EvermindTextEmbedder(lm, codec);
        const r = await engine.curatePendingExperience(pending, snap, { embedder, known });
        docs = r.kept;
        curation = { dropped: r.dropped };
    }
    if (docs.length === 0) return { status: "curated_out", pending: pending.length, version, curation: curation ?? { dropped: [] } };

    const next = nextExperienceVersion(version);
    // The engine's package type; the structural view is what was loaded from disk.
    let pkg = model.pkg as unknown as Parameters<typeof adaptPackage>[0];
    const learned: string[] = [];
    let lossSum = 0;
    let passes = 0;
    for (const pass of engine.packDocuments(docs, engine.EVERMIND_ADAPT_MAX_CHARS)) {
        const r = engine.adaptPackage(pkg, model.codec, pass.map((d) => d.text).join("\n\n"), next);
        if (!r) continue;
        pkg = r.pkg;
        lossSum += r.loss;
        passes++;
        learned.push(...pass.map((d) => d.id));
    }
    if (passes === 0) return { status: "no_window", pending: docs.length, version, ...(curation ? { curation } : {}) };

    const loss = lossSum / passes;
    // Keep the model this pass started from, so one bad adaptation is one rename away
    // from undone. The new model is written beside the old one and renamed into place,
    // so a crash mid-write never leaves a truncated model. The ledger is written only
    // after the new model is on disk.
    const previousModel = `${modelFile}.prev`;
    model.fs.copyFileSync(modelFile, previousModel);
    const staged = `${modelFile}.next`;
    model.fs.writeFileSync(staged, new Uint8Array(pkg.toBlob()));
    model.fs.renameSync(staged, modelFile);
    await store.recordAdaptation({ index: experienceIndex(next), version: next, learned, passes, loss, at: Date.now() });
    return { status: "trained", learned: learned.length, passes, loss, version: next, previousModel, ...(curation ? { curation } : {}) };
}
