/**
 * Teach the person's private Evermind what they have done.
 *
 * Experience becomes text through the engine (`experienceDocuments`: each skill or
 * uncompiled demonstration as a written procedure), and the text becomes weights
 * through the engine's ONE adaptation recipe (`adaptPackage`). This module only
 * decides what is new, runs the passes over the local model file and writes the
 * result back, keeping the previous model beside it.
 *
 * "What is new" is the store's learned ledger, and it holds only for the model it was
 * written against: an adapted model carries a `+expN` version suffix, so a model
 * without one is a fresh base (a newly downloaded Evermind) that has learned nothing
 * yet and gets every procedure again.
 */

import type { adaptPackage, experienceDocuments, packDocuments, ExperienceStore } from "@seanhogg/builderforce-memory-engine";
import type { LoadedEvermind } from "../model/evermind-package.js";

/** The engine functions training needs, passed in so the peer stays optional. */
export interface TrainEngine {
    experienceDocuments: typeof experienceDocuments;
    packDocuments: typeof packDocuments;
    adaptPackage: typeof adaptPackage;
    EVERMIND_ADAPT_MAX_CHARS: number;
}

export type TrainOutcome =
    | { status: "nothing_new"; pending: 0; version: string }
    | { status: "no_window"; pending: number; version: string }
    | { status: "trained"; learned: number; passes: number; loss: number; version: string; previousModel: string };

const EXP_SUFFIX = /\+exp(\d+)$/;

/** The version an adaptation writes: `<base>+exp1`, then `+exp2`, … */
export function nextExperienceVersion(version: string): string {
    const m = EXP_SUFFIX.exec(version);
    return m ? version.replace(EXP_SUFFIX, `+exp${Number(m[1]) + 1}`) : `${version}+exp1`;
}

/** Whether this model was adapted on this store's experience (vs. a fresh base). */
export function isExperienceAdapted(version: string): boolean {
    return EXP_SUFFIX.test(version);
}

/** Procedures the model has not learned yet. */
export async function pendingDocuments(store: ExperienceStore, engine: TrainEngine, modelVersion: string) {
    const snap = await store.snapshot();
    const learned = isExperienceAdapted(modelVersion) ? snap.learned : {};
    return engine.experienceDocuments(snap.episodes, snap.skills).filter((d) => !(d.id in learned));
}

export async function trainExperience(
    store: ExperienceStore,
    engine: TrainEngine,
    model: LoadedEvermind,
    modelFile: string,
    now: number = Date.now(),
): Promise<TrainOutcome> {
    const version = model.pkg.manifest.version ?? "0";
    const docs = await pendingDocuments(store, engine, version);
    if (docs.length === 0) return { status: "nothing_new", pending: 0, version };

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
    if (passes === 0) return { status: "no_window", pending: docs.length, version };

    // Keep the model this pass started from, so one bad adaptation is one rename away
    // from undone. The new model is written beside the old one and renamed into place,
    // so a crash mid-write never leaves a truncated model. The ledger is written only
    // after the new model is on disk.
    const previousModel = `${modelFile}.prev`;
    model.fs.copyFileSync(modelFile, previousModel);
    const staged = `${modelFile}.next`;
    model.fs.writeFileSync(staged, new Uint8Array(pkg.toBlob()));
    model.fs.renameSync(staged, modelFile);
    await store.markLearned(learned, now);
    return { status: "trained", learned: learned.length, passes, loss: lossSum / passes, version: next, previousModel };
}
