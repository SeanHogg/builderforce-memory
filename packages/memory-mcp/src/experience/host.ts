/**
 * Stand up experience for a headless server: the engine, the durable store and the
 * private model it trains — or nothing, when experience is not switched on.
 *
 * Opt-in by design, like the Synapse feature it serves: a plain memory server (the
 * one every coding agent installs) never pays for experience tools in its tool list.
 */

import type {
    ExperienceStore,
    InMemoryExperienceStore,
    ExperienceSnapshot,
    dueSkills,
    experienceOverview,
    newExperienceId,
    trainOnce,
} from "@seanhogg/builderforce-memory-engine";
import { dynamicImport } from "../dynamic-import.js";
import { MODEL_FILE_ENV, TOKENIZER_FILE_ENV } from "../embedding/index.js";
import { loadEvermindPackage, readEvermindVersion, type LoadedEvermind } from "../model/evermind-package.js";
import { SharedJsonFile, type SharedFileFs } from "../persistence/shared-json-file.js";
import { FileExperienceStore } from "./file-store.js";
import type { TrainEngine } from "./train.js";

/** '1' turns the experience tools on. */
export const EXPERIENCE_ENV = "BUILDERFORCE_MEMORY_EXPERIENCE";

/** The engine surface experience uses, passed around so the peer stays optional. */
export interface ExperienceEngine extends TrainEngine {
    trainOnce: typeof trainOnce;
    dueSkills: typeof dueSkills;
    newExperienceId: typeof newExperienceId;
    experienceOverview: typeof experienceOverview;
}

export interface ExperienceHost {
    store: ExperienceStore;
    engine: ExperienceEngine;
    /** The memory snapshot experience sits beside (the facts). */
    memoryFile: string;
    /** The experience snapshot file. */
    file: string;
    /** Where each episode's screenshots go: `<episodesDir>/<episodeId>/`. */
    episodesDir: string;
    /** The private model experience trains, when one is configured. */
    modelFile?: string;
    /** Load the model fresh from disk (training rewrites it, so it is never cached). */
    loadModel(): Promise<LoadedEvermind | null>;
    /** The model's version from its manifest alone, or null without a readable one. */
    modelVersion(): Promise<string | null>;
}

type HostFs = SharedFileFs & {
    mkdirSync(path: string, opts: { recursive: boolean }): void;
    rmSync(path: string, opts: { recursive: boolean; force: boolean }): void;
};
type HostPath = { dirname(p: string): string; join(...p: string[]): string };

export interface ExperienceHostOptions {
    env?: Record<string, string | undefined>;
    /** The memory snapshot in effect — experience lives beside it. */
    memoryFile: string;
}

/**
 * The experience host, or `null` when experience is off or the engine peer is
 * absent. Never throws: a memory server must start without it.
 */
export async function createExperienceHost(opts: ExperienceHostOptions): Promise<ExperienceHost | null> {
    const env = opts.env ?? process.env;
    if (env[EXPERIENCE_ENV] !== "1") return null;
    try {
        const engine = (await dynamicImport("@seanhogg/builderforce-memory-engine")) as Partial<
            ExperienceEngine & {
                InMemoryExperienceStore: new (
                    initial?: ExperienceSnapshot,
                    onChange?: (s: ExperienceSnapshot) => void,
                ) => InMemoryExperienceStore;
                parseSnapshot: (raw: unknown) => ExperienceSnapshot;
            }
        >;
        if (!engine?.InMemoryExperienceStore || !engine.parseSnapshot || !engine.trainOnce || !engine.adaptPackage) return null;

        const fs = (await dynamicImport("node:fs")) as HostFs;
        const path = (await dynamicImport("node:path")) as HostPath;
        const dir = path.dirname(opts.memoryFile);
        const file = path.join(dir, "experience.json");
        const episodesDir = path.join(dir, "episodes");
        fs.mkdirSync(episodesDir, { recursive: true });

        const shared = new SharedJsonFile(file, fs);
        const inner = new engine.InMemoryExperienceStore(undefined, (s) => shared.write(JSON.stringify(s, null, 2)));
        const store = new FileExperienceStore(inner, shared, engine.parseSnapshot, {
            remove: (id) => fs.rmSync(path.join(episodesDir, id), { recursive: true, force: true }),
        });
        const modelFile = env[MODEL_FILE_ENV] || undefined;
        const tokenizerFile = env[TOKENIZER_FILE_ENV] || undefined;
        const loadModel = async () => (modelFile ? loadEvermindPackage(modelFile, tokenizerFile) : null);
        const modelVersion = async () => (modelFile ? readEvermindVersion(modelFile) : null);
        return {
            store,
            engine: engine as ExperienceEngine,
            memoryFile: opts.memoryFile,
            file,
            episodesDir,
            modelFile,
            loadModel,
            modelVersion,
        };
    } catch {
        return null;
    }
}
