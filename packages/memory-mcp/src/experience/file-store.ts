/**
 * The engine's experience store, made durable on disk.
 *
 * The engine owns experience — the types, Train Once, the snapshot format and the
 * reference `InMemoryExperienceStore`. This host adds only what a pure package
 * cannot: the snapshot lives in `experience.json` beside `memory.json`, shared by
 * every process on the machine the same way memories are (one {@link SharedJsonFile}
 * implementation, so the freshness check and the loop guard are the memory store's),
 * and each episode's screenshots live in `episodes/<id>/`, removed when it is forgotten.
 */

import type {
    Episode,
    EpisodeSummary,
    ExperienceSnapshot,
    ExperienceStore,
    InMemoryExperienceStore,
    Run,
    RunStepLog,
    Skill,
} from "@seanhogg/builderforce-memory-engine";
import type { SharedJsonFile } from "../persistence/shared-json-file.js";

/** The filesystem calls the store makes besides the snapshot's own. */
export interface EpisodeFolders {
    /** Delete one episode's screenshot folder, if it exists. Only ever called with a safe id. */
    remove(episodeId: string): void;
    /** Delete every episode's screenshot folder. */
    removeAll(): void;
}

/** Episode ids are also folder names — anything that could leave the folder is refused. */
export function isSafeExperienceId(id: string): boolean {
    return /^[A-Za-z0-9_-]{1,64}$/.test(id);
}

export class FileExperienceStore implements ExperienceStore {
    constructor(
        private readonly inner: InMemoryExperienceStore,
        private readonly file: SharedJsonFile,
        private readonly parse: (raw: unknown) => ExperienceSnapshot,
        private readonly folders: EpisodeFolders,
    ) {}

    /**
     * Absorb the file when another process rewrote it. Also the boot hydration. A file
     * that does not parse is left unabsorbed (a half-written snapshot): the store keeps
     * what it has and the next call looks again.
     */
    private fresh(): void {
        const changed = this.file.readIfChanged();
        if (!changed) return;
        let raw: unknown;
        try {
            raw = JSON.parse(changed.text);
        } catch {
            return;
        }
        this.inner.load(this.parse(raw));
        changed.accept();
    }

    async putEpisode(episode: Episode): Promise<void> {
        this.fresh();
        await this.inner.putEpisode(episode);
    }
    async getEpisode(id: string): Promise<Episode | undefined> {
        this.fresh();
        return this.inner.getEpisode(id);
    }
    async listEpisodes(): Promise<EpisodeSummary[]> {
        this.fresh();
        return this.inner.listEpisodes();
    }
    async deleteEpisode(id: string): Promise<boolean> {
        this.fresh();
        const ok = await this.inner.deleteEpisode(id);
        if (isSafeExperienceId(id)) this.folders.remove(id);
        return ok;
    }

    async putSkill(skill: Skill): Promise<void> {
        this.fresh();
        await this.inner.putSkill(skill);
    }
    async getSkill(id: string): Promise<Skill | undefined> {
        this.fresh();
        return this.inner.getSkill(id);
    }
    async listSkills(): Promise<Skill[]> {
        this.fresh();
        return this.inner.listSkills();
    }
    async deleteSkill(id: string): Promise<boolean> {
        this.fresh();
        return this.inner.deleteSkill(id);
    }

    async putRun(run: Run): Promise<void> {
        this.fresh();
        await this.inner.putRun(run);
    }
    async appendRunStep(runId: string, step: RunStepLog): Promise<boolean> {
        this.fresh();
        return this.inner.appendRunStep(runId, step);
    }
    async getRun(id: string): Promise<Run | undefined> {
        this.fresh();
        return this.inner.getRun(id);
    }
    async listRuns(limit: number): Promise<Run[]> {
        this.fresh();
        return this.inner.listRuns(limit);
    }

    async markLearned(ids: string[], at: number): Promise<void> {
        this.fresh();
        await this.inner.markLearned(ids, at);
    }

    async snapshot(): Promise<ExperienceSnapshot> {
        this.fresh();
        return this.inner.snapshot();
    }
    async clear(): Promise<void> {
        await this.inner.clear();
        this.folders.removeAll();
    }
}
