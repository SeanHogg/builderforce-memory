/**
 * experience/store.ts — the persistence port for experience, and its snapshot format.
 *
 * The engine does no I/O (checkpoints are `ArrayBuffer`s the host stores); experience
 * follows the same rule. `ExperienceStore` is the port every host implements over its own
 * storage — a JSON file on disk (memory-mcp), IndexedDB, a KV. `InMemoryExperienceStore`
 * is the reference implementation, and `ExperienceSnapshot` is the one serialized shape,
 * so a store on one runtime can be read by another.
 */
import type { Adaptation, Episode, EpisodeSummary, Run, RunStepLog, Skill } from './types.js';

export const EXPERIENCE_SCHEMA = 'evermind.experience/1';
/** Runs kept, newest first — the audit trail is recent history, not an archive. */
export const MAX_RUNS = 500;
/** Adaptations kept, newest first — enough history to chart how learning went. */
export const MAX_ADAPTATIONS = 200;

export interface ExperienceSnapshot {
  schema: typeof EXPERIENCE_SCHEMA;
  episodes: Episode[];
  skills: Skill[];
  runs: Run[];
  /**
   * Which episodes and skills the local Evermind has been adapted on, and by which
   * adaptation (the N of the model's `+expN` version) — so training reads each
   * procedure once, and a model rolled back to an earlier N re-learns what the
   * discarded versions had learned.
   */
  learned: Record<string, number>;
  /** Every adaptation's measurements, newest first (capped at {@link MAX_ADAPTATIONS}). */
  adaptations: Adaptation[];
}

export interface ExperienceStore {
  putEpisode(episode: Episode): Promise<void>;
  getEpisode(id: string): Promise<Episode | undefined>;
  listEpisodes(): Promise<EpisodeSummary[]>;
  /** Forget a demonstration. Skills compiled from it keep working — they carry their own steps. */
  deleteEpisode(id: string): Promise<boolean>;

  putSkill(skill: Skill): Promise<void>;
  getSkill(id: string): Promise<Skill | undefined>;
  listSkills(): Promise<Skill[]>;
  deleteSkill(id: string): Promise<boolean>;

  putRun(run: Run): Promise<void>;
  /** Append one audit line to a run; false when the run is unknown. */
  appendRunStep(runId: string, step: RunStepLog): Promise<boolean>;
  getRun(id: string): Promise<Run | undefined>;
  listRuns(limit: number): Promise<Run[]>;

  /**
   * Record one adaptation: its items join the learned ledger under its index (the
   * model's `+expN`) and its measurements join the history — one write, so the ledger
   * and the chart can never disagree.
   */
  recordAdaptation(adaptation: Adaptation): Promise<void>;

  snapshot(): Promise<ExperienceSnapshot>;
  /** Delete every episode, skill and run. */
  clear(): Promise<void>;
}

export function emptySnapshot(): ExperienceSnapshot {
  return { schema: EXPERIENCE_SCHEMA, episodes: [], skills: [], runs: [], learned: {}, adaptations: [] };
}

/** Parse a snapshot, tolerating a missing or foreign file as empty. */
export function parseSnapshot(raw: unknown): ExperienceSnapshot {
  const s = raw as Partial<ExperienceSnapshot> | null;
  if (!s || s.schema !== EXPERIENCE_SCHEMA) return emptySnapshot();
  return {
    schema: EXPERIENCE_SCHEMA,
    episodes: Array.isArray(s.episodes) ? s.episodes : [],
    skills: Array.isArray(s.skills) ? s.skills : [],
    runs: Array.isArray(s.runs) ? s.runs : [],
    learned: s.learned && typeof s.learned === 'object' && !Array.isArray(s.learned) ? s.learned : {},
    // Optional within schema 1: snapshots written before the history existed have none.
    adaptations: Array.isArray(s.adaptations) ? s.adaptations : [],
  };
}

function summary(e: Episode): EpisodeSummary {
  return { id: e.id, name: e.name, program: e.program, startedAt: e.startedAt, steps: e.steps.length };
}

/**
 * The reference store: the snapshot held in memory. Hosts that persist wrap it and
 * write `snapshot()` after each change (see `onChange`).
 */
export class InMemoryExperienceStore implements ExperienceStore {
  private data: ExperienceSnapshot;

  constructor(initial?: ExperienceSnapshot, private readonly onChange?: (s: ExperienceSnapshot) => void | Promise<void>) {
    this.data = initial ? parseSnapshot(initial) : emptySnapshot();
  }

  /** Replace the contents (a host re-reading its file after another process wrote it). */
  load(snapshot: ExperienceSnapshot): void {
    this.data = parseSnapshot(snapshot);
  }

  private async changed(): Promise<void> {
    await this.onChange?.(this.data);
  }

  private static upsert<T extends { id: string }>(list: T[], item: T): void {
    const i = list.findIndex((x) => x.id === item.id);
    if (i >= 0) list[i] = item;
    else list.push(item);
  }

  private static remove<T extends { id: string }>(list: T[], id: string): boolean {
    const i = list.findIndex((x) => x.id === id);
    if (i < 0) return false;
    list.splice(i, 1);
    return true;
  }

  async putEpisode(episode: Episode): Promise<void> {
    InMemoryExperienceStore.upsert(this.data.episodes, episode);
    await this.changed();
  }
  async getEpisode(id: string): Promise<Episode | undefined> {
    return this.data.episodes.find((e) => e.id === id);
  }
  async listEpisodes(): Promise<EpisodeSummary[]> {
    return [...this.data.episodes].sort((a, b) => b.startedAt - a.startedAt).map(summary);
  }
  async deleteEpisode(id: string): Promise<boolean> {
    const ok = InMemoryExperienceStore.remove(this.data.episodes, id);
    delete this.data.learned[id];
    if (ok) await this.changed();
    return ok;
  }

  async putSkill(skill: Skill): Promise<void> {
    InMemoryExperienceStore.upsert(this.data.skills, skill);
    await this.changed();
  }
  async getSkill(id: string): Promise<Skill | undefined> {
    return this.data.skills.find((s) => s.id === id);
  }
  async listSkills(): Promise<Skill[]> {
    return [...this.data.skills].sort((a, b) => b.createdAt - a.createdAt);
  }
  async deleteSkill(id: string): Promise<boolean> {
    const ok = InMemoryExperienceStore.remove(this.data.skills, id);
    delete this.data.learned[id];
    if (ok) await this.changed();
    return ok;
  }

  async putRun(run: Run): Promise<void> {
    InMemoryExperienceStore.upsert(this.data.runs, run);
    this.data.runs.sort((a, b) => b.startedAt - a.startedAt);
    this.data.runs.length = Math.min(this.data.runs.length, MAX_RUNS);
    await this.changed();
  }
  async appendRunStep(runId: string, step: RunStepLog): Promise<boolean> {
    const run = this.data.runs.find((r) => r.id === runId);
    if (!run) return false;
    run.steps.push(step);
    await this.changed();
    return true;
  }
  async getRun(id: string): Promise<Run | undefined> {
    return this.data.runs.find((r) => r.id === id);
  }
  async listRuns(limit: number): Promise<Run[]> {
    return this.data.runs.slice(0, Math.max(0, limit));
  }

  async recordAdaptation(adaptation: Adaptation): Promise<void> {
    for (const id of adaptation.learned) this.data.learned[id] = adaptation.index;
    this.data.adaptations.unshift(adaptation);
    this.data.adaptations.length = Math.min(this.data.adaptations.length, MAX_ADAPTATIONS);
    await this.changed();
  }

  async snapshot(): Promise<ExperienceSnapshot> {
    return structuredClone(this.data);
  }
  async clear(): Promise<void> {
    this.data = emptySnapshot();
    await this.changed();
  }
}
