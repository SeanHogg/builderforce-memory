/**
 * experience/overview.ts — what experience says about the brain, for anything that draws
 * it. Pure: the host passes the snapshot, which adaptation its model is at, the clock
 * and the local UTC offset; every number comes from the snapshot, none is estimated.
 *
 * The regions are the parts of the brain experience lives in:
 *   - **hippocampus** — episodic memory: the demonstrations;
 *   - **basalGanglia** — action selection: the skills, and how their runs went;
 *   - **amygdala** — salience: the decisions made at irreversible steps;
 *   - **hypothalamus** — drives: the routines that start skills by themselves;
 *   - **neocortex** — the private model's weights: the procedures adapted into it.
 * Facts (semantic memory) and attention belong to other stores; a host adds those.
 */
import { experienceDocuments } from './corpus.js';
import type { ExperienceSnapshot } from './store.js';
import type { Adaptation, RunStatus } from './types.js';

export type ExperienceRegion = 'hippocampus' | 'basalGanglia' | 'amygdala' | 'hypothalamus' | 'neocortex';

/** One thing that happened, placed in the region it landed in. */
export interface ExperienceEvent {
  region: ExperienceRegion;
  kind: 'demonstration' | 'skill' | 'run' | 'approval' | 'adaptation';
  /** The episode, skill, run or model version it concerns. */
  id: string;
  name: string;
  at: number;
  /** Run status, approval outcome, or the adaptation's loss — what happened. */
  detail?: string;
}

export interface ExperienceDay {
  /** Local calendar day, `YYYY-MM-DD`. */
  day: string;
  demonstrations: number;
  skills: number;
  runs: number;
  learned: number;
}

export interface ExperienceOverview {
  regions: Record<ExperienceRegion, number>;
  /** Procedures the model at `modelIndex` has learned, and those it has not yet. */
  learned: number;
  pending: number;
  runs: Record<RunStatus, number>;
  approvals: { approved: number; denied: number };
  /** Oldest first — the order a chart reads. */
  adaptations: Adaptation[];
  /** One entry per day of the window, oldest first, empty days included. */
  days: ExperienceDay[];
  /** Newest first. */
  recent: ExperienceEvent[];
}

const DAY_MS = 86_400_000;

export interface OverviewOptions {
  /** N of the loaded model's `+expN` version (0 for a base model or none). */
  modelIndex: number;
  now: number;
  /** Local minus UTC, in minutes. */
  utcOffsetMinutes: number;
  /** Days of activity to chart. Default 30. */
  days?: number;
  /** Events to list. Default 40. */
  recent?: number;
}

function localDay(at: number, offsetMinutes: number): string {
  return new Date(at + offsetMinutes * 60_000).toISOString().slice(0, 10);
}

export function experienceOverview(snap: ExperienceSnapshot, opts: OverviewOptions): ExperienceOverview {
  const known = (id: string) => {
    const n = snap.learned[id];
    return typeof n === 'number' && n >= 1 && n <= opts.modelIndex;
  };
  const docs = experienceDocuments(snap.episodes, snap.skills);
  const learned = docs.filter((d) => known(d.id)).length;

  const runs: Record<RunStatus, number> = { running: 0, succeeded: 0, failed: 0, stopped: 0, denied: 0 };
  const approvals = { approved: 0, denied: 0 };
  const events: ExperienceEvent[] = [];

  for (const e of snap.episodes) {
    events.push({ region: 'hippocampus', kind: 'demonstration', id: e.id, name: e.name, at: e.startedAt });
  }
  for (const s of snap.skills) {
    events.push({ region: 'basalGanglia', kind: 'skill', id: s.id, name: s.name, at: s.createdAt });
  }
  for (const r of snap.runs) {
    runs[r.status] = (runs[r.status] ?? 0) + 1;
    events.push({ region: 'basalGanglia', kind: 'run', id: r.id, name: r.skillName, at: r.startedAt, detail: r.status });
    for (const step of r.steps) {
      if (step.outcome !== 'approved' && step.outcome !== 'denied') continue;
      approvals[step.outcome]++;
      events.push({ region: 'amygdala', kind: 'approval', id: r.id, name: r.skillName, at: step.at, detail: step.outcome });
    }
  }
  // Only adaptations the loaded model descends from: a rolled-back model's discarded
  // versions are not what it knows.
  const adaptations = snap.adaptations.filter((a) => a.index <= opts.modelIndex).reverse();
  for (const a of adaptations) {
    events.push({ region: 'neocortex', kind: 'adaptation', id: a.version, name: a.version, at: a.at, detail: a.loss.toFixed(3) });
  }

  const windowDays = Math.max(1, opts.days ?? 30);
  const days: ExperienceDay[] = [];
  const index = new Map<string, ExperienceDay>();
  for (let i = windowDays - 1; i >= 0; i--) {
    const day = localDay(opts.now - i * DAY_MS, opts.utcOffsetMinutes);
    if (index.has(day)) continue;
    const entry = { day, demonstrations: 0, skills: 0, runs: 0, learned: 0 };
    index.set(day, entry);
    days.push(entry);
  }
  const bump = (at: number, field: Exclude<keyof ExperienceDay, 'day'>, by = 1) => {
    const d = index.get(localDay(at, opts.utcOffsetMinutes));
    if (d) d[field] += by;
  };
  for (const e of snap.episodes) bump(e.startedAt, 'demonstrations');
  for (const s of snap.skills) bump(s.createdAt, 'skills');
  for (const r of snap.runs) bump(r.startedAt, 'runs');
  for (const a of adaptations) bump(a.at, 'learned', a.learned.length);

  return {
    regions: {
      hippocampus: snap.episodes.length,
      basalGanglia: snap.skills.length,
      amygdala: approvals.approved + approvals.denied,
      hypothalamus: snap.skills.filter((s) => s.routine?.enabled).length,
      neocortex: learned,
    },
    learned,
    pending: docs.length - learned,
    runs,
    approvals,
    adaptations,
    days,
    recent: events.sort((a, b) => b.at - a.at).slice(0, Math.max(0, opts.recent ?? 40)),
  };
}
