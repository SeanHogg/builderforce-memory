/**
 * knowledge/reviewRun.ts — the PURE recorder of a review run (PRD 41 A8): fold the events a review
 * emits (`onProgress`, `onStep`, the end) into a {@link ReviewRun} a host can persist and a view can
 * draw — phase timeline, worker lanes, the last steps, every failure, the rate and ETA.
 *
 * No clock, no I/O: every function takes the time it should use, so the reducer is deterministic and a
 * stored run replays the same. The host owns persistence (last N runs per root) and throttling.
 */

import type { ReviewProgress, ReviewReport } from './reviewModel.js';
import type { ReviewFailure, ReviewPhase, ReviewStep } from './reviewRunModel.js';

export type ReviewRunKind = 'review' | 're-review' | 'seed';
export type ReviewRunOutcome = 'done' | 'stopped' | 'failed';

/** How many of the latest steps a run keeps (failures are kept whole). */
export const REVIEW_RUN_STEP_RING = 200;

export interface ReviewRunPhase {
    phase: ReviewPhase;
    startedAt: number;
    endedAt?: number;
    /** Items reached: files for extract · judge · write (the review's progress `done` is a 0-based index), else as reported. */
    done: number;
    total: number;
}

/** One worker of a phase (extraction workers, judge lanes): the file it is on (or last was on), files finished and time spent busy. */
export interface ReviewRunLane {
    phase: ReviewPhase;
    id: number;
    path?: string;
    done: number;
    busyMs: number;
}

export interface ReviewRun {
    id: string;
    root: string;
    kind: ReviewRunKind;
    startedAt: number;
    endedAt?: number;
    /** The time of the latest event: the clock {@link runRate} measures an unfinished phase against. */
    updatedAt: number;
    phases: ReviewRunPhase[];
    lanes: ReviewRunLane[];
    /** The last {@link REVIEW_RUN_STEP_RING} steps, oldest first. */
    steps: ReviewStep[];
    failures: ReviewFailure[];
    outcome?: ReviewRunOutcome;
    report?: ReviewReport;
}

export type ReviewRunEvent =
    | { type: 'progress'; progress: ReviewProgress }
    | { type: 'step'; step: ReviewStep }
    | { type: 'end'; outcome: ReviewRunOutcome; report?: ReviewReport };

export function startRun(a: { id: string; root: string; kind: ReviewRunKind; at: number }): ReviewRun {
    return { id: a.id, root: a.root, kind: a.kind, startedAt: a.at, updatedAt: a.at, phases: [], lanes: [], steps: [], failures: [] };
}

/** Phases whose progress `done` is the index of the file being reached. */
const PER_FILE: ReadonlySet<ReviewPhase> = new Set<ReviewPhase>(['extract', 'judge', 'write']);

function withProgress(phases: ReviewRunPhase[], p: ReviewProgress, at: number): ReviewRunPhase[] {
    const done = PER_FILE.has(p.phase) ? p.done + 1 : p.done;
    const i = phases.findIndex((x) => x.phase === p.phase);
    const prior = i < 0 ? undefined : phases[i]!;
    const next: ReviewRunPhase = {
        phase: p.phase, startedAt: prior?.startedAt ?? at, done: Math.max(prior?.done ?? 0, done), total: p.total,
    };
    // Once ended it stays ended (a stale or repeated event never reopens it) unless the total grew past what is done.
    if (p.phase === 'done' || next.done >= next.total) next.endedAt = prior?.endedAt ?? at;
    if (i < 0) return [...phases, next];
    return phases.map((x, j) => (j === i ? next : x));
}

const PHASE_ORDER: readonly ReviewPhase[] = ['plan', 'extract', 'resolve', 'judge', 'compare', 'write', 'index', 'mapped', 'done'];

/** Lanes are keyed by phase and id: extract workers and judge lanes are different workers. */
function withLane(lanes: ReviewRunLane[], step: ReviewStep): ReviewRunLane[] {
    if (step.lane === undefined) return lanes;
    const same = (l: ReviewRunLane) => l.phase === step.phase && l.id === step.lane;
    const prior = lanes.find(same);
    const next: ReviewRunLane = {
        phase: step.phase, id: step.lane, path: step.path, done: (prior?.done ?? 0) + 1, busyMs: (prior?.busyMs ?? 0) + step.ms,
    };
    return [...lanes.filter((l) => !same(l)), next]
        .sort((a, b) => PHASE_ORDER.indexOf(a.phase) - PHASE_ORDER.indexOf(b.phase) || a.id - b.id);
}

/** Folds one event into the run; the input run is never mutated. */
export function applyRunEvent(run: ReviewRun, event: ReviewRunEvent, at: number): ReviewRun {
    const base = { ...run, updatedAt: at };
    switch (event.type) {
        case 'progress': {
            const phases = withProgress(run.phases, event.progress, at);
            // A new phase reached: the plan is over (it is never revisited).
            return { ...base, phases: phases.map((p) => (p.phase === 'plan' && p.endedAt === undefined ? { ...p, endedAt: at } : p)) };
        }
        case 'step': {
            const s = event.step;
            return {
                ...base,
                lanes: withLane(run.lanes, s),
                steps: [...run.steps, s].slice(-REVIEW_RUN_STEP_RING),
                failures: s.outcome === 'failed' ? [...run.failures, { path: s.path, phase: s.phase, reason: s.reason ?? 'failed' }] : run.failures,
            };
        }
        case 'end':
            return {
                ...base, endedAt: at, outcome: event.outcome, ...(event.report ? { report: event.report } : {}),
                phases: run.phases.map((p) => (p.endedAt === undefined ? { ...p, endedAt: at } : p)),
            };
    }
}

const phaseOf = (run: ReviewRun, phase: ReviewPhase): ReviewRunPhase | undefined => run.phases.find((p) => p.phase === phase);

/** Items per second a phase has sustained so far; 0 when it has not started or no time has passed. */
export function runRate(run: ReviewRun, phase: ReviewPhase): number {
    const p = phaseOf(run, phase);
    if (!p) return 0;
    const ms = (p.endedAt ?? run.endedAt ?? run.updatedAt) - p.startedAt;
    return ms > 0 && p.done > 0 ? (p.done * 1000) / ms : 0;
}

/** Milliseconds until a phase finishes at its measured rate; 0 when it is complete, undefined when it cannot be told. */
export function runEta(run: ReviewRun, phase: ReviewPhase): number | undefined {
    const p = phaseOf(run, phase);
    if (!p) return undefined;
    const remaining = p.total - p.done;
    if (remaining <= 0) return 0;
    const rate = runRate(run, phase);
    return rate > 0 ? Math.round((remaining / rate) * 1000) : undefined;
}
