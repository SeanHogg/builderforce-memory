/**
 * experience/types.ts — what Evermind has DONE, as opposed to what it knows.
 *
 * Facts live in `@seanhogg/builderforce-memory` (MemoryStore / EvermindCognition) and
 * weights in the model; this is the third kind of learned item: experience.
 *   - an **Episode** is one demonstration — a person doing a task in a program, recorded
 *     as semantic steps (the UI element acted on and the value set, never raw keystrokes);
 *   - a **Skill** is what Train Once compiles from an episode — a procedure that runs
 *     again with new values, with approval gates on irreversible steps;
 *   - a **Run** is one execution of a skill, with its step-by-step audit trail.
 *
 * Plain data, JSON-shaped (camelCase, actions tagged by `kind`), so any recorder — the
 * native Windows UI Automation one in Synapse included — produces it and any runtime
 * persists it. Timestamps are epoch milliseconds.
 */

/** A UI element, identified the way the platform's accessibility API sees it. */
export interface ElementRef {
  name: string;
  automationId: string;
  /** `Button`, `Edit`, `MenuItem`, … */
  controlType: string;
  className: string;
  /** The top-level window the element sits in. */
  windowTitle: string;
  processName: string;
  /** Where it was acted on, relative to the window's top-left — the replay fallback. */
  relX: number;
  relY: number;
}

/** One thing the person did. A secret field records that it was filled, never the value. */
export type Action =
  | { kind: 'click'; target: ElementRef }
  | { kind: 'setValue'; target: ElementRef; value: string; secret?: boolean }
  | { kind: 'keys'; keys: string; target?: ElementRef | null };

export interface RecordedStep {
  id: string;
  /** Milliseconds since the recording started. */
  atMs: number;
  action: Action;
  /** Screenshot file name inside the episode's folder, if one was taken. */
  screenshot?: string | null;
}

export interface Episode {
  id: string;
  name: string;
  program: string;
  args: string[];
  startedAt: number;
  endedAt?: number | null;
  steps: RecordedStep[];
}

export interface EpisodeSummary {
  id: string;
  name: string;
  program: string;
  startedAt: number;
  steps: number;
}

/** A value a skill asks for at run time. */
export interface SkillParam {
  /** Stable key, e.g. `amount`. */
  name: string;
  /** What the field was called where it was typed, e.g. `Amount`. */
  label: string;
  /** The value from the demonstration; never set for a secret. */
  default?: string | null;
  secret: boolean;
}

export type SkillValue = { from: 'param'; name: string } | { from: 'literal'; text: string };

export type SkillAction =
  | { kind: 'click'; target: ElementRef }
  | { kind: 'setValue'; target: ElementRef; value: SkillValue }
  | { kind: 'keys'; keys: string; target?: ElementRef | null };

export interface SkillStep {
  id: string;
  action: SkillAction;
  /** Pause and ask the person before this step. */
  requiresApproval: boolean;
  screenshot?: string | null;
}

/** When a skill runs by itself. Local time of day for `day`. */
export type Schedule = { every: 'minutes'; minutes: number } | { every: 'day'; hour: number; minute: number };

/** A skill's routine: at most one per skill, so it lives on the skill. */
export interface SkillRoutine {
  schedule: Schedule;
  /** Values for the non-secret parameters; secrets come from the host's vault. */
  values: Record<string, string>;
  enabled: boolean;
  lastRunAt?: number | null;
}

export interface Skill {
  id: string;
  name: string;
  program: string;
  args: string[];
  params: SkillParam[];
  steps: SkillStep[];
  sourceEpisode: string;
  createdAt: number;
  routine?: SkillRoutine | null;
}

export type RunStatus = 'running' | 'succeeded' | 'failed' | 'stopped' | 'denied';

export type StepOutcome = 'ok' | 'fallback' | 'failed' | 'approved' | 'denied';

/** One line of a run's audit trail. A step can have two (approved, then ok). */
export interface RunStepLog {
  idx: number;
  stepId: string;
  outcome: StepOutcome;
  detail?: string | null;
  at: number;
}

export interface Run {
  id: string;
  skillId: string;
  skillName: string;
  /** `manual` or `routine`. */
  trigger: string;
  status: RunStatus;
  startedAt: number;
  endedAt?: number | null;
  error?: string | null;
  steps: RunStepLog[];
}

/**
 * One adaptation of the private model on experience: what it learned and how the fit
 * went. Measured when the adaptation ran, never estimated afterwards.
 */
export interface Adaptation {
  /** N of the `+expN` version this adaptation wrote. */
  index: number;
  version: string;
  /** Episode and skill ids it learned. */
  learned: string[];
  passes: number;
  /** Mean fit loss over the passes. */
  loss: number;
  at: number;
}

/** A short random id — unique enough for one person's store. */
export function newExperienceId(): string {
  const bytes = new Uint8Array(8);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}
