/**
 * Experience — the learned items that are not facts or weights: demonstrations
 * (episodes), the skills Train Once compiles from them, and the runs of those skills.
 * Pure and storage-agnostic, like the rest of the engine.
 */
export type {
  ElementRef,
  Action,
  RecordedStep,
  Episode,
  EpisodeSummary,
  SkillParam,
  SkillValue,
  SkillAction,
  SkillStep,
  Schedule,
  SkillRoutine,
  Skill,
  RunStatus,
  StepOutcome,
  RunStepLog,
  Run,
} from './types.js';
export { newExperienceId } from './types.js';
export { isIrreversibleLabel, elementLabel, needsApproval } from './approval.js';
export { trainOnce, slug } from './trainOnce.js';
export type { ReviewEdits } from './trainOnce.js';
export { isDue, dueSkills } from './schedule.js';
export { skillText, episodeText, experienceCorpus, experienceDocuments, packDocuments } from './corpus.js';
export type { ExperienceDocument } from './corpus.js';
export { EXPERIENCE_SCHEMA, MAX_RUNS, emptySnapshot, parseSnapshot, InMemoryExperienceStore } from './store.js';
export type { ExperienceSnapshot, ExperienceStore } from './store.js';
