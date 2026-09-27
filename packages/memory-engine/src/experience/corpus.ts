/**
 * experience/corpus.ts — experience as text Evermind can learn from.
 *
 * Evermind already learns from text through ONE recipe (`evermind/adapt.ts`
 * `adaptAndDiff`: fit a private copy of a version, diff it). Rather than a second,
 * trajectory-shaped training path, each demonstration is rendered as the procedure a
 * person would write down — the task, then numbered steps — and handed to that recipe.
 * Values that became parameters are written as `<name>` placeholders, so the model
 * learns the procedure rather than one day's numbers. Secrets never appear: a secret
 * field's value was never recorded.
 */
import { elementLabel } from './approval.js';
import type { Action, Episode, Skill, SkillAction } from './types.js';

function where(a: Action | SkillAction): string {
  const w = 'target' in a && a.target ? a.target.windowTitle : '';
  return w ? ` in “${w}”` : '';
}

function stepSentence(a: Action | SkillAction): string {
  switch (a.kind) {
    case 'click':
      return `Click “${elementLabel(a.target)}”${where(a)}.`;
    case 'keys':
      return a.target ? `Press ${a.keys} in “${elementLabel(a.target)}”${where(a)}.` : `Press ${a.keys}.`;
    case 'setValue': {
      const field = `“${elementLabel(a.target)}”`;
      if (typeof a.value === 'string') {
        return 'secret' in a && a.secret ? `Enter the secret for ${field}${where(a)}.` : `Set ${field} to “${a.value}”${where(a)}.`;
      }
      return a.value.from === 'param' ? `Set ${field} to <${a.value.name}>${where(a)}.` : `Set ${field} to “${a.value.text}”${where(a)}.`;
    }
  }
}

function programName(program: string): string {
  return program.split(/[\\/]/).pop() ?? program;
}

/** A skill as a written procedure (parameters as placeholders, approvals marked). */
export function skillText(skill: Skill): string {
  const lines = [`Task: ${skill.name} (${programName(skill.program)})`];
  skill.steps.forEach((s, i) => lines.push(`${i + 1}. ${stepSentence(s.action)}${s.requiresApproval ? ' Ask before doing this.' : ''}`));
  return lines.join('\n');
}

/** A raw demonstration as a written procedure. */
export function episodeText(episode: Episode): string {
  const lines = [`Task: ${episode.name} (${programName(episode.program)})`];
  episode.steps.forEach((s, i) => lines.push(`${i + 1}. ${stepSentence(s.action)}`));
  return lines.join('\n');
}

/**
 * The corpus to learn from: every skill as a procedure, plus demonstrations that have
 * not been compiled into one yet (a compiled skill supersedes its raw episode — the
 * reviewed version is the one worth learning).
 */
export function experienceCorpus(episodes: Episode[], skills: Skill[]): string {
  return experienceDocuments(episodes, skills)
    .map((d) => d.text)
    .join('\n\n');
}

/** One learnable procedure and the experience item it came from. */
export interface ExperienceDocument {
  id: string;
  text: string;
}

/** The corpus as separate procedures, each tagged with its source item's id. */
export function experienceDocuments(episodes: Episode[], skills: Skill[]): ExperienceDocument[] {
  const compiled = new Set(skills.map((s) => s.sourceEpisode));
  return [
    ...skills.map((s) => ({ id: s.id, text: skillText(s) })),
    ...episodes.filter((e) => !compiled.has(e.id)).map((e) => ({ id: e.id, text: episodeText(e) })),
  ];
}

/**
 * Pack procedures into passes of at most `maxChars` (one adaptation pass reads that
 * much), never splitting a procedure — one longer than a pass gets a pass of its own.
 */
export function packDocuments(docs: ExperienceDocument[], maxChars: number): ExperienceDocument[][] {
  const passes: ExperienceDocument[][] = [];
  let cur: ExperienceDocument[] = [];
  let len = 0;
  for (const d of docs) {
    const add = cur.length ? d.text.length + 2 : d.text.length;
    if (cur.length && len + add > maxChars) {
      passes.push(cur);
      cur = [];
      len = 0;
      cur.push(d);
      len = d.text.length;
      continue;
    }
    cur.push(d);
    len += add;
  }
  if (cur.length) passes.push(cur);
  return passes;
}
