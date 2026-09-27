/**
 * experience/trainOnce.ts — Train Once: one reviewed demonstration becomes a skill that
 * runs again with new values.
 *
 * Deterministic, local, no model call. Every value the person typed becomes a named
 * parameter (named after the field it went into), every secret field a secret parameter,
 * and every irreversible click an approval gate. The review edits decide what stays.
 */
import { elementLabel, needsApproval } from './approval.js';
import { newExperienceId, type Episode, type Skill, type SkillAction, type SkillParam, type SkillStep } from './types.js';

/** What the person changed in review, keyed by recorded step id. */
export interface ReviewEdits {
  /** The skill's name (defaults to the episode's). */
  name?: string | null;
  /** Steps left out of the skill. */
  removed?: string[];
  /** Typed values kept exactly as recorded instead of asked for each run. */
  fixed?: string[];
  /** Approval gate per step, where the person overrode the default. */
  approvals?: Record<string, boolean>;
  /** Parameter names the person chose, per step. */
  paramNames?: Record<string, string>;
}

export function trainOnce(episode: Episode, edits: ReviewEdits = {}, now: number = Date.now()): Skill {
  const removed = new Set(edits.removed ?? []);
  const fixed = new Set(edits.fixed ?? []);
  const params: SkillParam[] = [];
  const steps: SkillStep[] = [];
  for (const rec of episode.steps) {
    if (removed.has(rec.id)) continue;
    const a = rec.action;
    let action: SkillAction;
    if (a.kind === 'setValue') {
      if (!a.secret && fixed.has(rec.id)) {
        action = { kind: 'setValue', target: a.target, value: { from: 'literal', text: a.value } };
      } else {
        const chosen = slug(edits.paramNames?.[rec.id] ?? '');
        const base = chosen || slug(elementLabel(a.target)) || 'value';
        const name = uniqueName(params, base);
        params.push({ name, label: elementLabel(a.target), default: a.secret ? null : a.value, secret: !!a.secret });
        action = { kind: 'setValue', target: a.target, value: { from: 'param', name } };
      }
    } else {
      action = a;
    }
    steps.push({
      id: rec.id,
      action,
      requiresApproval: edits.approvals?.[rec.id] ?? needsApproval(a),
      screenshot: rec.screenshot ?? null,
    });
  }
  return {
    id: newExperienceId(),
    name: edits.name?.trim() || episode.name,
    program: episode.program,
    args: [...episode.args],
    params,
    steps,
    sourceEpisode: episode.id,
    createdAt: now,
    routine: null,
  };
}

/** `Invoice Amount (USD)` → `invoice_amount_usd`. Letters in any script survive. */
export function slug(label: string): string {
  return label
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter(Boolean)
    .join('_')
    .slice(0, 40)
    .replace(/_+$/, '');
}

function uniqueName(params: SkillParam[], base: string): string {
  if (!params.some((p) => p.name === base)) return base;
  for (let i = 2; ; i++) {
    const n = `${base}_${i}`;
    if (!params.some((p) => p.name === n)) return n;
  }
}
