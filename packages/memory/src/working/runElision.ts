/**
 * In-run elision — how the drop-oldest window keeps ONE long run inside the history budget.
 *
 * The window ({@link windowed}) drops whole turns from the front, but never past the most
 * recent user turn: a request must open on a user turn, and that turn is the run's task. So
 * inside a single long run nothing after the prompt was ever dropped, and with the summary
 * unavailable (timed out, failed) the prompt grew by every tool result the run produced —
 * VS Code chat #128305 went from 31k to 191k tokens over one run, the 64k budget never
 * applying.
 *
 * This is the cut INSIDE the run. Everything stays — the user turn, every assistant turn (its
 * narration, a plan, the tool calls it made), and every tool result's ROW — but a tool result
 * older than the recent tail has its CONTENT replaced by one line naming the tool, its target
 * and its size ({@link elidedResultStub}). A call and its result are never split (the row
 * stays, only its content shrinks), so strict vendors accept the request; and the model is
 * told what it no longer sees and how to get it back.
 *
 * The cut moves in fixed chunks counted from the run's own start, so the elided prefix — what
 * a provider prefix cache hashes — changes once per chunk of new work, not on every turn.
 *
 * Pure: no `node:*`, no I/O.
 */

import { contentText, defaultEstimateTokens, messageTokens as handoffMessageTokens } from '../handoff/budget.js';
import type { HandoffMessage } from '../handoff/types.js';
import { callsOf, nameOf, stringArgOf, type NamedToolCall } from './fold.js';

/** The budget an in-run cut works to — the transcript module supplies its own constants. */
export interface RunElisionBudget {
  /** The history budget: below it (counting `extraTokens`) nothing is elided. */
  budgetTokens: number;
  /** How much of the run's most recent work stays verbatim once it is over. */
  tailTokens: number;
  /** The step the cut moves in, counted from the run's start (prefix-cache hysteresis). */
  chunkTokens: number;
  /** The fewest most-recent messages kept verbatim, however large. */
  minTailMessages: number;
  /** Tokens the caller sends in front of these messages (a memo note) — charged to the budget. */
  extraTokens?: number;
}

/** A result this small costs less to keep than its stub saves — never elided. */
export const MIN_ELIDED_RESULT_TOKENS = 200;

/** Tools whose results ARE the run's plan or ledger: never elided — the model steers by them. */
const PINNED_RESULT_TOOL = /plan|todo|ledger/i;

/** The arguments a stub names as the call's target, in order of preference. */
const TARGET_ARGS = ['path', 'file_path', 'file', 'pattern', 'query', 'url', 'command'] as const;

/** The longest target a stub quotes. */
const MAX_TARGET_CHARS = 120;

/** Every stub starts with this — a stub is never elided again. */
const STUB_PREFIX = '[result elided:';

/**
 * The one line an elided tool result becomes: the tool, what it was called on, and how big
 * the result was — so the model knows what it no longer sees and that it can fetch it again.
 */
export function elidedResultStub(tool: string, target: string | null, chars: number): string {
  const kb = Math.max(1, Math.round(chars / 1024));
  const on = target ? ` ${target.length > MAX_TARGET_CHARS ? `${target.slice(0, MAX_TARGET_CHARS)}…` : target}` : '';
  return `${STUB_PREFIX} ${tool}${on}, ${kb} KB — re-read if needed]`;
}

function messageTokens(m: HandoffMessage): number {
  return handoffMessageTokens(m, defaultEstimateTokens);
}

/** The call each tool row answers: by `tool_call_id`, else the next unanswered call of the turn before it. */
function callsAnswered(msgs: readonly HandoffMessage[]): Map<number, NamedToolCall> {
  const byId = new Map<string, NamedToolCall>();
  const answered = new Map<number, NamedToolCall>();
  let unanswered: NamedToolCall[] = [];
  msgs.forEach((m, i) => {
    if (m.role === 'assistant') {
      unanswered = callsOf(m).slice();
      for (const call of unanswered) if (typeof call.id === 'string' && call.id.length > 0) byId.set(call.id, call);
      return;
    }
    if (m.role !== 'tool') return;
    const call = (m.tool_call_id ? byId.get(m.tool_call_id) : undefined) ?? unanswered[0];
    if (!call) return;
    unanswered = unanswered.filter((c) => c !== call);
    answered.set(i, call);
  });
  return answered;
}

/**
 * Where the verbatim tail begins: `runStart` while the run's own messages fit the tail, else
 * the first fixed cut point (the message after which the run's running total crosses a
 * multiple of `chunkTokens`) that leaves at most `tailTokens` after it — never inside the
 * last `minTailMessages`.
 */
function runCut(sizes: readonly number[], runStart: number, budget: RunElisionBudget): number {
  let tail = 0;
  for (let i = runStart; i < sizes.length; i++) tail += sizes[i]!;
  let cut = runStart;
  if (tail > budget.tailTokens) {
    let crossed = 0;
    let nextCut = budget.chunkTokens;
    for (let i = runStart; i < sizes.length; i++) {
      crossed += sizes[i]!;
      tail -= sizes[i]!;
      if (crossed < nextCut) continue;
      while (crossed >= nextCut) nextCut += budget.chunkTokens;
      cut = i + 1;
      // The first cut POINT that leaves the tail within budget — the tail may drop under it between two points.
      if (tail <= budget.tailTokens) break;
    }
  }
  return Math.min(cut, Math.max(runStart, sizes.length - budget.minTailMessages));
}

/**
 * Elide the content of the current run's older tool results so `msgs` fits the budget.
 * Returns the input itself (same reference) when it already fits or nothing could be elided;
 * otherwise a new list whose elided rows are shallow copies with only `content` changed.
 *
 * The run is everything after the LAST user turn (all of `msgs` when there is none). Rows
 * before it, the user turn itself, every assistant turn, results under
 * {@link MIN_ELIDED_RESULT_TOKENS}, plan/ledger results and the verbatim tail are kept as sent.
 */
export function elideRunResults<M extends HandoffMessage>(msgs: readonly M[], budget: RunElisionBudget): M[] {
  const sizes = msgs.map(messageTokens);
  const total = sizes.reduce((sum, n) => sum + n, 0) + (budget.extraTokens ?? 0);
  if (total <= budget.budgetTokens) return msgs as M[];
  const runStart = msgs.map((m) => m.role).lastIndexOf('user') + 1;
  const cut = runCut(sizes, runStart, budget);
  if (cut <= runStart) return msgs as M[];
  const answered = callsAnswered(msgs);
  let changed = false;
  const out = msgs.map((m, i) => {
    if (i < runStart || i >= cut || m.role !== 'tool' || sizes[i]! < MIN_ELIDED_RESULT_TOKENS) return m;
    const text = contentText(m.content);
    if (text.startsWith(STUB_PREFIX)) return m;
    const call = answered.get(i);
    const tool = (call && nameOf(call)) ?? 'tool';
    if (PINNED_RESULT_TOOL.test(tool)) return m;
    changed = true;
    return { ...m, content: elidedResultStub(tool, call ? stringArgOf(call, TARGET_ARGS) : null, text.length) } as M;
  });
  return changed ? out : (msgs as M[]);
}
