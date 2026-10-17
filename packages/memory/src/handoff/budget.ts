/**
 * Token estimate + budget math for the hand-off packer.
 *
 * The estimate matches Builderforce's gateway heuristic (`workingTranscript.ts`
 * `messageTokens`): a string content's length / 4, a non-string content's JSON length / 4,
 * plus any `tool_calls` payload's JSON length, plus 4 tokens of role/framing overhead.
 */

import type { HandoffBudgetInput, HandoffMessage, TokenEstimator } from './types.js';

/** Per-message role/framing overhead, in tokens. */
export const MESSAGE_OVERHEAD_TOKENS = 4;

/** Default estimator — chars/4, rounded up. */
export const defaultEstimateTokens: TokenEstimator = (text) => Math.ceil(text.length / 4);

/**
 * The tokens a hand-off may use: the next model's window less the reply reserve and the
 * advertised tool schemas. Never negative.
 */
export function handoffBudget(input: HandoffBudgetInput): number {
  const budget = input.targetContextTokens - input.reserveOutputTokens - (input.toolsTokens ?? 0);
  return Math.max(0, Math.floor(budget));
}

/** A message's content as text: the string itself, else its JSON. */
export function contentText(content: unknown): string {
  if (typeof content === 'string') return content;
  return JSON.stringify(content ?? '') ?? '';
}

/** Does this assistant message carry tool calls? */
export function hasToolCalls(m: HandoffMessage): boolean {
  if (m.tool_calls == null) return false;
  return Array.isArray(m.tool_calls) ? m.tool_calls.length > 0 : true;
}

/** Estimated tokens for one message (content + tool-call payload + framing). */
export function messageTokens(m: HandoffMessage, estimate: TokenEstimator): number {
  let text = contentText(m.content);
  if (m.tool_calls != null) text += JSON.stringify(m.tool_calls) ?? '';
  return estimate(text) + MESSAGE_OVERHEAD_TOKENS;
}

/** Estimated tokens for a list of messages. */
export function tokensOf(msgs: readonly HandoffMessage[], estimate: TokenEstimator): number {
  let total = 0;
  for (const m of msgs) total += messageTokens(m, estimate);
  return total;
}

/**
 * The longest prefix of `text` (with a trailing ellipsis when cut) whose estimate is at
 * most `maxTokens`; '' when not even one character fits. Binary search, so it works for
 * any monotone estimator, not only chars/4.
 */
export function fitText(text: string, maxTokens: number, estimate: TokenEstimator): string {
  if (maxTokens <= 0) return '';
  if (estimate(text) <= maxTokens) return text;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (estimate(`${text.slice(0, mid)}…`) <= maxTokens) lo = mid;
    else hi = mid - 1;
  }
  return lo > 0 ? `${text.slice(0, lo).trimEnd()}…` : '';
}

/** A char budget a `recall` callback can honour for `tokens` tokens under `estimate`. */
export function charsForTokens(tokens: number, estimate: TokenEstimator): number {
  if (tokens <= 0) return 0;
  // Calibrate on a probe so a custom estimator gets a proportionate budget; fitText
  // enforces the token ceiling on whatever comes back either way.
  const probe = 'x'.repeat(400);
  const perChar = Math.max(estimate(probe), 1) / probe.length;
  return Math.max(0, Math.floor(tokens / perChar));
}
