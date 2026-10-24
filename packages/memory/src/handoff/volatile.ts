/**
 * The per-request VOLATILE tail — everything that changes turn to turn (the run's
 * working-memory brief, the affect block, the budget line, task state) in ONE `user`
 * message appended AFTER the history and opened by {@link VOLATILE_TAIL_TAG}.
 *
 * Two readers depend on that one convention, which is why it lives here and every host
 * builds its tail with {@link volatileTail}:
 *
 *   - The gateway's prompt caching places the history breakpoint BEFORE a tagged block,
 *     so a tail that changes every turn never becomes the cached prefix. An untagged
 *     trailing block is where the breakpoint lands — and then every turn re-bills the
 *     whole transcript.
 *   - {@link packHandoff} never treats a tagged block as "the latest user directive"
 *     and keeps it LAST: an untagged one would be protected in place of the user's real
 *     request, which could then be dropped into the summarised middle.
 *
 * A trailing user turn is valid after a tool-result turn on both the OpenAI and the
 * Anthropic shape (Anthropic merges it into the tool_result user turn). It is built per
 * request and never persisted into a transcript.
 */

import type { HandoffMessage } from './types.js';

export const VOLATILE_TAIL_TAG = '<run-state>';
const VOLATILE_TAIL_CLOSE = '</run-state>';

/** True when `content` (a string or a content-part array) opens with the tag. */
export function isVolatileContent(content: unknown): boolean {
  if (typeof content === 'string') return content.startsWith(VOLATILE_TAIL_TAG);
  if (Array.isArray(content)) {
    const first = content[0] as { text?: unknown } | undefined;
    return typeof first?.text === 'string' && first.text.startsWith(VOLATILE_TAIL_TAG);
  }
  return false;
}

/** True for a `user` message that is a volatile tail. */
export function isVolatileTail(message: Pick<HandoffMessage, 'role' | 'content'>): boolean {
  return message.role === 'user' && isVolatileContent(message.content);
}

/** The ONE tail message for these sections, or null when every section is empty. */
export function volatileTail(sections: readonly (string | null | undefined)[]): { role: 'user'; content: string } | null {
  const body = sections.map((s) => (s ?? '').trim()).filter(Boolean).join('\n\n');
  return body ? { role: 'user', content: `${VOLATILE_TAIL_TAG}\n${body}\n${VOLATILE_TAIL_CLOSE}` } : null;
}
