/**
 * Atomic units of a transcript — the pieces the packer keeps or drops whole.
 *
 * An assistant message carrying `tool_calls` and the `tool` results that follow it form
 * ONE unit: strict vendors reject a tool result whose call is missing, and a call whose
 * results are missing. Every other message is a unit of its own. A stray `tool` result
 * with no owning call joins the unit before it, so it can never lead a verbatim tail.
 */

import { hasToolCalls, messageTokens } from './budget.js';
import type { HandoffMessage, TokenEstimator } from './types.js';

/** A contiguous, never-split run of messages: `messages = convo.slice(start, end)`. */
export interface Segment {
  start: number;
  end: number;
  tokens: number;
}

/** Split `convo` into atomic units, in order. */
export function segmentTranscript(convo: readonly HandoffMessage[], estimate: TokenEstimator): Segment[] {
  const out: Segment[] = [];
  let i = 0;
  while (i < convo.length) {
    const m = convo[i]!;
    let end = i + 1;
    if (m.role === 'assistant' && hasToolCalls(m)) {
      while (end < convo.length && convo[end]!.role === 'tool') end += 1;
    }
    let tokens = 0;
    for (let k = i; k < end; k++) tokens += messageTokens(convo[k]!, estimate);
    const prev = out[out.length - 1];
    if (m.role === 'tool' && prev) {
      prev.end = end;
      prev.tokens += tokens;
    } else {
      out.push({ start: i, end, tokens });
    }
    i = end;
  }
  return out;
}
