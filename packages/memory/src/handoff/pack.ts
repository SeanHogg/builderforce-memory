/**
 * `packHandoff` — fit a transcript into the NEXT model's window (PRD 34 §3.3).
 *
 * Shape of the result, mirroring Builderforce's `assembleCompacted`
 * (brain-embedded/src/workingTranscript.ts):
 *
 *   leading system message(s) · pinned messages that fell in the middle (`pinned`) ·
 *   ONE note for the dropped middle · the latest user directive re-injected verbatim
 *   (only when it fell in the middle) · the verbatim tail
 *
 * Only the LEADING run of system messages is the fixed prefix. A system message later in
 * the conversation stays where it was: in the tail when the tail reaches it, else in the
 * dropped middle — never hoisted to the front, where it would read as if it governed the
 * whole conversation.
 *
 * The note's role keeps turns alternating: it is an `assistant` note when a user turn (the
 * re-injected directive, or a tail that starts with one) follows it, and a `user` note
 * when an assistant turn follows or nothing does — never two assistant turns in a row.
 *
 * The tail is the longest run of whole units (see `segments.ts`) from the end that fits
 * what is left after the prefix, the directive and room for an elision marker. Whatever
 * room the tail leaves is the note's: summary, else recall, else the marker. The result
 * never exceeds the budget unless the prefix + directive alone do.
 */

import { defaultEstimateTokens, handoffBudget, messageTokens, tokensOf } from './budget.js';
import { buildNote, composeNote, elisionMarker, handoffLine, noteTokens } from './note.js';
import { segmentTranscript, type Segment } from './segments.js';
import type { HandoffMessage, HandoffPack, HandoffPackInput, TokenEstimator } from './types.js';
import { isVolatileTail } from './volatile.js';

interface Tail {
  /** Index in the non-system transcript where the verbatim tail begins. */
  start: number;
  tokens: number;
}

/** Length of the leading run of `system` messages. */
function leadingSystemCount(all: readonly HandoffMessage[]): number {
  let n = 0;
  while (n < all.length && all[n]!.role === 'system') n += 1;
  return n;
}

/** The note's role: `user` before an assistant turn (or at the end), else `assistant`. */
function noteRole(next: HandoffMessage | undefined): 'user' | 'assistant' {
  return next === undefined || next.role === 'assistant' ? 'user' : 'assistant';
}

/** Index of the last `user` message that is a real directive (never a volatile tail), or -1. */
function latestDirectiveIndex(convo: readonly HandoffMessage[]): number {
  for (let i = convo.length - 1; i >= 0; i--) if (convo[i]!.role === 'user' && !isVolatileTail(convo[i]!)) return i;
  return -1;
}

/** Where the trailing run of volatile tails begins (`messages.length` when there is none). */
function trailingVolatileStart(messages: readonly HandoffMessage[]): number {
  let i = messages.length;
  while (i > 0 && isVolatileTail(messages[i - 1]!)) i -= 1;
  return i;
}

/** Convo indices the caller pinned (pinnable roles only), excluding the directive. */
function pinnedIndices(
  convo: readonly HandoffMessage[],
  prefixLen: number,
  directiveIdx: number,
  pinned: HandoffPackInput['pinned'],
): number[] {
  if (!pinned) return [];
  const out: number[] = [];
  for (let i = 0; i < convo.length; i++) {
    const m = convo[i]!;
    const pinnable = m.role === 'user' || (m.role === 'assistant' && !m.tool_calls);
    if (pinnable && i !== directiveIdx && pinned(m, prefixLen + i)) out.push(i);
  }
  return out;
}

/**
 * Longest suffix of whole segments that fits `room`, counting the directive's cost
 * separately while it still sits before the suffix. Never leads with a `tool` result.
 */
function selectTail(
  convo: readonly HandoffMessage[],
  segs: readonly Segment[],
  directiveIdx: number,
  directiveTokens: number,
  room: number,
): Tail {
  let start = convo.length;
  let tokens = 0;
  let first = segs.length;
  for (let s = segs.length - 1; s >= 0; s--) {
    const seg = segs[s]!;
    const directiveOutside = directiveIdx >= 0 && directiveIdx < seg.start;
    if (tokens + seg.tokens + (directiveOutside ? directiveTokens : 0) > room) break;
    tokens += seg.tokens;
    start = seg.start;
    first = s;
  }
  // Only an orphan result at the very front can lead a unit with `tool`; drop it.
  while (first < segs.length && convo[segs[first]!.start]!.role === 'tool') {
    tokens -= segs[first]!.tokens;
    first += 1;
    start = first < segs.length ? segs[first]!.start : convo.length;
  }
  return { start, tokens };
}

function result(
  messages: HandoffMessage[],
  dropped: number,
  estimate: TokenEstimator,
  extra: Partial<Pick<HandoffPack, 'summarized' | 'recalled' | 'fits'>> = {},
): HandoffPack {
  return {
    messages,
    dropped,
    summarized: extra.summarized ?? false,
    recalled: extra.recalled ?? false,
    estimatedTokens: tokensOf(messages, estimate),
    fits: extra.fits ?? true,
  };
}

export async function packHandoff(input: HandoffPackInput): Promise<HandoffPack> {
  const estimate = input.estimateTokens ?? defaultEstimateTokens;
  const budget = handoffBudget(input);
  // Under budget: hand the transcript over untouched.
  if (tokensOf(input.messages, estimate) <= budget) return result(input.messages.slice(), 0, estimate);

  // The volatile tail (`volatile.ts`) is set aside, the history packed into what is
  // left, and the tail put back LAST — never the directive, never in the middle. When
  // the history needs the whole budget, the tail is what goes: it is a per-turn aid,
  // the history is the work.
  const cut = trailingVolatileStart(input.messages);
  const volatile = input.messages.slice(cut);
  const volatileTokens = tokensOf(volatile, estimate);
  if (volatile.length > 0) {
    const history = input.messages.slice(0, cut);
    const withTail = await packHistory({ ...input, messages: history }, budget - volatileTokens, estimate);
    if (withTail.fits) return result([...withTail.messages, ...volatile], withTail.dropped, estimate, withTail);
    return packHistory({ ...input, messages: history }, budget, estimate);
  }
  return packHistory(input, budget, estimate);
}

async function packHistory(input: HandoffPackInput, budget: number, estimate: TokenEstimator): Promise<HandoffPack> {
  const all = input.messages;
  if (tokensOf(all, estimate) <= budget) return result(all.slice(), 0, estimate);

  const prefixLen = leadingSystemCount(all);
  const system = all.slice(0, prefixLen);
  const convo = all.slice(prefixLen);
  const systemTokens = tokensOf(system, estimate);
  const directiveIdx = latestDirectiveIndex(convo);
  const directive = directiveIdx >= 0 ? convo[directiveIdx] : undefined;
  const directiveTokens = directive ? messageTokens(directive, estimate) : 0;
  // Pinned messages are charged up front (conservatively, even if the tail reaches them).
  const pins = pinnedIndices(convo, prefixLen, directiveIdx, input.pinned);
  const pinTokens = pins.reduce((sum, i) => sum + messageTokens(convo[i]!, estimate), 0);
  const coreTokens = directiveTokens + pinTokens;

  // The irreducible core does not fit: send it anyway and say so.
  if (systemTokens + coreTokens > budget) {
    const core = [...system, ...pins.map((i) => convo[i]!), ...(directive ? [directive] : [])];
    return result(core, convo.length - pins.length - (directive ? 1 : 0), estimate, { fits: false });
  }

  // Reserve room for the smallest note (the marker, at its widest count) when it fits.
  const header = handoffLine(input.from, input.to);
  const minNote = noteTokens(composeNote(header, elisionMarker(convo.length)), estimate);
  const noteFits = systemTokens + coreTokens + minNote <= budget;
  const room = budget - systemTokens - pinTokens - (noteFits ? minNote : 0);

  const tail = selectTail(convo, segmentTranscript(convo, estimate), directiveIdx, directiveTokens, room);
  const reinject = directive !== undefined && directiveIdx < tail.start;
  const pinnedBefore = pins.filter((i) => i < tail.start);
  const keptBefore = new Set(pinnedBefore);
  const droppedMsgs = convo.slice(0, tail.start).filter((_, i) => !(reinject && i === directiveIdx) && !keptBefore.has(i));

  const used = systemTokens + tail.tokens + pinTokens + (reinject ? directiveTokens : 0);
  const note = droppedMsgs.length > 0
    ? await buildNote({
        header,
        dropped: droppedMsgs,
        directive,
        room: budget - used,
        estimate,
        summarize: input.summarize,
        recall: input.recall,
        timeoutMs: input.timeoutMs,
      })
    : null;

  const after: HandoffMessage[] = reinject && directive ? [directive] : [];
  after.push(...convo.slice(tail.start));
  const messages: HandoffMessage[] = [...system, ...pinnedBefore.map((i) => convo[i]!)];
  if (note) messages.push({ role: noteRole(after[0]), content: note.text });
  messages.push(...after);
  return result(messages, droppedMsgs.length, estimate, {
    summarized: note?.summarized ?? false,
    recalled: note?.recalled ?? false,
  });
}
