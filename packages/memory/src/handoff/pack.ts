/**
 * `packHandoff` — fit a transcript into the NEXT model's window (PRD 34 §3.3).
 *
 * Shape of the result, mirroring Builderforce's `assembleCompacted`
 * (brain-embedded/src/workingTranscript.ts):
 *
 *   system message(s) · ONE assistant note for the dropped middle · the latest user
 *   directive re-injected verbatim (only when it fell in the middle) · the verbatim tail
 *
 * The tail is the longest run of whole units (see `segments.ts`) from the end that fits
 * what is left after the system message(s), the directive and room for an elision marker.
 * Whatever room the tail leaves is the note's: summary, else recall, else the marker.
 * The result never exceeds the budget unless the system message(s) + directive alone do.
 */

import { defaultEstimateTokens, handoffBudget, messageTokens, tokensOf } from './budget.js';
import { buildNote, composeNote, elisionMarker, handoffLine, noteTokens } from './note.js';
import { segmentTranscript, type Segment } from './segments.js';
import type { HandoffMessage, HandoffPack, HandoffPackInput, TokenEstimator } from './types.js';

interface Tail {
  /** Index in the non-system transcript where the verbatim tail begins. */
  start: number;
  tokens: number;
}

/** Index of the last `user` message, or -1. */
function latestDirectiveIndex(convo: readonly HandoffMessage[]): number {
  for (let i = convo.length - 1; i >= 0; i--) if (convo[i]!.role === 'user') return i;
  return -1;
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
  const all = input.messages;

  // Under budget: hand the transcript over untouched.
  if (tokensOf(all, estimate) <= budget) return result(all.slice(), 0, estimate);

  const system = all.filter((m) => m.role === 'system');
  const convo = all.filter((m) => m.role !== 'system');
  const systemTokens = tokensOf(system, estimate);
  const directiveIdx = latestDirectiveIndex(convo);
  const directive = directiveIdx >= 0 ? convo[directiveIdx] : undefined;
  const directiveTokens = directive ? messageTokens(directive, estimate) : 0;

  // The irreducible core does not fit: send it anyway and say so.
  if (systemTokens + directiveTokens > budget) {
    const core = directive ? [...system, directive] : [...system];
    return result(core, convo.length - (directive ? 1 : 0), estimate, { fits: false });
  }

  // Reserve room for the smallest note (the marker, at its widest count) when it fits.
  const header = handoffLine(input.from, input.to);
  const minNote = noteTokens(composeNote(header, elisionMarker(convo.length)), estimate);
  const noteFits = systemTokens + directiveTokens + minNote <= budget;
  const room = budget - systemTokens - (noteFits ? minNote : 0);

  const tail = selectTail(convo, segmentTranscript(convo, estimate), directiveIdx, directiveTokens, room);
  const reinject = directive !== undefined && directiveIdx < tail.start;
  const droppedMsgs = convo.slice(0, tail.start).filter((_, i) => !(reinject && i === directiveIdx));

  const used = systemTokens + tail.tokens + (reinject ? directiveTokens : 0);
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

  const messages: HandoffMessage[] = [...system];
  if (note) messages.push({ role: 'assistant', content: note.text });
  if (reinject && directive) messages.push(directive);
  messages.push(...convo.slice(tail.start));
  return result(messages, droppedMsgs.length, estimate, {
    summarized: note?.summarized ?? false,
    recalled: note?.recalled ?? false,
  });
}
