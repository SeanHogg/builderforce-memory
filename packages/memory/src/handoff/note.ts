/**
 * The ONE note that stands in for the dropped middle of a handed-off
 * transcript: an optional one-line hand-off note, then the first body source that
 * succeeds — the `summarize` result, else the `recall` memory block for the latest
 * directive, else an elision marker. A failing, timing-out or empty source falls through
 * to the next; nothing here throws for those.
 */

import { MESSAGE_OVERHEAD_TOKENS, charsForTokens, contentText, fitText } from './budget.js';
import type { HandoffMessage, HandoffPackInput, TokenEstimator } from './types.js';

/** The hand-off line, or '' when either ref is missing or they are the same model. */
export function handoffLine(from?: string, to?: string): string {
  return from && to && from !== to ? `Handed off from ${from} to ${to}.` : '';
}

/** The marker for `n` messages that were dropped with nothing to stand in for them. */
export function elisionMarker(n: number): string {
  return `[${n} earlier messages omitted]`;
}

/** Header + body as one note text. */
export function composeNote(header: string, body: string): string {
  return header ? `${header}\n${body}` : body;
}

/** Tokens the note message costs, framing included. */
export function noteTokens(text: string, estimate: TokenEstimator): number {
  return estimate(text) + MESSAGE_OVERHEAD_TOKENS;
}

export interface NoteRequest {
  header: string;
  dropped: readonly HandoffMessage[];
  directive: HandoffMessage | undefined;
  /** Tokens the note message may cost, framing included. */
  room: number;
  estimate: TokenEstimator;
  summarize?: HandoffPackInput['summarize'];
  recall?: HandoffPackInput['recall'];
  timeoutMs?: number;
}

export interface NoteResult {
  text: string;
  summarized: boolean;
  recalled: boolean;
}

/** Settle `run()` to its trimmed string, or '' on throw, rejection, timeout or non-string. */
async function attempt(run: () => Promise<string>, timeoutMs?: number): Promise<string> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const work = Promise.resolve().then(run);
    const raced = timeoutMs && timeoutMs > 0
      ? Promise.race([work, new Promise<string>((resolve) => { timer = setTimeout(() => resolve(''), timeoutMs); })])
      : work;
    const out = await raced;
    return typeof out === 'string' ? out.trim() : '';
  } catch {
    return '';
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

/** `body` cut so header + body fits `room`; '' when nothing of it fits. */
function fitBody(req: NoteRequest, body: string): string {
  const headerTokens = req.header ? req.estimate(`${req.header}\n`) : 0;
  let max = req.room - MESSAGE_OVERHEAD_TOKENS - headerTokens;
  // The estimator need not be additive; shrink until the composed note really fits.
  for (let tries = 0; tries < 16 && max > 0; tries++, max--) {
    const fitted = fitText(body, max, req.estimate);
    if (!fitted) return '';
    if (noteTokens(composeNote(req.header, fitted), req.estimate) <= req.room) return fitted;
  }
  return '';
}

/** Build the note, or null when not even the elision marker fits `room`. */
export async function buildNote(req: NoteRequest): Promise<NoteResult | null> {
  if (req.summarize) {
    const summarize = req.summarize;
    const summary = fitBody(req, await attempt(() => summarize(req.dropped), req.timeoutMs));
    if (summary) return { text: composeNote(req.header, summary), summarized: true, recalled: false };
  }

  if (req.recall && req.directive) {
    const recall = req.recall;
    const headerTokens = req.header ? req.estimate(`${req.header}\n`) : 0;
    const budgetChars = charsForTokens(req.room - MESSAGE_OVERHEAD_TOKENS - headerTokens, req.estimate);
    if (budgetChars > 0) {
      const query = contentText(req.directive.content);
      const block = fitBody(req, await attempt(() => recall(query, budgetChars), req.timeoutMs));
      if (block) return { text: composeNote(req.header, block), summarized: false, recalled: true };
    }
  }

  const marker = composeNote(req.header, elisionMarker(req.dropped.length));
  if (noteTokens(marker, req.estimate) <= req.room) return { text: marker, summarized: false, recalled: false };
  return null;
}
