/**
 * The transcript summarizer — the ONE prompt every fold, hand-off note and chat
 * consolidation compresses a conversation with. The model call is the host's
 * ({@link SummaryCompletion}); this module owns what is asked and how the transcript is
 * rendered for it, so a compaction note, a hand-off note and a "Consolidate" summary read
 * the same everywhere.
 */

import type { TranscriptMessage } from './transcript.js';

/** The summarizer's instructions. */
export const TRANSCRIPT_SUMMARY_PROMPT =
  'You compress an in-progress AI agent transcript into a concise MEMORY the agent keeps working from. Capture: the CURRENT outstanding instruction from the user (the most recent user message is authoritative — earlier requests it supersedes are history, not the active task), concrete facts/answers discovered, tool results that matter (ids, paths, values), decisions made, and what still remains to do. Keep every file path, symbol name and line number the remaining work depends on, with the specific facts learned from each file — the agent no longer sees those results, so what you leave out it must read again. Be information-dense; drop pleasantries. No preamble.';

/**
 * The note's output ceiling. A compaction note is a UTILITY completion: a bounded answer
 * with no thinking. ~2.5k tokens: the note is the agent's only record of every file it
 * folds (paths, symbols, line numbers), and 1.2k was too little to carry them.
 */
export const TRANSCRIPT_SUMMARY_MAX_TOKENS = 2_500;

/** One no-tools, no-reasoning completion — the host's model call. Resolves the text. */
export type SummaryCompletion = (request: {
  system: string;
  user: string;
  maxTokens: number;
  signal?: AbortSignal;
}) => Promise<string | null | undefined>;

interface NamedToolCall {
  function?: { name?: unknown };
}

/** Render a slice of the transcript to the compact text the summarizer compresses. */
export function renderForSummary(msgs: readonly TranscriptMessage[]): string {
  return msgs
    .map((m) => {
      const content = typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? '');
      const calls = Array.isArray(m.tool_calls) && m.tool_calls.length > 0
        ? ` [called: ${(m.tool_calls as NamedToolCall[]).map((t) => t.function?.name).filter((n): n is string => typeof n === 'string' && n.length > 0).join(', ')}]`
        : '';
      return `${m.role}${calls}: ${content}`;
    })
    .join('\n\n');
}

/**
 * Compress `msgs` into a memory note. Null on any failure or an empty answer, so the
 * caller falls back (drop-oldest, elision, "nothing to consolidate").
 */
export async function summarizeTranscript(
  complete: SummaryCompletion,
  msgs: readonly TranscriptMessage[],
  signal?: AbortSignal,
): Promise<string | null> {
  if (msgs.length === 0) return null;
  try {
    const out = ((await complete({
      system: TRANSCRIPT_SUMMARY_PROMPT,
      user: renderForSummary(msgs),
      maxTokens: TRANSCRIPT_SUMMARY_MAX_TOKENS,
      ...(signal ? { signal } : {}),
    })) ?? '').trim();
    return out.length > 0 ? out : null;
  } catch {
    return null;
  }
}
