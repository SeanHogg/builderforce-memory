/**
 * The transcript summarizer — the ONE prompt every fold, hand-off note and chat
 * consolidation compresses a conversation with. The model call is the host's
 * ({@link SummaryCompletion}); this module owns what is asked and how the transcript is
 * rendered for it, so a compaction note, a hand-off note and a "Consolidate" summary read
 * the same everywhere.
 */

import { transcriptTokens, type TranscriptMessage } from './transcript.js';

/** The summarizer's instructions. */
export const TRANSCRIPT_SUMMARY_PROMPT =
  'You compress an in-progress AI agent transcript into a concise MEMORY the agent keeps working from. Capture: the CURRENT outstanding instruction from the user (the most recent user message is authoritative — earlier requests it supersedes are history, not the active task), concrete facts/answers discovered, tool results that matter (ids, paths, values), decisions made, and what still remains to do. Keep every file path, symbol name and line number the remaining work depends on, with the specific facts learned from each file — the agent no longer sees those results, so what you leave out it must read again. Be information-dense; drop pleasantries. No preamble.';

/**
 * The note's output ceiling. A compaction note is a UTILITY completion: a bounded answer
 * with no thinking. ~2.5k tokens: the note is the agent's only record of every file it
 * folds (paths, symbols, line numbers), and 1.2k was too little to carry them.
 */
export const TRANSCRIPT_SUMMARY_MAX_TOKENS = 2_500;

/**
 * The most transcript one summarizer call reads (~tokens). A longer input — a whole long
 * chat on "Consolidate", a first fold over a big seeded history — is summarized in
 * CHUNKS of this size, each folded onto the note so far, so no single call can overrun
 * a model's window.
 */
export const TRANSCRIPT_SUMMARY_INPUT_TOKENS = 48_000;

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
 * Split a transcript into consecutive chunks of at most `budget` tokens. A single message
 * larger than the budget is a chunk of its own (its text is clipped when rendered).
 */
export function summaryChunks(msgs: readonly TranscriptMessage[], budget = TRANSCRIPT_SUMMARY_INPUT_TOKENS): TranscriptMessage[][] {
  const chunks: TranscriptMessage[][] = [];
  let chunk: TranscriptMessage[] = [];
  let used = 0;
  for (const m of msgs) {
    const cost = transcriptTokens([m]);
    if (chunk.length > 0 && used + cost > budget) {
      chunks.push(chunk);
      chunk = [];
      used = 0;
    }
    chunk.push(m);
    used += cost;
  }
  if (chunk.length > 0) chunks.push(chunk);
  return chunks;
}

/** ~4 characters a token: the rendered text one call may carry. */
const clip = (text: string, budget: number) => {
  const max = budget * 4;
  return text.length > max ? `${text.slice(0, max)}\n[…clipped]` : text;
};

/**
 * Compress `msgs` into a memory note — in one call, or chunk by chunk with each chunk
 * folded onto the note so far when the transcript is longer than one call may read. Null
 * on any failure or an empty answer, so the caller falls back (drop-oldest, elision,
 * "nothing to consolidate").
 */
export async function summarizeTranscript(
  complete: SummaryCompletion,
  msgs: readonly TranscriptMessage[],
  signal?: AbortSignal,
  budget = TRANSCRIPT_SUMMARY_INPUT_TOKENS,
): Promise<string | null> {
  if (msgs.length === 0) return null;
  let note: string | null = null;
  for (const chunk of summaryChunks(msgs, budget)) {
    const input: TranscriptMessage[] = note ? [{ role: 'assistant', content: note } as TranscriptMessage, ...chunk] : chunk;
    try {
      const out: string = ((await complete({
        system: TRANSCRIPT_SUMMARY_PROMPT,
        user: clip(renderForSummary(input), budget),
        maxTokens: TRANSCRIPT_SUMMARY_MAX_TOKENS,
        ...(signal ? { signal } : {}),
      })) ?? '').trim();
      if (!out) return null;
      note = out;
    } catch {
      return null;
    }
  }
  return note;
}
