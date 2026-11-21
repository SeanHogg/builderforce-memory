/**
 * The working transcript — what of a run the model is SHOWN each turn.
 *
 * A run's full transcript grows without bound; the request sent to the model must not.
 * This module owns the two ways it is kept in bounds: the message-count + token WINDOW
 * (drop-oldest, anchored on a user turn) and AUTO-COMPACTION (fold the older part into a
 * memory note, keep every later message verbatim). A host supplies only what is
 * run-specific: the summarizer it calls (see `summary.ts`) and what a fold records.
 *
 * Pure except for {@link buildWorkingTranscript}, which awaits the injected summarizer and
 * updates the caller's `compactMemo`. Generic over the host's message shape: anything with
 * the OpenAI-style `role` / `content` / `tool_calls` fields (`HandoffMessage`).
 */

import { defaultEstimateTokens, messageTokens as handoffMessageTokens, tokensOf as handoffTokensOf } from '../handoff/budget.js';
import type { HandoffMessage } from '../handoff/types.js';
import { CONSOLIDATION_MARKER_PREFIX } from './consolidationMarker.js';
import { elideRunResults, type RunElisionBudget } from './runElision.js';

/** A transcript message — the hand-off packer's shape, so both count a message alike. */
export type TranscriptMessage = HandoffMessage;

/** How much history we send to the model (message-count ceiling). */
export const HISTORY_WINDOW = 80;

/**
 * Token budget for the working transcript sent to the model each turn — how much of this
 * run the model can SEE: the file windows it read, the searches it ran, what it decided.
 *
 * It bounds context, which message-count windowing (HISTORY_WINDOW) alone does not: one
 * `tasks.list` result can be tens of thousands of tokens. It is NOT what keeps a request
 * inside the serving model's window — the gateway fits every request to a model that can
 * hold it.
 *
 * It was 24k, "sized under the smallest pool model's window". That protected nothing — the
 * ~16k-token system prompt + tool catalog in front of it already put a full turn at ~40k,
 * past any 32k window — and it cost every coding run its memory. Six 4k-token file windows
 * filled it, compaction folded them into a 1.2k-token note, and the model went back for the
 * files it had just read (chat #105: 55 turns, 63% of calls revisiting, zero edits, prompt
 * peak 40,351). 64k holds a coding task's working set — a dozen-plus file windows —
 * verbatim, while a full turn stays well inside the 128k window every coding-capable pool
 * model has. See {@link windowed} and {@link buildWorkingTranscript}.
 */
export const HISTORY_TOKEN_BUDGET = 64_000;
/**
 * How much of {@link HISTORY_TOKEN_BUDGET} the verbatim TAIL may keep when the older part
 * of the transcript is compacted. The rest is headroom new work fills before the next
 * fold, so the memo is re-folded once per ~20k tokens of progress, not on every turn.
 */
export const COMPACT_TAIL_TOKEN_BUDGET = 40_000;

/** Estimated tokens for one message (content + tool-call payload + framing) — THE
 *  hand-off packer's estimate, so the working transcript and a hand-off to a smaller
 *  model count a message the same way. */
function messageTokens(m: TranscriptMessage): number {
  return handoffMessageTokens(m, defaultEstimateTokens);
}

/**
 * The compressed memory of a transcript's older part: `note` summarizes
 * `transcript[0, coveredEnd)`; every message from `coveredEnd` on is sent verbatim.
 */
export interface CompactMemo {
  note: string;
  coveredEnd: number;
}

/**
 * What the working transcript is built from — the run's full transcript and its cached
 * compressed memory. {@link buildWorkingTranscript} updates `compactMemo` in place.
 */
export interface WorkingTranscriptState<M extends TranscriptMessage = TranscriptMessage> {
  transcript: M[];
  compactMemo: CompactMemo | null;
}

/**
 * How far the message window's front moves at a time — a quarter of
 * {@link HISTORY_WINDOW}. See {@link windowed}.
 */
export const HISTORY_WINDOW_DROP_CHUNK = HISTORY_WINDOW / 4;

/**
 * How far (in tokens) the token window's front moves at a time — a quarter of
 * {@link HISTORY_TOKEN_BUDGET}, so a drop leaves the window at ~75% of the budget and
 * the next one waits for ~16k tokens of new work. See {@link windowed}.
 */
export const HISTORY_TOKEN_DROP_CHUNK = HISTORY_TOKEN_BUDGET / 4;

/**
 * Where the message-count window begins: 0 while the transcript fits
 * {@link HISTORY_WINDOW}, else the first multiple of {@link HISTORY_WINDOW_DROP_CHUNK}
 * that brings it back under. The cut points are fixed positions in the transcript, so
 * the front stays put while new messages arrive and jumps a whole chunk when it moves.
 */
function messageWindowStart(length: number): number {
  if (length <= HISTORY_WINDOW) return 0;
  return Math.ceil((length - HISTORY_WINDOW) / HISTORY_WINDOW_DROP_CHUNK) * HISTORY_WINDOW_DROP_CHUNK;
}

/**
 * Where the token window begins: 0 while the transcript fits
 * {@link HISTORY_TOKEN_BUDGET}, else the first cut point that brings it back under. A
 * cut point is the message after which the transcript's running total crosses a multiple
 * of {@link HISTORY_TOKEN_DROP_CHUNK} — counted from the transcript's own start, so a
 * cut point never moves as the transcript grows at the end.
 */
function tokenWindowStart(convo: readonly TranscriptMessage[]): number {
  const sizes = convo.map(messageTokens);
  let remaining = sizes.reduce((sum, n) => sum + n, 0);
  if (remaining <= HISTORY_TOKEN_BUDGET) return 0;
  let dropped = 0;
  let nextCut = HISTORY_TOKEN_DROP_CHUNK;
  for (let i = 0; i < convo.length; i++) {
    dropped += sizes[i]!;
    remaining -= sizes[i]!;
    if (dropped < nextCut) continue;
    while (dropped >= nextCut) nextCut += HISTORY_TOKEN_DROP_CHUNK;
    if (remaining <= HISTORY_TOKEN_BUDGET) return i + 1;
  }
  return convo.length;
}

/**
 * Trim the in-memory transcript to the history window — the message-count window
 * ({@link HISTORY_WINDOW}) and the token budget ({@link HISTORY_TOKEN_BUDGET}) — before
 * sending it to the model.
 *
 * The OLDEST turns go first, in CHUNKS with hysteresis: once over, the front jumps to
 * the next fixed cut point ({@link HISTORY_WINDOW_DROP_CHUNK} messages /
 * {@link HISTORY_TOKEN_DROP_CHUNK} tokens), leaving room the following turns fill before
 * it moves again. Dropping exactly to the limit slid the front by a message or two on
 * every turn, so the request's prefix changed every turn and no provider prefix cache
 * could hit past the system prompt.
 *
 * The window is then anchored at a user turn: a window starting on an `assistant`
 * tool-call turn or an orphaned `tool` result (whose owning call fell off the front) is
 * invalid for strict vendors — Gemini rejects a request whose conversation does not
 * begin with a user turn (the cascade `[googleai] 400 INVALID_ARGUMENT` after a long
 * tool loop crossed the window boundary). The most recent user turn is never dropped:
 * correctness over the size cap, when a single turn is itself oversized.
 *
 * A RUN that outgrows the budget after that user turn is cut inside ({@link elideRunResults}):
 * its older tool results become one-line stubs, keeping the user turn, every assistant turn
 * and a ~{@link COMPACT_TAIL_TOKEN_BUDGET}-token verbatim tail. Without it one long run grew
 * the prompt by every result it produced (chat #128305: 31k → 191k tokens).
 *
 * A consolidation marker that LEADS the transcript (a reloaded chat's seed begins with its saved
 * fold note) is never dropped: it is the note that stands for everything before the rest, and
 * walking forward to the first user turn would hand the model the verbatim tail with no
 * memory of what came before it. It sits in front of the window the way a compaction note
 * does (`assembleCompacted`).
 */
export function windowed<M extends TranscriptMessage>(convo: M[]): M[] {
  const lastUser = convo.map((m) => m.role).lastIndexOf('user');
  if (lastUser < 0) return transcriptTokens(convo) <= HISTORY_TOKEN_BUDGET ? convo.slice() : convo.slice(-1);
  let start = Math.min(Math.max(messageWindowStart(convo.length), tokenWindowStart(convo)), lastUser);
  while (convo[start]!.role !== 'user') start += 1;
  const lead = convo[0]!;
  const window = start > 0 && isLeadingMarker(lead) ? [lead, ...convo.slice(start)] : convo.slice(start);
  return elideRunResults(window, runElisionBudget());
}

/** The in-run cut's budget: the history budget, the compacted tail, the token drop chunk. */
function runElisionBudget(extraTokens = 0): RunElisionBudget {
  return {
    budgetTokens: HISTORY_TOKEN_BUDGET,
    tailTokens: COMPACT_TAIL_TOKEN_BUDGET,
    chunkTokens: HISTORY_TOKEN_DROP_CHUNK,
    minTailMessages: COMPACT_TAIL_TURNS,
    extraTokens,
  };
}

/** Is this row a saved consolidation note (the visible marker the next seed starts from)? */
function isLeadingMarker(m: TranscriptMessage): boolean {
  return m.role === 'assistant' && typeof m.content === 'string' && m.content.startsWith(CONSOLIDATION_MARKER_PREFIX);
}

/**
 * Is a transcript message still part of what the model sees? False once auto-compaction
 * has folded it into the memory note, or the drop-oldest window has let it fall off the
 * front. Deliberately conservative: it judges against the CURRENT transcript (which has
 * grown since the last working set was built), so it can only ever call a message gone a
 * turn early — a needless replay costs tokens, a stub for a vanished result costs the run.
 * A result the in-run cut elided ({@link elideRunResults}) is gone too: the model sees its
 * stub, not the row.
 */
export function stillInWorkingContext<M extends TranscriptMessage>(state: WorkingTranscriptState<M>, anchor: unknown): boolean {
  const convo = state.transcript;
  const idx = convo.indexOf(anchor as M);
  if (idx < 0) return false;
  const memo = state.compactMemo;
  if (memo) {
    if (idx < verbatimStart(convo, memo.coveredEnd)) return false;
    return boundedCompacted('', convo, memo.note, memo.coveredEnd).includes(convo[idx]!);
  }
  return windowed(convo).includes(convo[idx]!);
}

// ---------------------------------------------------------------------------
// Auto-compaction — summarize the bulky MIDDLE instead of dropping it.
//
// `windowed` keeps the request inside the window by DROPPING the oldest turns. That
// never 413s, but it silently LOSES context — which made a weak model re-read files and
// thrash ("LOOP EXHAUSTED", chat #50). When a summarizer is available the older turns are
// compressed into ONE memory note instead, so the model keeps working from a distilled
// memory. Falls back to `windowed` when no summarizer is reachable.
//
// Why this fold is NOT `packHandoff` (`../handoff`): the packer is stateless and gives the
// verbatim tail priority, handing the note whatever room the tail leaves. The working
// transcript needs the opposite on both counts — an INCREMENTAL memo reused across turns
// (one summarizer call per ~20k tokens of progress), a tail capped at
// COMPACT_TAIL_TOKEN_BUDGET so the note and the next turns have headroom, and a
// message-count window. What the two share — the token estimate — is the packer's.
// ---------------------------------------------------------------------------

/** The fewest recent messages the verbatim tail keeps, however large they are. */
export const COMPACT_TAIL_TURNS = 8;

/** The most recent messages the verbatim tail may hold — half the message window, so a
 *  compacted transcript stays inside {@link HISTORY_WINDOW} with room to grow. */
const COMPACT_TAIL_MAX_MESSAGES = HISTORY_WINDOW / 2;

/** The first index at or after `from` that is not a `tool` result: a tool row whose
 *  assistant call was folded into the memo would be orphaned (strict vendors 400 on it). */
export function verbatimStart(convo: readonly TranscriptMessage[], from: number): number {
  let start = Math.max(0, Math.min(from, convo.length));
  while (start < convo.length && convo[start]!.role === 'tool') start += 1;
  return start;
}

/** Start index of the last `tailTurns` messages, walked off a leading `tool` result. */
export function compactTailStart(convo: readonly TranscriptMessage[], tailTurns: number): number {
  return verbatimStart(convo, convo.length - tailTurns);
}

/**
 * Where the verbatim tail begins when the transcript is compacted: as many of the most
 * recent messages as fit in `budgetTokens` — never fewer than {@link COMPACT_TAIL_TURNS},
 * never more than {@link COMPACT_TAIL_MAX_MESSAGES} — walked off a leading tool result.
 *
 * Sized by TOKENS, not a message count: a fixed eight messages is four tool results, so
 * after a fold the fifth file the model needed was always the one it had to read again.
 */
export function compactTailStartForBudget(convo: readonly TranscriptMessage[], budgetTokens: number): number {
  let start = convo.length;
  let tokens = 0;
  while (start > 0) {
    const kept = convo.length - start;
    if (kept >= COMPACT_TAIL_MAX_MESSAGES) break;
    const next = messageTokens(convo[start - 1]!);
    if (kept >= COMPACT_TAIL_TURNS && tokens + next > budgetTokens) break;
    tokens += next;
    start -= 1;
  }
  return verbatimStart(convo, start);
}

/**
 * Index of the MOST RECENT user turn to re-inject verbatim ahead of the tail — the
 * ACTIVE directive — or -1 when the latest user turn is already inside the tail. Pinning
 * the FIRST user turn (the old behaviour) made a multi-instruction chat revert to its
 * opening request (chat #55); the current instruction is always the LATEST user turn.
 */
export function pinnedDirectiveIndex(convo: readonly TranscriptMessage[], tailStart: number): number {
  const lastUser = convo.map((m) => m.role).lastIndexOf('user');
  return lastUser >= 0 && lastUser < tailStart ? lastUser : -1;
}

/**
 * Assemble the working transcript from a compressed-memory `note` that covers
 * `convo[0, coveredEnd)`: system + the memory note + the ACTIVE user directive re-injected
 * verbatim (when the note covers it) + EVERY message from `coveredEnd` on (tool-pairing
 * safe via {@link verbatimStart}). Nothing is in neither the note nor the tail. The
 * directive sits immediately before the tail so the model reads it as the CURRENT
 * instruction.
 */
export function assembleCompacted<M extends TranscriptMessage>(
  systemPrompt: string,
  convo: M[],
  note: string,
  coveredEnd: number,
): M[] {
  const tailStart = verbatimStart(convo, coveredEnd);
  const out: M[] = [{ role: 'system', content: systemPrompt } as M];
  out.push({ role: 'assistant', content: note } as M);
  const directiveIdx = pinnedDirectiveIndex(convo, tailStart);
  if (directiveIdx >= 0) out.push(convo[directiveIdx]!);
  out.push(...convo.slice(tailStart));
  return out;
}

/**
 * {@link assembleCompacted}, kept inside the history budget: when the note's verbatim
 * remainder has outgrown it and no new fold was made (the summary failed or timed out),
 * the current run's older tool results are elided ({@link elideRunResults}) — the note and
 * the system prompt are kept, the note charged to the budget as the window charges it.
 */
function boundedCompacted<M extends TranscriptMessage>(systemPrompt: string, convo: M[], note: string, coveredEnd: number): M[] {
  const [system, noteRow, ...rest] = assembleCompacted(systemPrompt, convo, note, coveredEnd);
  return [system!, noteRow!, ...elideRunResults(rest, runElisionBudget(transcriptTokens([noteRow!])))];
}

/** Estimated tokens of a message list — the ONE count compaction budgets by, exported so
 *  a run measures what compaction kept out of a call with the same rule. */
export function transcriptTokens(msgs: readonly TranscriptMessage[]): number {
  return handoffTokensOf(msgs, defaultEstimateTokens);
}

/** Can these messages (plus `extraTokens` in front of them) go to the model verbatim —
 *  inside both the token budget and the message window? */
function fitsVerbatim(msgs: readonly TranscriptMessage[], extraTokens = 0): boolean {
  return msgs.length <= HISTORY_WINDOW && extraTokens + transcriptTokens(msgs) <= HISTORY_TOKEN_BUDGET;
}

/** What a fold reports: how many messages it newly compressed, the note it wrote, and
 *  the transcript index the note now covers up to (exclusive). */
export interface TranscriptFold {
  droppedMessages: number;
  note: string;
  coveredEnd: number;
}

/**
 * Build the working transcript for a turn. When the whole transcript fits → all of it,
 * verbatim. Otherwise → a memory note of the older part plus EVERY message after it,
 * verbatim; `onFolded` is told about each NEW fold, so the caller can record it (a
 * visible step, a durable consolidation marker).
 *
 * The note is folded INCREMENTALLY, and only when it has to be. A note whose verbatim
 * remainder still fits is reused as-is; once it no longer fits, the previous note plus
 * exactly the messages since it are folded into a new one, up to a token-sized tail
 * ({@link compactTailStartForBudget}). Falls back to drop-oldest when the summarizer is
 * unavailable (`summarize` resolves null). A message-count overflow folds too.
 */
export async function buildWorkingTranscript<M extends TranscriptMessage>(
  state: WorkingTranscriptState<M>,
  systemPrompt: string,
  summarize: (msgs: M[]) => Promise<string | null>,
  onFolded: (fold: TranscriptFold) => void,
): Promise<M[]> {
  const convo = state.transcript;
  if (fitsVerbatim(convo)) {
    state.compactMemo = null; // back under budget — a later overflow summarizes afresh
    return [{ role: 'system', content: systemPrompt } as M, ...windowed(convo)];
  }
  // A note written against a transcript that has since been replaced covers nothing here.
  const memo = state.compactMemo && state.compactMemo.coveredEnd <= convo.length ? state.compactMemo : null;
  if (memo && fitsVerbatim(convo.slice(verbatimStart(convo, memo.coveredEnd)), defaultEstimateTokens(memo.note))) {
    return assembleCompacted(systemPrompt, convo, memo.note, memo.coveredEnd);
  }
  const from = memo?.coveredEnd ?? 0;
  const to = Math.max(from, compactTailStartForBudget(convo, COMPACT_TAIL_TOKEN_BUDGET));
  const fold: M[] = memo
    ? [{ role: 'assistant', content: memo.note } as M, ...convo.slice(from, to)]
    : convo.slice(from, to);
  const summary = to > from ? await summarize(fold) : null;
  if (summary == null) {
    // Nothing new to fold, or no summarizer: keep the note already held, else drop-oldest —
    // either way cut inside the run once it outgrows the budget (`elideRunResults`).
    return memo
      ? boundedCompacted(systemPrompt, convo, memo.note, memo.coveredEnd)
      : [{ role: 'system', content: systemPrompt } as M, ...windowed(convo)];
  }
  const note = `Compressed memory of the earlier part of this conversation:\n${summary}`;
  state.compactMemo = { note, coveredEnd: to };
  onFolded({ droppedMessages: to - from, note, coveredEnd: to });
  return assembleCompacted(systemPrompt, convo, note, to);
}
