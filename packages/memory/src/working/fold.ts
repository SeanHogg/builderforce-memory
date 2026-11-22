/**
 * Advisory folds of the working transcript (PRD 45 §4.7 B6/B7) — pure post-processing
 * over the messages {@link buildWorkingTranscript} returns, before a host sends them to
 * the model. The folds are COMPRESSION ONLY: the run's own transcript (`state.transcript`)
 * and its trace keep every message at full fidelity; only what the MODEL sees is folded.
 *
 * B6 — fold superseded reads: when an edit/write to path P succeeds at step N, every
 * earlier `read_file` result for P in the model transcript becomes a one-line marker.
 * The stale content leaves the model's window (the edit changed those bytes — anything
 * the model still needs it re-reads), and stays in the trace at full fidelity.
 *
 * B7 — fold repeated narration: assistant text PRECEDING tool calls that restates the
 * previous such text — lead-sentence overlap ≥ {@link LEAD_SENTENCE_OVERLAP} OR whole-text
 * overlap ≥ {@link NARRATION_OVERLAP} — keeps only its genuinely NEW sentences; the
 * restated ones collapse to {@link RESTATEMENT_NOTE} (or {@link PROGRESS_NOTE} when the
 * whole narration restates). A final answer and a turn without tool calls are NEVER
 * folded — those carry the run's actual output.
 *
 * A "step" is the index of the assistant message carrying the tool call, in the message
 * list the fold runs over (the model transcript, system prompt included).
 *
 * Pure: no `node:*`, no I/O — runs wherever the `./working` subpath runs.
 */

import type { TranscriptMessage } from './transcript.js';
import {
  LEAD_SENTENCE_OVERLAP,
  NARRATION_OVERLAP,
  RESTATEMENT_NOTE,
  isRepeatedNarration,
  leadSentenceOverlap,
  newNarrationSentences,
  splitSentences,
} from './narrationOverlap.js';

// The ONE normalizer lives in narrationOverlap.ts (the host-side repeatedNarration
// detector imports the same functions). Re-exported here so fold.ts's existing
// consumers keep importing the measures from the fold module.
export {
  NARRATION_OVERLAP,
  LEAD_SENTENCE_OVERLAP,
  RESTATEMENT_NOTE,
  wordOverlap,
  wordContained,
  isRepeatedNarration,
  leadSentenceOverlap,
  splitSentences,
  newNarrationSentences,
} from './narrationOverlap.js';

/** B6's replacement for a read result whose file a later successful edit superseded. */
export function supersededReadMarker(path: string, step: number): string {
  return `(read of ${path} superseded by the edit at step ${step} — re-read if needed)`;
}

/** B7's replacement for a tool-call turn's narration that merely repeats the previous one. */
export const PROGRESS_NOTE = '(progress note)';

/** An OpenAI-style tool call, read defensively (a host's shape is not trusted). Shared with `runElision.ts`. */
export interface NamedToolCall {
  id?: unknown;
  function?: { name?: unknown; arguments?: unknown };
}

interface ToolResult {
  ok?: unknown;
  error?: unknown;
}

export function callsOf(m: TranscriptMessage): NamedToolCall[] {
  return Array.isArray(m.tool_calls) ? (m.tool_calls as NamedToolCall[]) : [];
}

export function nameOf(call: NamedToolCall): string | null {
  const name = call.function?.name;
  return typeof name === 'string' && name.length > 0 ? name : null;
}

/**
 * The first non-empty string argument of a tool call among `keys`, in that order — from the object form or parsed out of
 * a JSON string. Null when none is present or the arguments are not JSON (a malformed call names no target).
 */
export function stringArgOf(call: NamedToolCall, keys: readonly string[]): string | null {
  const raw = call.function?.arguments;
  let parsed: unknown = raw;
  if (typeof raw === 'string') {
    try {
      parsed = JSON.parse(raw);
    } catch {
      return null;
    }
  }
  if (!parsed || typeof parsed !== 'object') return null;
  for (const key of keys) {
    const value = (parsed as Record<string, unknown>)[key];
    if (typeof value === 'string' && value.length > 0) return value;
  }
  return null;
}

/** The `path` argument of a tool call. */
function pathOf(call: NamedToolCall): string | null {
  return stringArgOf(call, ['path']);
}

/** The tool result answering `call`: by `tool_call_id` when present, else positionally. */
function resultFor(convo: readonly TranscriptMessage[], call: NamedToolCall, callIndex: number): TranscriptMessage | undefined {
  const id = typeof call.id === 'string' && call.id.length > 0 ? call.id : null;
  if (id) {
    const byId = convo.find((m) => m.role === 'tool' && m.tool_call_id === id);
    if (byId) return byId;
  }
  // No ids (or an unmatched one): the nth tool result answers the nth call.
  const results = convo.filter((m) => m.role === 'tool');
  return results[callIndex];
}

/** Did the tool call SUCCEED? An explicit `{ok:false}` / `error` fails; anything else —
 *  including a textual error, which is still informative — passes. Absent result fails
 *  (a write nobody confirmed changed nothing). */
function succeeded(m: TranscriptMessage | undefined): boolean {
  if (!m) return false;
  const content: unknown = m.content;
  if (typeof content === 'string') return content.trim().length > 0;
  if (!content || typeof content !== 'object') return false;
  const r = content as ToolResult;
  if (typeof r.ok === 'boolean') return r.ok;
  return r.error == null;
}

const EDIT_OR_WRITE = /^(edit|write)/i;
const READ = /^read/i;

/** The narration an assistant turn carries before its tool calls — null when it carries
 *  none (an empty-content tool-call turn, or any turn without tool calls). */
function narrationOf(m: TranscriptMessage): string | null {
  const content = m.content;
  const text = typeof content === 'string' ? content : content == null ? '' : JSON.stringify(content);
  const t = text.trim();
  return callsOf(m).length > 0 && t.length > 0 ? t : null;
}

interface ReadRow {
  /** The assistant step that issued the read. */
  step: number;
  path: string;
  /** The message carrying the read's result — what the marker replaces. */
  result: TranscriptMessage;
  /** The read call itself — B7 skips a narration that narrated a now-folded read. */
  call: NamedToolCall;
}

/** Tracks which read CALLS were superseded, so `foldWorkingTranscript`'s B7 pass can
 *  skip the narration that introduced a read whose content B6 just replaced. */
const SUPERSEDED_CALL = new WeakSet<object>();

/**
 * B6 — fold reads a later successful edit/write superseded. Returns the input unchanged
 * (same reference, no copy) when nothing is superseded; otherwise a new list whose
 * replaced messages are shallow copies with only `content` changed. The caller's
 * transcript is never touched.
 */
export function foldSupersededReads<M extends TranscriptMessage>(msgs: readonly M[]): M[] {
  interface Edit {
    path: string;
    step: number;
  }
  const edits: Edit[] = [];
  const reads: ReadRow[] = [];
  let callIndex = 0;
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i]!;
    for (const call of callsOf(m)) {
      const name = nameOf(call);
      if (!name) continue;
      const path = pathOf(call);
      const result = resultFor(msgs, call, callIndex);
      if (path && EDIT_OR_WRITE.test(name)) {
        if (succeeded(result)) edits.push({ path, step: i });
      } else if (path && READ.test(name) && result) {
        reads.push({ step: i, path, result, call });
      }
      callIndex += 1;
    }
  }
  if (edits.length === 0 || reads.length === 0) return msgs as M[];
  const superseded = new Map<TranscriptMessage, string>();
  for (const read of reads) {
    // The LATEST successful edit of the same path after the read wins — an earlier read
    // re-made after an edit is not superseded by that edit.
    let step = -1;
    for (const e of edits) {
      if (e.path === read.path && e.step > read.step && e.step > step) step = e.step;
    }
    if (step >= 0) {
      superseded.set(read.result, supersededReadMarker(read.path, step));
      SUPERSEDED_CALL.add(read.call);
    }
  }
  if (superseded.size === 0) return msgs as M[];
  return msgs.map((m) => {
    const marker = superseded.get(m);
    return marker == null ? m : ({ ...m, content: marker } as M);
  });
}

/**
 * B7 — fold a tool-call turn's narration when it restates an earlier one.
 * Only assistant messages WITH tool calls are candidates: a final answer and a turn
 * without tool calls are never folded. The FIRST narration is kept verbatim (it has no
 * predecessor to repeat); each later one folds when its lead-sentence overlap with the
 * most recent kept narration is ≥ {@link LEAD_SENTENCE_OVERLAP} OR their whole-text
 * overlap is ≥ `overlap` (default {@link NARRATION_OVERLAP}).
 *
 * On fold, only the narration's genuinely NEW sentences survive — sentences whose own
 * overlap with EVERY earlier kept sentence is < {@link LEAD_SENTENCE_OVERLAP}; the
 * restated ones collapse to {@link RESTATEMENT_NOTE} ({@link PROGRESS_NOTE} when nothing
 * new remains). A fully folded note does not become the comparison text — the model
 * keeps the last thing actually said; surviving new sentences join the kept record.
 */
export function foldRepeatedNarration<M extends TranscriptMessage>(msgs: readonly M[], overlap = NARRATION_OVERLAP): M[] {
  let lastKept: string | null = null;
  const keptSentences: string[] = [];
  let changed = false;
  const out = msgs.map((m) => {
    if (m.role !== 'assistant') return m;
    const narration = narrationOf(m);
    if (narration == null) return m; // no tool calls, or no narration: never folded
    // A narration that introduced a read B6 folded is the model's only record of why it
    // read — it never folds itself, but it IS what a later repeat is compared against.
    if (callsOf(m).some((call) => SUPERSEDED_CALL.has(call))) {
      lastKept = narration;
      keptSentences.push(...splitSentences(narration));
      return m;
    }
    const restates =
      lastKept != null &&
      (leadSentenceOverlap(narration, lastKept) >= LEAD_SENTENCE_OVERLAP || isRepeatedNarration(narration, lastKept, overlap));
    if (restates) {
      changed = true;
      const fresh = newNarrationSentences(narration, keptSentences);
      if (fresh.length === 0) return { ...m, content: PROGRESS_NOTE } as M;
      keptSentences.push(...fresh);
      return { ...m, content: `${fresh.join(' ')} ${RESTATEMENT_NOTE}` } as M;
    }
    lastKept = narration;
    keptSentences.push(...splitSentences(narration));
    return m;
  });
  return changed ? out : (msgs as M[]);
}

/**
 * Both folds, B6 then B7, over the working transcript a host is about to send to the
 * model. Pure and advisory: `msgs` is not mutated, and the run's own transcript/trace
 * keep every message at full fidelity.
 */
export function foldWorkingTranscript<M extends TranscriptMessage>(msgs: readonly M[]): M[] {
  return foldRepeatedNarration(foldSupersededReads(msgs));
}
