/**
 * Chunked text summaries — the TEXT-level primitive beside the message-level {@link summaryChunks} /
 * {@link summarizeTranscript} (summary.ts): a small window (an on-device model) takes a long text.
 *
 *   1. the text fits the chunk limit → ONE `summarizeChunk(text)` call;
 *   2. otherwise split on paragraph, then line boundaries (then a hard cut) into chunks under the limit,
 *      re-measured with the caller's own `measure` and split again when the char estimate under-cut;
 *   3. summarize each chunk; any chunk declined (`null`) → `null`;
 *   4. summarize the joined summaries, recursing while still over the limit, up to `maxMergeDepth`;
 *   5. past `maxChunks` leaf chunks, past the depth, or a merge that did not shrink the text → `null`
 *      (the caller's existing path takes it).
 *
 * Pure over an injected `measure` and an injected one-chunk summarizer: it knows no model API. An aborted
 * `signal` rejects with an `AbortError` (and only an abort throws). Ported from brain-embedded's
 * `onDevice/chunkedSummary.ts`, behaviour for behaviour; the one change is that a hard cut never splits a
 * UTF-16 surrogate pair.
 */

import { chunkText as chunkTextShared } from '../retrieval/chunk.js';

/** Separators from the coarsest boundary to the finest; the empty one is the hard cut. */
const SEPARATORS = ['\n\n', '\n', ''];
/** Bounds the re-measure/re-split of a chunk the first estimate under-cut. */
const MAX_REFIT_ATTEMPTS = 3;
/** Merging summaries must shrink the text; this bounds a pathological recursion. */
export const DEFAULT_MAX_MERGE_DEPTH = 3;

export interface ChunkedTextSummaryOptions {
  text: string;
  /** The window, in the units `measure` counts (tokens). */
  limit: number;
  /** The size of `text` in the window's units. */
  measure(text: string, signal?: AbortSignal): number | Promise<number>;
  /** Summarizes one in-window piece; `null` = declined (the whole summary is then `null`). */
  summarizeChunk(chunk: string, signal?: AbortSignal): Promise<string | null>;
  /** More leaf chunks than this → `null` without summarizing any. */
  maxChunks: number;
  /** Merge rounds before giving up (default {@link DEFAULT_MAX_MERGE_DEPTH}). */
  maxMergeDepth?: number;
  /** Units each call spends outside the chunk (instruction + expected reply); never more than half the window. Default 0. */
  reserveTokens?: number;
  /** Share of the remaining window a chunk may fill (default 1; an on-device host uses 0.8 for headroom). */
  budgetShare?: number;
  signal?: AbortSignal;
}

function abortError(): Error {
  const e = new Error('Aborted');
  e.name = 'AbortError';
  return e;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw abortError();
}

/**
 * `text` as chunks of at most `maxChars` (a lone surrogate pair at `maxChars` 1 is the one exception): the shared
 * retrieval splitter in its exact mode, cut on paragraph, then line, then hard boundaries. Joined, the chunks give
 * back `text`, except that a trailing whitespace-only remainder is dropped.
 */
function splitText(text: string, maxChars: number): string[] {
  return chunkTextShared(text, { chunkSize: Math.max(1, Math.floor(maxChars)), exact: true, separators: SEPARATORS }).map((c) => c.text);
}

/** Re-measure each chunk with the real measure; a chunk still over `limit` is split again (bounded). */
async function fitChunks(chunks: readonly string[], limit: number, o: ChunkedTextSummaryOptions): Promise<string[]> {
  const out: string[] = [];
  for (const chunk of chunks) {
    let parts = [chunk];
    for (let attempt = 0; attempt < MAX_REFIT_ATTEMPTS; attempt++) {
      const next: string[] = [];
      let over = false;
      for (const part of parts) {
        throwIfAborted(o.signal);
        const tokens = await o.measure(part, o.signal);
        if (tokens <= limit || part.length <= 1) { next.push(part); continue; }
        over = true;
        next.push(...splitText(part, Math.max(1, Math.floor(((part.length / tokens) * limit) * 0.9))));
      }
      parts = next;
      if (!over) break;
    }
    out.push(...parts);
  }
  return out;
}

/** See the module comment. Resolves to the summary, or `null` when it cannot be made within the bounds. */
export async function chunkedTextSummary(o: ChunkedTextSummaryOptions): Promise<string | null> {
  return summarizeAt(o, 0);
}

async function summarizeAt(o: ChunkedTextSummaryOptions, depth: number): Promise<string | null> {
  const { text, signal } = o;
  throwIfAborted(signal);
  // The reserve never takes more than half the window, so a tiny window still holds a chunk.
  const room = Math.max(Math.floor(o.limit / 2), o.limit - (o.reserveTokens ?? 0));
  const limit = Math.max(1, Math.floor(room * (o.budgetShare ?? 1)));
  const tokens = await o.measure(text, signal);
  if (tokens <= limit) return o.summarizeChunk(text, signal);
  if (depth >= (o.maxMergeDepth ?? DEFAULT_MAX_MERGE_DEPTH)) return null;

  const chunks = await fitChunks(splitText(text, Math.max(1, Math.floor((text.length / tokens) * limit))), limit, o);
  if (chunks.length > o.maxChunks) return null;

  const summaries: string[] = [];
  for (const chunk of chunks) {
    throwIfAborted(signal);
    const summary = await o.summarizeChunk(chunk, signal);
    if (summary === null) return null;
    summaries.push(summary);
  }
  const merged = summaries.join('\n\n');
  // A merge that did not shrink the text will not converge.
  if (merged.length >= text.length) return null;
  return summarizeAt({ ...o, text: merged }, depth + 1);
}
