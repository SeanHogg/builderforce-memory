/**
 * The ONE narration-overlap normalizer (chat #168571, fix brief W5).
 *
 * `foldRepeatedNarration` (fold.ts, B7) and the host-side `repeatedNarration`
 * detector both measure how much an assistant narration restates an earlier one.
 * They share this module so the two can never drift:
 *
 *   - {@link wordOverlap} / {@link wordContained} / {@link isRepeatedNarration}:
 *     whole-text word-set measures (the original B7 detector, moved here verbatim).
 *   - {@link leadSentenceOverlap}: token Jaccard of the two texts' FIRST sentence,
 *     normalized case/punctuation/backtick-insensitively. A growing restatement
 *     (the same lead finding plus more detail every turn) keeps its lead sentence
 *     while its whole-text Jaccard falls below 0.8 — the lead measure is what catches it.
 *   - {@link newNarrationSentences}: the sentences of a later narration that are
 *     genuinely NEW against every earlier kept sentence — what survives a fold.
 *
 * Pure: no `node:*`, no I/O — runs wherever the `./working` subpath runs.
 */

/** B7's whole-text fold threshold: |intersection| / |union| of the two texts' word sets (Jaccard). */
export const NARRATION_OVERLAP = 0.8;

/** The lead-sentence fold threshold: a first sentence this similar marks a restatement. */
export const LEAD_SENTENCE_OVERLAP = 0.7;

/** What a folded narration's restated sentences collapse to (its new sentences survive). */
export const RESTATEMENT_NOTE = '(restates earlier finding)';

function wordSet(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/[^a-z0-9_]+/)
      .filter((w) => w.length > 0),
  );
}

/**
 * Jaccard similarity of two texts' word sets (lower-cased `[a-z0-9_]` words): |A ∩ B| / |A ∪ B|, 0 when either has no word.
 * The measure B7 folds repeated narration by (see {@link isRepeatedNarration}).
 */
export function wordOverlap(a: string, b: string): number {
  const sa = wordSet(a);
  const sb = wordSet(b);
  if (sa.size === 0 || sb.size === 0) return 0;
  let intersection = 0;
  for (const w of sa) if (sb.has(w)) intersection += 1;
  return intersection / (sa.size + sb.size - intersection);
}

/** Containment: is every word of the smaller set present in the larger? True when one
 *  narration is the other plus a trailing qualifier ("…once more", "…again", "…now") —
 *  a repeat Jaccard under-counts because the union grows with the qualifier's words. */
export function wordContained(a: string, b: string): boolean {
  const sa = wordSet(a);
  const sb = wordSet(b);
  if (sa.size === 0 || sb.size === 0) return false;
  const [small, large] = sa.size <= sb.size ? [sa, sb] : [sb, sa];
  for (const w of small) if (!large.has(w)) return false;
  return true;
}

/**
 * B7's whole-text detector: `text` merely repeats `previous` — their {@link wordOverlap} is at least `overlap`
 * (default {@link NARRATION_OVERLAP}), or one's words are all in the other ({@link wordContained}).
 */
export function isRepeatedNarration(text: string, previous: string, overlap = NARRATION_OVERLAP): boolean {
  return wordOverlap(text, previous) >= overlap || wordContained(text, previous);
}

/** Split text into sentences on sentence-ending punctuation followed by whitespace.
 *  Dots inside paths/identifiers (`fold.ts`, `server.json`) do not split — no whitespace follows them. */
export function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

/** The first sentence of a text ('' when it has none). */
function leadSentence(text: string): string {
  return splitSentences(text)[0] ?? '';
}

/**
 * Token Jaccard of the two texts' FIRST sentences, normalized case/punctuation/backtick-insensitively
 * (the {@link wordOverlap} word set drops everything outside `[a-z0-9_]`). 0 when either lead has no word.
 * The shared restatement measure: B7 folds at ≥ {@link LEAD_SENTENCE_OVERLAP}, and the host-side
 * `repeatedNarration` detector imports THIS function — it must not grow its own copy.
 */
export function leadSentenceOverlap(a: string, b: string): number {
  return wordOverlap(leadSentence(a), leadSentence(b));
}

/**
 * The sentences of `text` that are genuinely NEW: each one's own {@link wordOverlap} with EVERY
 * earlier kept sentence is below {@link LEAD_SENTENCE_OVERLAP}. What a folded narration keeps.
 */
export function newNarrationSentences(text: string, earlierKept: readonly string[]): string[] {
  return splitSentences(text).filter((s) => earlierKept.every((k) => wordOverlap(s, k) < LEAD_SENTENCE_OVERLAP));
}
