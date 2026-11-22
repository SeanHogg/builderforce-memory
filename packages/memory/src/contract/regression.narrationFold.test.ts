/**
 * src/contract/regression.narrationFold.test.ts — chat #168571 W5.
 *
 * The incident: the model restated the SAME lead finding ("The ranking is about the
 * hosted Builderforce MCP/server, not the memory package.") at the head of every
 * tool-call narration, adding detail each turn (completions grew 282 → 846 → 931 →
 * 2590 → 3124 → 3708 tokens). Whole-text Jaccard fell below 0.8 as the detail grew,
 * so B7 never folded and the restatement was re-read in full every turn (~555K
 * cumulative prompt tokens).
 *
 * Contract: `foldRepeatedNarration` folds when the LEAD-sentence overlap is ≥ 0.7
 * (`leadSentenceOverlap`, the ONE normalizer the host-side detector also imports)
 * OR whole-text overlap is ≥ 0.8; on fold only genuinely NEW sentences survive and
 * the restated ones collapse to "(restates earlier finding)".
 */

import {
  foldWorkingTranscript,
  leadSentenceOverlap,
  wordOverlap,
  LEAD_SENTENCE_OVERLAP,
  NARRATION_OVERLAP,
  PROGRESS_NOTE,
  RESTATEMENT_NOTE,
} from '../working/index.js';
import type { TranscriptMessage } from '../working/transcript.js';

const LEAD = 'The ranking is about the hosted Builderforce MCP/server, not the memory package.';

/** The 7 growing restatements, modeled on the incident narration. Each keeps the lead
 *  finding verbatim and grows the detail — the shape whole-text Jaccard misses. */
const NARRATIONS: readonly string[] = [
  `${LEAD} I'll check the recommendation against the hosted server.`,
  `${LEAD} I'll check the recommendation against the hosted server. The paste proposes letting clients call initialize and tools/list without a token.`,
  `${LEAD} I'll check the recommendation against the hosted server. The paste proposes letting clients call initialize and tools/list without a token. It also proposes adding an icon to server.json and republishing.`,
  `${LEAD} I'll check the recommendation against the hosted server. The paste proposes letting clients call initialize and tools/list without a token. It also proposes adding an icon to server.json and republishing. The scoring weights discovery signals, so an unauthenticated handshake would raise the listing score.`,
  `${LEAD} I'll check the recommendation against the hosted server. The paste proposes letting clients call initialize and tools/list without a token. It also proposes adding an icon to server.json and republishing. The scoring weights discovery signals, so an unauthenticated handshake would raise the listing score. Our gateway requires a tenant key on every route, so the anonymous handshake needs a scoped exemption.`,
  `${LEAD} I'll check the recommendation against the hosted server. The paste proposes letting clients call initialize and tools/list without a token. It also proposes adding an icon to server.json and republishing. The scoring weights discovery signals, so an unauthenticated handshake would raise the listing score. Our gateway requires a tenant key on every route, so the anonymous handshake needs a scoped exemption. The exemption must not expose tenant-scoped tools or any memory content.`,
  `${LEAD} I'll check the recommendation against the hosted server. The paste proposes letting clients call initialize and tools/list without a token. It also proposes adding an icon to server.json and republishing. The scoring weights discovery signals, so an unauthenticated handshake would raise the listing score. Our gateway requires a tenant key on every route, so the anonymous handshake needs a scoped exemption. The exemption must not expose tenant-scoped tools or any memory content. I'll verify the registry schema accepts an icon URL before republishing.`,
];

const readCall = (id: string, path: string) => ({
  id,
  function: { name: 'read_file', arguments: JSON.stringify({ path }) },
});

/** user, then 7 (assistant narration + tool result) pairs, then a final answer. */
function incidentTranscript(): TranscriptMessage[] {
  const msgs: TranscriptMessage[] = [{ role: 'user', content: 'Review the recommendation for the MCP below; Is this something we want to do?' }];
  NARRATIONS.forEach((narration, i) => {
    msgs.push({ role: 'assistant', content: narration, tool_calls: [readCall(`r${i}`, `src/file${i}.ts`)] });
    msgs.push({ role: 'tool', tool_call_id: `r${i}`, content: `file ${i} text` });
  });
  msgs.push({ role: 'assistant', content: 'Recommendation reviewed: the anonymous handshake is not something we want; the icon is fine.' });
  return msgs;
}

describe('regression: chat #168571 — growing restatements fold, new sentences survive', () => {
  it('regression: 7 growing restatements — turns 5/7/9/11 fold on the lead sentence, genuinely new sentences survive', () => {
    const msgs = incidentTranscript();
    const out = foldWorkingTranscript(msgs);

    // The first narration (turn 3) is kept verbatim — nothing to restate yet.
    expect(out[1]).toBe(msgs[1]);

    // The later restatements (turns 5, 7, 9, 11, 13, 15) all fold: their lead sentence
    // is the same finding, even where whole-text Jaccard has fallen below 0.8.
    let sawBelowWholeTextThreshold = false;
    for (const idx of [3, 5, 7, 9, 11, 13]) {
      const original = msgs[idx]!.content as string;
      const folded = out[idx]!.content as string;
      expect(leadSentenceOverlap(original, NARRATIONS[0]!)).toBeGreaterThanOrEqual(LEAD_SENTENCE_OVERLAP);
      if (wordOverlap(original, NARRATIONS[0]!) < NARRATION_OVERLAP) sawBelowWholeTextThreshold = true;
      expect(folded).toContain(RESTATEMENT_NOTE);
      expect(folded).not.toContain(LEAD); // the restated lead is gone from the model's window
      expect(folded.length).toBeLessThan(original.length);
    }
    // The incident's actual failure mode is exercised: at least one restatement was
    // BELOW the old whole-text threshold and only the lead rule catches it.
    expect(sawBelowWholeTextThreshold).toBe(true);

    // Every genuinely new sentence survives exactly once, on the turn that introduced it.
    const newSentences = [
      'The paste proposes letting clients call initialize and tools/list without a token.',
      'It also proposes adding an icon to server.json and republishing.',
      'The scoring weights discovery signals, so an unauthenticated handshake would raise the listing score.',
      'Our gateway requires a tenant key on every route, so the anonymous handshake needs a scoped exemption.',
      'The exemption must not expose tenant-scoped tools or any memory content.',
      "I'll verify the registry schema accepts an icon URL before republishing.",
    ];
    const foldedAll = out.map((m) => (typeof m.content === 'string' ? m.content : '')).join('\n');
    for (const sentence of newSentences) {
      expect(foldedAll.split(sentence).length - 1).toBe(1);
    }
    // Each new sentence lands on its own introducing turn (assistant indexes 3..13).
    newSentences.forEach((sentence, i) => {
      expect(out[3 + i * 2]!.content as string).toContain(sentence);
    });

    // The final answer is never folded; the caller's transcript keeps full fidelity.
    expect(out[15]).toBe(msgs[15]);
    expect(msgs[13]!.content).toBe(NARRATIONS[6]);
    expect(out).toHaveLength(msgs.length);
  });

  it('leadSentenceOverlap is case/punctuation/backtick-insensitive on the first sentence only', () => {
    const a = 'The `ranking` is about the HOSTED Builderforce MCP/server, not the memory package! Different tail here.';
    const b = 'the ranking is about the hosted builderforce mcp server not the memory package. Another tail entirely.';
    expect(leadSentenceOverlap(a, b)).toBe(1);
    // Whole texts differ; the lead measure ignores the tails.
    expect(wordOverlap(a, b)).toBeLessThan(1);
    expect(leadSentenceOverlap('', b)).toBe(0);
    expect(leadSentenceOverlap('...', '...')).toBe(0);
  });

  it('a pure repeat with nothing new still collapses to the progress note (existing B7 behaviour)', () => {
    const msgs: TranscriptMessage[] = [
      { role: 'assistant', content: `${LEAD} Checking the server.`, tool_calls: [readCall('r1', 'a.ts')] },
      { role: 'tool', tool_call_id: 'r1', content: 'text' },
      { role: 'assistant', content: `${LEAD} Checking the server.`, tool_calls: [readCall('r2', 'b.ts')] },
      { role: 'tool', tool_call_id: 'r2', content: 'text' },
    ];
    const out = foldWorkingTranscript(msgs);
    expect(out[2]!.content).toBe(PROGRESS_NOTE);
  });
});
