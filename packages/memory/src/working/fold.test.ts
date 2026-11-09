/**
 * src/working/fold.test.ts — PRD 45 §4.7 B6 and B7.
 *
 * B6: a read of path P followed by a successful edit/write of P at step N folds the
 * earlier read result to "(read of P superseded by the edit at step N — re-read if
 * needed)". The trace (the caller's transcript) keeps full fidelity.
 *
 * B7: assistant narration preceding tool calls whose word-set overlap with the previous
 * such narration is ≥ 0.8 becomes "(progress note)". A final answer and a turn without
 * tool calls are NEVER folded.
 */

import {
  foldRepeatedNarration,
  foldSupersededReads,
  foldWorkingTranscript,
  NARRATION_OVERLAP,
  PROGRESS_NOTE,
  supersededReadMarker,
} from './fold.js';
import type { TranscriptMessage } from './transcript.js';

const readCall = (id: string, path: string) => ({
  id,
  function: { name: 'read_file', arguments: JSON.stringify({ path }) },
});
const editCall = (id: string, path: string) => ({
  id,
  function: { name: 'edit_file', arguments: JSON.stringify({ path, old_string: 'a', new_string: 'b' }) },
});

describe('B6 — fold superseded reads', () => {
  it('regression: a read of P followed by an edit of P at step N folds the earlier read with the right step', () => {
    const msgs: TranscriptMessage[] = [
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'fix src/a.ts' },
      { role: 'assistant', content: 'Reading the file first.', tool_calls: [readCall('r1', 'src/a.ts')] },
      { role: 'tool', tool_call_id: 'r1', content: 'the whole file text' },
      // step 4: the successful edit of the same path.
      { role: 'assistant', content: 'Applying the fix.', tool_calls: [editCall('e1', 'src/a.ts')] },
      { role: 'tool', tool_call_id: 'e1', content: JSON.stringify({ ok: true }) },
      { role: 'assistant', content: 'Done.' },
    ];
    const out = foldSupersededReads(msgs);
    expect(out[3]).toEqual({ role: 'tool', tool_call_id: 'r1', content: supersededReadMarker('src/a.ts', 4) });
    expect(out[3]!.content).toBe('(read of src/a.ts superseded by the edit at step 4 — re-read if needed)');
    // The successful edit's own result, and everything else, pass through untouched.
    expect(out[5]).toBe(msgs[5]);
    expect(out).toHaveLength(msgs.length);
    // Full fidelity in the trace: the caller's transcript is NOT mutated.
    expect(msgs[3]!.content).toBe('the whole file text');
  });

  it('a read re-made after the edit is not superseded by that edit', () => {
    const msgs: TranscriptMessage[] = [
      { role: 'assistant', content: '', tool_calls: [readCall('r1', 'p.ts')] },
      { role: 'tool', tool_call_id: 'r1', content: 'v1' },
      { role: 'assistant', content: '', tool_calls: [editCall('e1', 'p.ts')] },
      { role: 'tool', tool_call_id: 'e1', content: { ok: true } },
      { role: 'assistant', content: '', tool_calls: [readCall('r2', 'p.ts')] },
      { role: 'tool', tool_call_id: 'r2', content: 'v2 after edit' },
    ];
    const out = foldSupersededReads(msgs);
    expect(out[1]!.content).toBe(supersededReadMarker('p.ts', 2));
    expect(out[5]!.content).toBe('v2 after edit'); // read AFTER the edit: current, kept verbatim
  });

  it('a failed edit supersedes nothing', () => {
    const msgs: TranscriptMessage[] = [
      { role: 'assistant', content: '', tool_calls: [readCall('r1', 'p.ts')] },
      { role: 'tool', tool_call_id: 'r1', content: 'text' },
      { role: 'assistant', content: '', tool_calls: [editCall('e1', 'p.ts')] },
      { role: 'tool', tool_call_id: 'e1', content: { ok: false, error: 'no match' } },
    ];
    const out = foldSupersededReads(msgs);
    expect(out[1]).toBe(msgs[1]); // same reference: nothing folded
    expect(out[1]!.content).toBe('text');
  });

  it('an edit of a different path supersedes nothing', () => {
    const msgs: TranscriptMessage[] = [
      { role: 'assistant', content: '', tool_calls: [readCall('r1', 'a.ts')] },
      { role: 'tool', tool_call_id: 'r1', content: 'text of a' },
      { role: 'assistant', content: '', tool_calls: [editCall('e1', 'b.ts')] },
      { role: 'tool', tool_call_id: 'e1', content: { ok: true } },
    ];
    const out = foldSupersededReads(msgs);
    expect(out[1]!.content).toBe('text of a');
  });
});

describe('B7 — fold repeated narration', () => {
  it('regression (chat #216): two assistant texts before tool calls with ≥0.8 word overlap → the later is a progress note', () => {
    const msgs: TranscriptMessage[] = [
      { role: 'user', content: 'wire the folds in' },
      {
        role: 'assistant',
        content: 'Now reading the transcript builder to find where the fold should run.',
        tool_calls: [readCall('r1', 'src/working/transcript.ts')],
      },
      { role: 'tool', tool_call_id: 'r1', content: 'export async function buildWorkingTranscript…' },
      {
        role: 'assistant',
        content: 'Now reading the transcript builder again to find where the fold should run next.',
        tool_calls: [readCall('r2', 'src/working/fold.ts')],
      },
      { role: 'tool', tool_call_id: 'r2', content: 'export function foldWorkingTranscript…' },
    ];
    const out = foldRepeatedNarration(msgs);
    expect(out[1]).toBe(msgs[1]); // first narration is kept verbatim
    expect(out[3]!.content).toBe(PROGRESS_NOTE);
    expect(wordOverlapWouldFold(msgs[3]!.content as string, msgs[1]!.content as string)).toBe(true);
  });

  it('a final answer is NEVER folded', () => {
    const msgs: TranscriptMessage[] = [
      {
        role: 'assistant',
        content: 'Now reading the transcript builder to find where the fold should run.',
        tool_calls: [readCall('r1', 'src/working/transcript.ts')],
      },
      { role: 'tool', tool_call_id: 'r1', content: 'text' },
      // The FINAL answer: same words as the earlier narration, but no tool calls.
      { role: 'assistant', content: 'Now reading the transcript builder to find where the fold should run.' },
    ];
    const out = foldRepeatedNarration(msgs);
    expect(out[2]).toBe(msgs[2]);
    expect(out[2]!.content).not.toBe(PROGRESS_NOTE);
  });

  it('a turn without tool calls is NEVER folded, even mid-run', () => {
    const msgs: TranscriptMessage[] = [
      {
        role: 'assistant',
        content: 'Reading the file to find where the fold should run in the builder.',
        tool_calls: [readCall('r1', 'a.ts')],
      },
      { role: 'tool', tool_call_id: 'r1', content: 'text' },
      { role: 'assistant', content: 'Reading the file to find where the fold should run in the builder.' }, // no tool_calls
    ];
    const out = foldRepeatedNarration(msgs);
    expect(out[2]!.content).not.toBe(PROGRESS_NOTE);
  });

  it('narration below the 0.8 overlap threshold is kept', () => {
    const msgs: TranscriptMessage[] = [
      { role: 'assistant', content: 'Reading the auth middleware to see how tokens verify.', tool_calls: [readCall('r1', 'auth.ts')] },
      { role: 'tool', tool_call_id: 'r1', content: 'text' },
      { role: 'assistant', content: 'Editing the auth middleware so expired tokens refresh.', tool_calls: [editCall('e1', 'auth.ts')] },
    ];
    const out = foldRepeatedNarration(msgs);
    expect(out[2]!.content).toBe('Editing the auth middleware so expired tokens refresh.');
  });

  it('a folded note does not become the comparison text', () => {
    const msgs: TranscriptMessage[] = [
      { role: 'assistant', content: 'Reading the builder to locate the fold hook now.', tool_calls: [readCall('r1', 'a.ts')] },
      { role: 'tool', tool_call_id: 'r1', content: 'text' },
      { role: 'assistant', content: 'Reading the builder to locate the fold hook now, again.', tool_calls: [readCall('r2', 'b.ts')] },
      { role: 'tool', tool_call_id: 'r2', content: 'text' },
      // Different words from the FIRST narration: must NOT fold just because it differs from the folded one.
      { role: 'assistant', content: 'Searching the repo for other callers of the fold hook.', tool_calls: [readCall('r3', 'c.ts')] },
    ];
    const out = foldRepeatedNarration(msgs);
    expect(out[2]!.content).toBe(PROGRESS_NOTE);
    expect(out[4]!.content).toBe('Searching the repo for other callers of the fold hook.');
  });
});

describe('foldWorkingTranscript — B6 then B7 together', () => {
  it('applies both folds over the model transcript without mutating the input', () => {
    const msgs: TranscriptMessage[] = [
      { role: 'assistant', content: 'Reading the config to find the lever wiring.', tool_calls: [readCall('r1', 'cfg.ts')] },
      { role: 'tool', tool_call_id: 'r1', content: 'lever config text' },
      { role: 'assistant', content: 'Reading the config to find the lever wiring once more.', tool_calls: [editCall('e1', 'cfg.ts')] },
      { role: 'tool', tool_call_id: 'e1', content: { ok: true } },
    ];
    const out = foldWorkingTranscript(msgs);
    expect(out[1]!.content).toBe(supersededReadMarker('cfg.ts', 2));
    expect(out[2]!.content).toBe(PROGRESS_NOTE);
    expect(msgs[1]!.content).toBe('lever config text'); // trace intact
    expect(msgs[2]!.content).toContain('once more');
  });
});

/** Local re-derivation for the assertion above: the two narrations must actually meet the
 *  exported threshold (kept honest if NARRATION_OVERLAP ever changes). */
function wordOverlapWouldFold(a: string, b: string): boolean {
  const words = (t: string) => new Set(t.toLowerCase().split(/[^a-z0-9_]+/).filter((w) => w.length > 0));
  const sa = words(a);
  const sb = words(b);
  let intersection = 0;
  for (const w of sa) if (sb.has(w)) intersection += 1;
  const union = sa.size + sb.size - intersection;
  return union > 0 && intersection / union >= NARRATION_OVERLAP;
}
