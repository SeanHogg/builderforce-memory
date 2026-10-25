/**
 * src/working/summary.test.ts
 * The ONE transcript summarizer: what it asks, how it renders tool calls, and that a
 * failing or empty completion reads as "no summary" (the caller falls back).
 */

import { renderForSummary, summaryChunks, summarizeTranscript, TRANSCRIPT_SUMMARY_MAX_TOKENS, TRANSCRIPT_SUMMARY_PROMPT } from './summary.js';
import { buildWorkingTranscript, HISTORY_TOKEN_BUDGET, type TranscriptFold, type TranscriptMessage, type WorkingTranscriptState } from './transcript.js';

describe('renderForSummary', () => {
  it('names the tools an assistant turn called', () => {
    const text = renderForSummary([
      { role: 'user', content: 'fix it' },
      { role: 'assistant', content: '', tool_calls: [{ function: { name: 'read_file' } }, { function: { name: 'edit' } }] },
      { role: 'tool', content: { ok: true } },
    ]);
    expect(text).toBe('user: fix it\n\nassistant [called: read_file, edit]: \n\ntool: {"ok":true}');
  });
});

describe('summarizeTranscript', () => {
  it('asks the shared prompt with the bounded budget and trims the answer', async () => {
    const seen: Array<{ system: string; maxTokens: number }> = [];
    const out = await summarizeTranscript(async (req) => { seen.push(req); return '  the note  '; }, [{ role: 'user', content: 'x' }]);
    expect(out).toBe('the note');
    expect(seen[0]).toMatchObject({ system: TRANSCRIPT_SUMMARY_PROMPT, maxTokens: TRANSCRIPT_SUMMARY_MAX_TOKENS });
  });

  it('is null for nothing to summarize, an empty answer, or a failure', async () => {
    expect(await summarizeTranscript(async () => 'x', [])).toBeNull();
    expect(await summarizeTranscript(async () => '   ', [{ role: 'user', content: 'x' }])).toBeNull();
    expect(await summarizeTranscript(async () => { throw new Error('down'); }, [{ role: 'user', content: 'x' }])).toBeNull();
  });
});

describe('buildWorkingTranscript fold report', () => {
  it('reports the note and the index it covers up to', async () => {
    const big = 'x'.repeat(HISTORY_TOKEN_BUDGET); // ~16k tokens each
    const state: WorkingTranscriptState = {
      transcript: Array.from({ length: 12 }, (_, i): TranscriptMessage => ({ role: i % 2 ? 'assistant' : 'user', content: big })),
      compactMemo: null,
    };
    const folds: TranscriptFold[] = [];
    const out = await buildWorkingTranscript(state, 'sys', async () => 'summary', (f) => folds.push(f));
    expect(folds).toHaveLength(1);
    expect(folds[0]!.droppedMessages).toBe(folds[0]!.coveredEnd);
    expect(folds[0]!.note).toContain('summary');
    expect(state.compactMemo).toEqual({ note: folds[0]!.note, coveredEnd: folds[0]!.coveredEnd });
    expect(out[1]).toEqual({ role: 'assistant', content: folds[0]!.note });
  });
});

describe('summarizeTranscript over a transcript longer than one call reads', () => {
  const long = (n: number): TranscriptMessage[] =>
    Array.from({ length: n }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: `turn ${i} ${'x'.repeat(400)}` }));

  it('splits into consecutive chunks within the budget, losing nothing', () => {
    const msgs = long(40);
    const chunks = summaryChunks(msgs, 1_000);
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.flat()).toEqual(msgs);
  });

  it('folds each chunk onto the note so far and returns the last note', async () => {
    const inputs: string[] = [];
    let call = 0;
    const out = await summarizeTranscript(async (req) => { inputs.push(req.user); return `note ${++call}`; }, long(40), undefined, 1_000);
    expect(call).toBeGreaterThan(1);
    expect(out).toBe(`note ${call}`);
    expect(inputs[0]).not.toContain('note ');
    expect(inputs[1]!.startsWith('assistant: note 1')).toBe(true);
  });

  it('is null when any chunk fails — never a note that skipped part of the history', async () => {
    let call = 0;
    expect(await summarizeTranscript(async () => (++call === 2 ? '' : 'n'), long(40), undefined, 1_000)).toBeNull();
  });

  it('clips one message larger than the whole budget', async () => {
    let seen = '';
    await summarizeTranscript(async (req) => { seen = req.user; return 'n'; }, [{ role: 'user', content: 'y'.repeat(50_000) }], undefined, 1_000);
    expect(seen.length).toBeLessThan(5_000);
    expect(seen.endsWith('[…clipped]')).toBe(true);
  });
});
