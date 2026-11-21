/**
 * src/contract/regression.inRunWindow.test.ts — the drop-oldest window never cut INSIDE a run.
 *
 * VS Code chat #128305 (VSIX 2026.10.204): one long run's prompt grew from 31k tokens at step 0 by ~4–5k a turn to 191k,
 * with no reduction. The fold's summary kept timing out, so every turn fell back to `windowed`, which clamps its cut at
 * the last user turn — and inside one run that turn is the prompt, so nothing after it was ever dropped and the 64k
 * history budget never applied. The window now elides the run's older tool results (`elideRunResults`): the user turn,
 * every assistant turn and a ~40k verbatim tail stay; a call and its result are never split.
 */

import {
    buildWorkingTranscript,
    COMPACT_TAIL_TOKEN_BUDGET,
    elidedResultStub,
    elideRunResults,
    HISTORY_TOKEN_BUDGET,
    MIN_ELIDED_RESULT_TOKENS,
    stillInWorkingContext,
    transcriptTokens,
    windowed,
    type TranscriptMessage,
    type WorkingTranscriptState,
} from '../working/index.js';

/** ~4.5k tokens: one file window of a coding run. */
const RESULT = (i: number) => `// file ${i}\n${'x'.repeat(18_000)}`;

/** One acting turn: an assistant tool call and the result that answers it. */
function step(i: number, tool = 'read_file', content = RESULT(i)): TranscriptMessage[] {
    const id = `call_${i}`;
    return [
        {
            role: 'assistant',
            content: `Reading the next file (${i}).`,
            tool_calls: [{ id, type: 'function', function: { name: tool, arguments: JSON.stringify({ path: `src/f${i}.ts` }) } }],
        },
        { role: 'tool', tool_call_id: id, content },
    ];
}

/** A single run: the person's request, then `turns` tool steps and nothing else. */
function run(turns: number): TranscriptMessage[] {
    const out: TranscriptMessage[] = [{ role: 'user', content: 'Review the Visualizer and fix what is broken.' }];
    for (let i = 0; i < turns; i++) out.push(...step(i));
    return out;
}

/** Every assistant tool call is answered by the row right after it, and every tool row answers the call before it. */
function pairingIntact(msgs: readonly TranscriptMessage[]): boolean {
    return msgs.every((m, i) => {
        if (m.role !== 'tool') return true;
        const prev = msgs[i - 1] as TranscriptMessage & { tool_calls?: Array<{ id: string }> };
        return prev?.role === 'assistant' && (prev.tool_calls ?? []).some((c) => c.id === m.tool_call_id);
    });
}

describe('regression: chat #128305 — the fallback window cuts inside one long run', () => {
    it('regression: a 40-turn run stays inside the history budget, keeps every row, the request and a verbatim tail', () => {
        const convo = run(40);
        expect(transcriptTokens(convo)).toBeGreaterThan(2.5 * HISTORY_TOKEN_BUDGET); // ~182k: what every turn used to send
        const w = windowed(convo);
        expect(transcriptTokens(w)).toBeLessThan(48_000); // ~44k: the ~40k tail, the request, every call and its stub
        expect(transcriptTokens(w)).toBeLessThanOrEqual(HISTORY_TOKEN_BUDGET);
        expect(w).toHaveLength(convo.length); // nothing dropped: rows are elided in place
        expect(w[0]).toBe(convo[0]); // the request, verbatim
        expect(pairingIntact(w)).toBe(true);
        // Every assistant turn is the transcript's own row.
        w.forEach((m, i) => { if (m.role === 'assistant') expect(m).toBe(convo[i]); });
        // The most recent results are verbatim, and together within the tail budget.
        const firstVerbatim = w.findIndex((m, i) => m.role === 'tool' && m === convo[i]);
        expect(firstVerbatim).toBeGreaterThan(1);
        expect(w.slice(firstVerbatim).every((m, i) => m === convo[firstVerbatim + i])).toBe(true);
        expect(transcriptTokens(w.slice(firstVerbatim))).toBeLessThanOrEqual(COMPACT_TAIL_TOKEN_BUDGET);
        // An older result is one line naming the tool, its target and its size.
        expect(w[2]!.content).toBe(elidedResultStub('read_file', 'src/f0.ts', String(convo[2]!.content).length));
        expect(w[2]!.content).toBe('[result elided: read_file src/f0.ts, 18 KB — re-read if needed]');
        expect(w[2]!.tool_call_id).toBe('call_0');
    });

    it('regression: with the summary failing every turn, buildWorkingTranscript stays bounded across the whole run', async () => {
        const state: WorkingTranscriptState = { transcript: [], compactMemo: null };
        let peak = 0;
        for (let turn = 1; turn <= 40; turn++) {
            state.transcript = run(turn);
            const out = await buildWorkingTranscript(state, 'sys', async () => null, () => undefined);
            expect(pairingIntact(out.slice(1))).toBe(true);
            const sent = transcriptTokens(out.slice(1));
            // Once over the budget, the cut leaves the ~40k tail plus the calls and their stubs — on EVERY turn, not only some.
            if (transcriptTokens(state.transcript) > HISTORY_TOKEN_BUDGET) expect(sent).toBeLessThan(48_000);
            peak = Math.max(peak, sent);
        }
        expect(peak).toBeLessThanOrEqual(HISTORY_TOKEN_BUDGET);
    });

    it('regression: a memo held from an earlier fold does not let the verbatim remainder grow without bound when the next summary fails', async () => {
        const convo = run(40);
        const state: WorkingTranscriptState = { transcript: convo, compactMemo: { note: 'Compressed memory: the first steps.', coveredEnd: 3 } };
        const out = await buildWorkingTranscript(state, 'sys', async () => null, () => undefined);
        expect(out[0]).toEqual({ role: 'system', content: 'sys' });
        expect(out[1]!.content).toBe('Compressed memory: the first steps.');
        expect(transcriptTokens(out.slice(1))).toBeLessThanOrEqual(HISTORY_TOKEN_BUDGET);
        expect(pairingIntact(out.slice(1).filter((m) => m.role !== 'user'))).toBe(true);
        expect(state.compactMemo!.coveredEnd).toBe(3); // a failed summary keeps the note it had
    });

    it('regression: an elided result is out of the working context (the read cache replays it), a tail result is in', () => {
        const convo = run(40);
        const state: WorkingTranscriptState = { transcript: convo, compactMemo: null };
        expect(stillInWorkingContext(state, convo[2])).toBe(false);
        expect(stillInWorkingContext(state, convo[convo.length - 1])).toBe(true);
        const memoState: WorkingTranscriptState = { transcript: convo, compactMemo: { note: 'n', coveredEnd: 3 } };
        expect(stillInWorkingContext(memoState, convo[4])).toBe(false);
        expect(stillInWorkingContext(memoState, convo[convo.length - 1])).toBe(true);
    });

    it('regression: the elided prefix moves in chunks, not on every turn (prefix-cache stability)', () => {
        let changes = 0;
        let previous = -1;
        for (let turn = 15; turn <= 40; turn++) {
            const w = windowed(run(turn));
            const cut = w.filter((m) => m.role === 'tool' && String(m.content).startsWith('[result elided:')).length;
            if (cut !== previous) changes += 1;
            previous = cut;
        }
        // 26 turns of ~4.5k tokens each, a 16k chunk: the cut moves about once every 3–4 turns.
        expect(changes).toBeLessThan(13);
    });

    it('keeps plan/ledger results, small results, and everything before the run', () => {
        const big = 'y'.repeat(18_000);
        const convo: TranscriptMessage[] = [
            { role: 'user', content: 'earlier request' },
            { role: 'assistant', content: 'earlier answer' },
            { role: 'user', content: 'the request' },
            ...step(0, 'update_plan', big),
            ...step(1, 'read_file', 'tiny'),
        ];
        for (let i = 2; i < 30; i++) convo.push(...step(i));
        const w = elideRunResults(convo, { budgetTokens: HISTORY_TOKEN_BUDGET, tailTokens: COMPACT_TAIL_TOKEN_BUDGET, chunkTokens: 16_000, minTailMessages: 8 });
        expect(w.slice(0, 3)).toEqual(convo.slice(0, 3));
        expect(w[4]).toBe(convo[4]); // the plan result
        expect(w[6]).toBe(convo[6]); // under MIN_ELIDED_RESULT_TOKENS
        expect(MIN_ELIDED_RESULT_TOKENS).toBeGreaterThan(0);
        expect(String(w[8]!.content)).toMatch(/^\[result elided: read_file src\/f2\.ts, \d+ KB — re-read if needed\]$/);
    });

    it('returns the input itself when it fits, and is idempotent once elided', () => {
        const short = run(3);
        expect(windowed(short)).toEqual(short);
        const budget = { budgetTokens: HISTORY_TOKEN_BUDGET, tailTokens: COMPACT_TAIL_TOKEN_BUDGET, chunkTokens: 16_000, minTailMessages: 8 };
        expect(elideRunResults(short, budget)).toBe(short);
        const once = elideRunResults(run(40), budget);
        expect(elideRunResults(once, budget)).toBe(once);
    });

    it('names a positional result (no tool_call_id) by the call before it, and stubs without a target when none is named', () => {
        const convo: TranscriptMessage[] = [{ role: 'user', content: 'go' }];
        for (let i = 0; i < 30; i++) {
            convo.push({ role: 'assistant', content: '', tool_calls: [{ id: '', type: 'function', function: { name: 'search_code', arguments: '{}' } }] });
            convo.push({ role: 'tool', content: RESULT(i) });
        }
        const w = windowed(convo);
        expect(w[2]!.content).toBe(elidedResultStub('search_code', null, String(convo[2]!.content).length));
        expect(String(w[2]!.content)).toBe('[result elided: search_code, 18 KB — re-read if needed]');
    });
});
