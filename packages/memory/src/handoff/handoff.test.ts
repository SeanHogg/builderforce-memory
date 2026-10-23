/**
 * src/handoff/handoff.test.ts
 * `packHandoff` (PRD 34 §3.3) — the hand-off packing rules: untouched under budget, the
 * dropped middle as ONE note (summary → recall → elision), tool pairs never split, the
 * latest directive verbatim, system kept, the from→to line, and `fits:false` only when
 * the irreducible core overflows.
 */

import { handoffBudget, packHandoff, type HandoffMessage, type HandoffPack } from './index.js';

/** A message costing exactly 14 tokens under the default estimate (40 chars / 4 + 4). */
function msg(role: HandoffMessage['role'], tag: string): HandoffMessage {
  return { role, content: tag.padEnd(40, '.') };
}

const SYSTEM = msg('system', 'sys');

/** u1 a1 u2 a2 u3 a3 u4 a4 — the latest directive is u4. */
function chat(): HandoffMessage[] {
  const out: HandoffMessage[] = [SYSTEM];
  for (let i = 1; i <= 4; i++) out.push(msg('user', `u${i}`), msg('assistant', `a${i}`));
  return out;
}

const BUDGET = { targetContextTokens: 100, reserveOutputTokens: 20 }; // 80 usable

function expectWithinBudget(pack: HandoffPack, budget = 80): void {
  expect(pack.fits).toBe(true);
  expect(pack.estimatedTokens).toBeLessThanOrEqual(budget);
}

/** The note (right after the leading system message); its role must alternate with what follows. */
function noteOf(pack: HandoffPack): string {
  const note = pack.messages[1]!;
  const next = pack.messages[2];
  expect(note.role).toBe(next === undefined || next.role === 'assistant' ? 'user' : 'assistant');
  return note.content as string;
}

/** No two consecutive user turns and no two consecutive assistant turns. */
function expectAlternating(messages: readonly HandoffMessage[]): void {
  for (let i = 1; i < messages.length; i++) {
    const [a, b] = [messages[i - 1]!.role, messages[i]!.role];
    if (a === 'user' || a === 'assistant') expect(b).not.toBe(a);
  }
}

describe('handoffBudget', () => {
  it('subtracts the reply reserve and tool schemas, never below zero', () => {
    expect(handoffBudget({ targetContextTokens: 8192, reserveOutputTokens: 1024, toolsTokens: 500 })).toBe(6668);
    expect(handoffBudget({ targetContextTokens: 4096, reserveOutputTokens: 96 })).toBe(4000);
    expect(handoffBudget({ targetContextTokens: 100, reserveOutputTokens: 200 })).toBe(0);
  });
});

describe('packHandoff', () => {
  it('hands the transcript over untouched when it fits', async () => {
    const messages = chat();
    const pack = await packHandoff({ messages, targetContextTokens: 10_000, reserveOutputTokens: 100, from: 'a', to: 'b' });
    expect(pack.messages).toEqual(messages);
    expect(pack).toMatchObject({ dropped: 0, summarized: false, recalled: false, fits: true });
    expect(pack.estimatedTokens).toBe(14 * messages.length);
  });

  it('drops the middle and puts the summary in ONE note after the system message', async () => {
    const seen: HandoffMessage[][] = [];
    const pack = await packHandoff({
      messages: chat(),
      ...BUDGET,
      summarize: async (dropped) => { seen.push([...dropped]); return 'the earlier work, summarized'; },
    });
    expectWithinBudget(pack);
    expect(pack.summarized).toBe(true);
    expect(pack.recalled).toBe(false);
    expect(pack.dropped).toBeGreaterThan(0);
    expect(pack.messages[0]).toBe(SYSTEM);
    expect(noteOf(pack)).toBe('the earlier work, summarized');
    expect(seen).toHaveLength(1);
    expect(seen[0]).toHaveLength(pack.dropped);
    expect(seen[0]!.some((m) => m.role === 'system')).toBe(false);
    // The tail is verbatim and ends on the last message.
    expect(pack.messages[pack.messages.length - 1]!.content).toBe(msg('assistant', 'a4').content);
    expect(pack.messages.filter((m) => m.content === 'the earlier work, summarized')).toHaveLength(1);
    expectAlternating(pack.messages);
  });

  it('cuts an oversized summary to the room left instead of overflowing', async () => {
    const pack = await packHandoff({ messages: chat(), ...BUDGET, summarize: async () => 'y'.repeat(5_000) });
    expectWithinBudget(pack);
    expect(pack.summarized).toBe(true);
    expect(noteOf(pack).endsWith('…')).toBe(true);
  });

  it('falls back to recall for the latest directive when summarize fails', async () => {
    const calls: Array<[string, number]> = [];
    const pack = await packHandoff({
      messages: chat(),
      ...BUDGET,
      summarize: async () => { throw new Error('summarizer down'); },
      recall: async (query, budgetChars) => { calls.push([query, budgetChars]); return 'remembered: X'; },
    });
    expectWithinBudget(pack);
    expect(pack).toMatchObject({ summarized: false, recalled: true });
    expect(calls).toHaveLength(1);
    expect(calls[0]![0]).toBe(msg('user', 'u4').content);
    expect(calls[0]![1]).toBeGreaterThan(0);
    expect(noteOf(pack)).toBe('remembered: X');
  });

  it('treats a summarize that outlives timeoutMs as a failure', async () => {
    const pack = await packHandoff({
      messages: chat(),
      ...BUDGET,
      timeoutMs: 10,
      summarize: () => new Promise<string>(() => { /* never settles */ }),
      recall: async () => 'recalled instead',
    });
    expect(pack.recalled).toBe(true);
    expect(noteOf(pack)).toBe('recalled instead');
  });

  it('falls back to the elision marker when neither summary nor recall is available', async () => {
    const pack = await packHandoff({
      messages: chat(),
      ...BUDGET,
      summarize: async () => '   ',
      recall: async () => { throw new Error('no memory'); },
    });
    expectWithinBudget(pack);
    expect(pack).toMatchObject({ summarized: false, recalled: false });
    expect(noteOf(pack)).toBe(`[${pack.dropped} earlier messages omitted]`);

    const bare = await packHandoff({ messages: chat(), ...BUDGET });
    expect(noteOf(bare)).toBe(`[${bare.dropped} earlier messages omitted]`);
  });

  it('never splits an assistant tool_calls message from its tool results', async () => {
    const call = {
      role: 'assistant' as const,
      content: '',
      tool_calls: [{ id: 'c1', type: 'function', function: { name: 'read', arguments: '{}' } }, { id: 'c2', type: 'function', function: { name: 'read', arguments: '{}' } }],
    };
    const messages: HandoffMessage[] = [
      SYSTEM,
      msg('user', 'go'),
      call,
      { role: 'tool', tool_call_id: 'c1', content: 'z'.repeat(400) },
      { role: 'tool', tool_call_id: 'c2', content: 'small result'.padEnd(40, '.') },
      msg('assistant', 'done'),
    ];
    for (const target of [70, 90, 140, 250]) {
      const pack = await packHandoff({ messages, targetContextTokens: target, reserveOutputTokens: 0 });
      expectWithinBudget(pack, target);
      const out = pack.messages;
      const hasCall = out.includes(call);
      expect(out.includes(messages[3]!)).toBe(hasCall);
      expect(out.includes(messages[4]!)).toBe(hasCall);
      out.forEach((m, i) => {
        if (m.role === 'tool') expect(out[i - 1] === call || out[i - 1]!.role === 'tool').toBe(true);
      });
    }
  });

  it('re-injects the latest directive verbatim right after the note when it falls in the middle', async () => {
    const directive = msg('user', 'fix the bug');
    const messages: HandoffMessage[] = [SYSTEM, msg('user', 'old task'), msg('assistant', 'ok'), directive];
    for (let i = 0; i < 10; i++) messages.push(msg('assistant', `step${i}`));
    const pack = await packHandoff({ messages, ...BUDGET });
    expectWithinBudget(pack);
    expect(pack.messages[0]).toBe(SYSTEM);
    expect(pack.messages[2]).toBe(directive);
    expect(pack.messages.filter((m) => m === directive)).toHaveLength(1);
    expect(pack.messages[pack.messages.length - 1]!.content).toBe(msg('assistant', 'step9').content);
    expect(pack.dropped).toBe(messages.length - pack.messages.length + 1); // +1: the note is new
  });

  it('keeps every system message, first', async () => {
    const extra = msg('system', 'policy');
    const messages = [SYSTEM, extra, ...chat().slice(1)];
    const pack = await packHandoff({ messages, ...BUDGET });
    expect(pack.messages.slice(0, 2)).toEqual([SYSTEM, extra]);
    expect(pack.messages.filter((m) => m.role === 'system')).toHaveLength(2);
  });

  it('keeps only LEADING system messages as the prefix; a mid-conversation one stays in place', async () => {
    const mid = msg('system', 'mid policy');
    // u1 a1 u2 a2 [mid] u3 a3 u4 a4 — the tail reaches back past `mid` at a large budget.
    const messages = [...chat().slice(0, 5), mid, ...chat().slice(5)];
    const wide = await packHandoff({ messages, targetContextTokens: 115, reserveOutputTokens: 0 });
    expectWithinBudget(wide, 115);
    expect(wide.messages[0]).toBe(SYSTEM);
    expect(wide.messages[1]!.role).not.toBe('system');
    const at = wide.messages.indexOf(mid);
    expect(at).toBeGreaterThan(1);
    expect(wide.messages[at + 1]!.content).toBe(msg('user', 'u3').content);

    // At a small budget `mid` falls in the dropped middle: dropped, not hoisted.
    const seen: HandoffMessage[][] = [];
    const tight = await packHandoff({ messages, ...BUDGET, summarize: async (d) => { seen.push([...d]); return 'S'; } });
    expectWithinBudget(tight);
    expect(tight.messages.includes(mid)).toBe(false);
    expect(tight.messages.filter((m) => m.role === 'system')).toEqual([SYSTEM]);
    expect(seen[0]).toContain(mid);
  });

  it('makes the note a user turn when the verbatim tail starts with an assistant turn', async () => {
    // Budget 80: the tail is a3 u4 a4, so an assistant note would sit next to a3.
    const pack = await packHandoff({ messages: chat(), ...BUDGET, summarize: async () => 'S' });
    expectWithinBudget(pack);
    expect(pack.messages[2]!.role).toBe('assistant');
    expect(pack.messages[1]).toEqual({ role: 'user', content: 'S' });
    expectAlternating(pack.messages);
  });

  it('keeps the note an assistant turn when the re-injected directive follows it', async () => {
    const directive = msg('user', 'fix the bug');
    const messages: HandoffMessage[] = [SYSTEM, msg('user', 'old task'), msg('assistant', 'ok'), directive];
    for (let i = 0; i < 10; i++) messages.push(msg('assistant', `step${i}`));
    const pack = await packHandoff({ messages, ...BUDGET });
    expect(pack.messages[1]!.role).toBe('assistant');
    expect(pack.messages[2]).toBe(directive);
  });

  it('prefixes the note with a hand-off line only when from and to differ', async () => {
    const handed = await packHandoff({ messages: chat(), ...BUDGET, from: 'cloud/claude', to: 'local/qwen', summarize: async () => 'S' });
    expectWithinBudget(handed);
    expect(noteOf(handed)).toBe('Handed off from cloud/claude to local/qwen.\nS');

    const same = await packHandoff({ messages: chat(), ...BUDGET, from: 'local/qwen', to: 'local/qwen', summarize: async () => 'S' });
    expect(noteOf(same)).toBe('S');
  });

  it('keeps a PINNED message (the task brief) verbatim after the system prefix when the middle is dropped', async () => {
    const brief = msg('user', 'the task brief');
    const nudge = msg('user', 'keep going');
    const messages: HandoffMessage[] = [SYSTEM, brief, msg('assistant', 'ok'), nudge];
    for (let i = 0; i < 10; i++) messages.push(msg('assistant', `step${i}`));
    const pinned = (_m: HandoffMessage, i: number) => i === 1;
    const pack = await packHandoff({ messages, ...BUDGET, pinned });
    expectWithinBudget(pack);
    expect(pack.messages[0]).toBe(SYSTEM);
    expect(pack.messages[1]).toBe(brief);
    expect(pack.messages.filter((m) => m === brief)).toHaveLength(1);
    expect(pack.messages).toContain(nudge);
    expect(pack.messages[pack.messages.length - 1]!.content).toBe(msg('assistant', 'step9').content);

    // Without the pin the brief is summarised away.
    const unpinned = await packHandoff({ messages, ...BUDGET });
    expect(unpinned.messages).not.toContain(brief);
  });

  it('never pins a tool-call turn (it would split from its results)', async () => {
    const call = { role: 'assistant' as const, content: '', tool_calls: [{ id: 'c1' }] };
    const messages: HandoffMessage[] = [SYSTEM, msg('user', 'task'), call, { role: 'tool', content: 'r'.repeat(40), tool_call_id: 'c1' }];
    for (let i = 0; i < 10; i++) messages.push(msg('assistant', `step${i}`));
    const pack = await packHandoff({ messages, ...BUDGET, pinned: (m) => m === call });
    expect(pack.messages[1]).not.toBe(call);
  });

  it('reports fits:false only when the system message + directive alone exceed the budget', async () => {
    const big = { role: 'system' as const, content: 'q'.repeat(400) }; // 104 tokens
    const directive = msg('user', 'do it');
    const over = await packHandoff({ messages: [big, msg('user', 'first'), msg('assistant', 'r'), directive], targetContextTokens: 110, reserveOutputTokens: 0 });
    expect(over.fits).toBe(false);
    expect(over.messages).toEqual([big, directive]);

    // Exactly system + directive: fits, everything else dropped, no room for a note.
    const exact = await packHandoff({ messages: [SYSTEM, msg('user', 'first'), msg('assistant', 'r'), directive], targetContextTokens: 28, reserveOutputTokens: 0 });
    expect(exact.fits).toBe(true);
    expect(exact.estimatedTokens).toBeLessThanOrEqual(28);
    expect(exact.messages).toEqual([SYSTEM, directive]);
    expect(exact.dropped).toBe(2);
  });
});
