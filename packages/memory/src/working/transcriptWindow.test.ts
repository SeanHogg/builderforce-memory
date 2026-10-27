/**
 * src/working/transcriptWindow.test.ts
 * Drop-oldest windowing drops in CHUNKS with hysteresis, so the front of the request —
 * the prefix a provider cache hashes — moves rarely instead of on every turn.
 */
import {
  windowed,
  HISTORY_WINDOW,
  HISTORY_WINDOW_DROP_CHUNK,
  HISTORY_TOKEN_BUDGET,
  HISTORY_TOKEN_DROP_CHUNK,
  transcriptTokens,
} from './transcript.js';
import type { TranscriptMessage } from './transcript.js';

const msg = (role: TranscriptMessage['role'], content = 'x'): TranscriptMessage => ({ role, content });

/** user/assistant pairs, `n` messages long, each message distinct. */
function convoOf(n: number, content = (i: number) => `m${i}`): TranscriptMessage[] {
  return Array.from({ length: n }, (_, i) => msg(i % 2 === 0 ? 'user' : 'assistant', content(i)));
}

describe('windowed — chunked message-count drops', () => {
  it('keeps everything up to the window', () => {
    const convo = convoOf(HISTORY_WINDOW);
    expect(windowed(convo)).toEqual(convo);
  });

  it('drops a whole chunk once over, then holds the front while messages arrive', () => {
    const fronts: unknown[] = [];
    for (let n = HISTORY_WINDOW + 1; n <= HISTORY_WINDOW + HISTORY_WINDOW_DROP_CHUNK; n++) {
      const w = windowed(convoOf(n));
      expect(w.length).toBeLessThanOrEqual(HISTORY_WINDOW);
      fronts.push(w[0]!.content);
    }
    // One front for the whole chunk: the prefix did not move turn to turn.
    expect(new Set(fronts).size).toBe(1);
    expect(fronts[0]).toBe(`m${HISTORY_WINDOW_DROP_CHUNK}`);
    // The first drop leaves room: ~75% of the window, not exactly the window.
    expect(windowed(convoOf(HISTORY_WINDOW + 1)).length).toBe(HISTORY_WINDOW + 1 - HISTORY_WINDOW_DROP_CHUNK);
  });

  it('moves the front by the next chunk only when the window fills again', () => {
    const w = windowed(convoOf(HISTORY_WINDOW + HISTORY_WINDOW_DROP_CHUNK + 1));
    expect(w[0]!.content).toBe(`m${2 * HISTORY_WINDOW_DROP_CHUNK}`);
  });
});

describe('windowed — chunked token drops', () => {
  // ~2k tokens per message (chars/4), so the budget holds ~32 of them.
  const big = (i: number) => `${i}:${'y'.repeat(8_000)}`;

  it('drops down to ~75% of the budget, not just under it', () => {
    let n = 2;
    while (transcriptTokens(convoOf(n, big)) <= HISTORY_TOKEN_BUDGET) n += 2;
    const w = windowed(convoOf(n, big));
    const kept = transcriptTokens(w);
    expect(kept).toBeLessThanOrEqual(HISTORY_TOKEN_BUDGET);
    expect(kept).toBeLessThanOrEqual(HISTORY_TOKEN_BUDGET - HISTORY_TOKEN_DROP_CHUNK + 4_100);
    expect(w[0]!.role).toBe('user');
  });

  it('holds the front across turns until a whole chunk of new work arrives', () => {
    let n = 2;
    while (transcriptTokens(convoOf(n, big)) <= HISTORY_TOKEN_BUDGET) n += 2;
    const first = windowed(convoOf(n, big))[0];
    // A few more turns (< one chunk of tokens) — same front.
    for (let extra = 2; extra <= 4; extra += 2) {
      expect(windowed(convoOf(n + extra, big))[0]!.content).toBe(first!.content);
    }
  });

  it('never drops the most recent user turn, however large', () => {
    const convo = [msg('user', 'old'), msg('assistant', 'a'), msg('user', 'z'.repeat(HISTORY_TOKEN_BUDGET * 5))];
    const w = windowed(convo);
    expect(w[0]!.content).toBe(convo[2]!.content);
  });

  it('still starts on a user turn after a drop (strict vendors)', () => {
    const convo: TranscriptMessage[] = [];
    for (let i = 0; i < 40; i++) convo.push(msg('user', `u${i}`), msg('assistant', big(i)), msg('tool', big(i)));
    expect(windowed(convo)[0]!.role).toBe('user');
  });
});
