/**
 * src/working/consolidationMarker.test.ts
 * Consolidation markers (moved unchanged from brain-embedded) + the auto-fold coverage rule.
 */
import {
  consolidationMetadata,
  consolidationMarkerContent,
  isConsolidationMarker,
  lastConsolidationIndex,
  scopeToConsolidation,
  CONSOLIDATION_MARKER_PREFIX,
  consolidationMarkerOf,
  foldKeepFromId,
} from './consolidationMarker.js';

const msg = (content: string, metadata: string | null = null) => ({ content, metadata });

describe('consolidation markers', () => {
  it('flags a message tagged in metadata', () => {
    expect(isConsolidationMarker(msg('x', consolidationMetadata()))).toBe(true);
    expect(isConsolidationMarker(msg('x', null))).toBe(false);
    expect(isConsolidationMarker(msg('x', '{"feedback":"up"}'))).toBe(false);
    expect(isConsolidationMarker(msg('x', 'not json'))).toBe(false);
  });

  it('finds the LAST marker index', () => {
    const list = [
      msg('a'),
      msg('sum1', consolidationMetadata()),
      msg('b'),
      msg('sum2', consolidationMetadata()),
      msg('c'),
    ];
    expect(lastConsolidationIndex(list)).toBe(3);
    expect(lastConsolidationIndex([msg('a'), msg('b')])).toBe(-1);
  });

  it('scopes the seed from the last marker (inclusive)', () => {
    const list = [msg('a'), msg('b'), msg('sum', consolidationMetadata()), msg('c')];
    expect(scopeToConsolidation(list).map((m) => m.content)).toEqual(['sum', 'c']);
  });

  it('returns the full list unchanged when there is no marker', () => {
    const list = [msg('a'), msg('b')];
    expect(scopeToConsolidation(list)).toBe(list);
  });

  it('wraps a summary with the visible marker prefix', () => {
    const content = consolidationMarkerContent('  hello  ');
    expect(content).toBe(`${CONSOLIDATION_MARKER_PREFIX}hello`);
  });
});

describe('auto consolidation markers (a persisted fold)', () => {
  const row = (id: number, content: string, metadata: string | null = null) => ({ id, content, metadata });

  it('keeps every message from keepFromId on, after the marker, and drops older notes', () => {
    const list = [
      row(1, 'old user'), row(2, 'old reply'),
      row(3, 'superseded note', consolidationMetadata({ keepFromId: 2 })),
      row(4, 'user turn'), row(5, 'narration'),
      row(6, 'note', consolidationMetadata({ keepFromId: 4, folded: 12 })),
      row(7, 'final reply'),
    ];
    expect(scopeToConsolidation(list).map((m) => m.content)).toEqual(['note', 'user turn', 'narration', 'final reply']);
  });

  it('ignores an auto marker whose keepFromId cannot be placed (falls back, never drops)', () => {
    const gone = [row(1, 'a'), row(2, 'note', consolidationMetadata({ keepFromId: 99 })), row(3, 'b')];
    expect(scopeToConsolidation(gone).map((m) => m.content)).toEqual(['a', 'b']);
    const after = [row(1, 'a'), row(2, 'note', consolidationMetadata({ keepFromId: 3 })), row(3, 'b')];
    expect(scopeToConsolidation(after).map((m) => m.content)).toEqual(['a', 'b']);
  });

  it('falls back to the previous marker when the newest auto marker is unplaceable', () => {
    const list = [
      row(1, 'old'), row(2, 'manual', consolidationMetadata()), row(3, 'kept'),
      row(4, 'auto', consolidationMetadata({ keepFromId: 99 })), row(5, 'new'),
    ];
    expect(scopeToConsolidation(list).map((m) => m.content)).toEqual(['manual', 'kept', 'new']);
  });

  it('round-trips the auto metadata', () => {
    expect(consolidationMarkerOf(row(1, 'x', consolidationMetadata({ keepFromId: 4, folded: 3 })))).toEqual({ consolidation: true, auto: true, keepFromId: 4, folded: 3 });
    expect(consolidationMarkerOf(row(1, 'x', consolidationMetadata()))).toEqual({ consolidation: true });
    expect(consolidationMarkerOf(row(1, 'x', null))).toBeNull();
  });
});

describe('foldKeepFromId', () => {
  const anchors = [{ index: 0, id: 10 }, { index: 1, id: 11 }, { index: 2, id: 12 }, { index: 7, id: 20 }];

  it('is exact when a persisted message sits at the cut', () => {
    expect(foldKeepFromId(anchors, 2)).toBe(12);
  });

  it('falls back to the nearest persisted message before the cut (covered twice, never lost)', () => {
    expect(foldKeepFromId(anchors, 5)).toBe(12);
    expect(foldKeepFromId(anchors, 30)).toBe(20);
  });

  it('is undefined with nothing persisted before the cut', () => {
    expect(foldKeepFromId([{ index: 4, id: 1 }], 2)).toBeUndefined();
  });
});
