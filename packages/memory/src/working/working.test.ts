/**
 * src/working/working.test.ts
 * Short-term working memory (the per-turn brief) and its consolidation into a long-term
 * code map, including the write-side merge a keyed store applies.
 */

import { consolidate, mergeConsolidated, WorkingMemory, type WorkingMemorySnapshot } from './index.js';

const AT = new Date('2026-10-24T12:00:00Z');

function snap(partial: Partial<WorkingMemorySnapshot>): WorkingMemorySnapshot {
  return { created: [], edited: [], deleted: [], read: [], searched: [], ...partial };
}

describe('WorkingMemory.brief', () => {
  it('stays silent while the run is still short enough to see its own results', () => {
    const m = new WorkingMemory();
    for (let i = 0; i < 9; i++) m.observe({ kind: 'read', path: `a/${i}.ts` });
    expect(m.brief()).toBeNull();
  });

  it('restates changes, reads with counts and searches once the run is long', () => {
    const m = new WorkingMemory();
    m.observe({ kind: 'searched', query: 'search_code "canvas"' });
    m.observe({ kind: 'searched', query: 'search_code "canvas"' });
    for (let i = 0; i < 6; i++) m.observe({ kind: 'read', path: 'web/src/App.tsx' });
    m.observe({ kind: 'edited', path: 'web/src/Surface.tsx' });
    m.observe({ kind: 'deleted', path: 'web/src/Old.tsx' });
    const brief = m.brief()!;
    expect(brief).toContain('## What this run has already established');
    expect(brief).toContain('Changed:\n- web/src/Surface.tsx');
    expect(brief).toContain('Deleted:\n- web/src/Old.tsx');
    expect(brief).toContain('- web/src/App.tsx (6×)');
    expect(brief).toContain('- search_code "canvas" (2×)');
    // Verification is not discouraged.
    expect(brief).toMatch(/re-running a search to verify an edit is fine/);
  });

  it('lets the host count its own activity toward the threshold', () => {
    const m = new WorkingMemory();
    m.observe({ kind: 'read', path: 'a.ts' });
    expect(m.brief({ observations: 12 })).toContain('- a.ts');
  });
});

describe('consolidate', () => {
  it('keys by where most of the work was, ranks changed then most-read', () => {
    const c = consolidate(
      snap({
        task: 'Move the   canvas',
        read: ['web/src/a/x.ts', 'web/src/a/x.ts', 'web/src/a/y.ts'],
        edited: ['web/src/a/z.ts', 'DONE.md'],
      }),
      { at: AT },
    )!;
    expect(c.key).toBe('codemap:web/src/a');
    expect(c.content).toContain('mapped 2026-10-24 by a run asked: "Move the canvas"');
    const entries = c.content.split('\n').filter((l) => l.startsWith('- '));
    expect(entries).toEqual(['- web/src/a/z.ts (changed)', '- DONE.md (changed)', '- web/src/a/x.ts (read 2×)', '- web/src/a/y.ts']);
  });

  it('resolves absolute, MSYS and dotted paths against the root and drops noise', () => {
    const c = consolidate(
      snap({
        read: [
          'C:\\code\\repo\\web\\src\\a.ts',
          '/c/code/repo/web/src/b.ts',
          'web/./src//c.ts',
          'web/node_modules/x.js',
          'web/dist/out.js',
          'pnpm-lock.yaml',
          '/elsewhere/d.ts',
          'scripts/build/gen.ts',
        ],
      }),
      { at: AT, root: 'C:\\code\\repo' },
    )!;
    expect(c.key).toBe('codemap:web/src');
    expect(c.content).toContain('- web/src/c.ts');
    expect(c.content).toContain('- scripts/build/gen.ts');
    expect(c.content).not.toMatch(/node_modules|dist|lock|elsewhere/);
  });

  it('skips a short lookup but keeps any change', () => {
    expect(consolidate(snap({ read: ['a/b/c.ts', 'a/b/d.ts'] }))).toBeNull();
    expect(consolidate(snap({ edited: ['api/src/r/x.ts'] }))?.key).toBe('codemap:api/src/r');
  });
});

describe('mergeConsolidated', () => {
  const prior = consolidate(snap({ read: ['w/s/a.ts', 'w/s/b.ts'], edited: ['w/s/c.ts'] }), { at: AT })!;

  it('adds a small run to the area map, restating the earlier run\'s edits', () => {
    const next = consolidate(snap({ task: 'fix d', edited: ['w/s/d.ts'] }), { at: AT })!;
    const merged = mergeConsolidated(next.key, prior.content, next.content);
    expect(merged).toContain('asked: "fix d"');
    const entries = merged.split('\n').filter((l) => l.startsWith('- '));
    expect(entries).toEqual(['- w/s/d.ts (changed)', '- w/s/c.ts (changed earlier)', '- w/s/a.ts', '- w/s/b.ts']);
  });

  it('removes a file this run deleted and never stores the deletion line', () => {
    const next = consolidate(snap({ deleted: ['w/s/a.ts'], edited: ['w/s/e.ts'] }), { at: AT })!;
    const merged = mergeConsolidated(next.key, prior.content, next.content);
    expect(merged).not.toContain('w/s/a.ts');
    expect(merged).not.toContain('(deleted)');
    expect(mergeConsolidated(next.key, null, next.content)).not.toContain('(deleted)');
  });

  it('caps the map and says how many more', () => {
    const big = consolidate(snap({ read: Array.from({ length: 12 }, (_, i) => `w/s/f${i}.ts`) }), { at: AT })!;
    const next = consolidate(snap({ edited: ['w/s/new.ts'] }), { at: AT })!;
    const merged = mergeConsolidated(next.key, big.content, next.content);
    expect(merged.split('\n').filter((l) => l.startsWith('- '))).toHaveLength(12);
    expect(merged).toContain('(+1 more file(s) touched)');
  });

  it('leaves every other key as sent', () => {
    expect(mergeConsolidated('learned:x', 'old', 'new')).toBe('new');
  });
});
