/**
 * src/working/codemap.test.ts
 * Code maps (2026.10.27): the 3-file floor for every run, the root area, reading a
 * stored map back, and which maps a run is handed.
 */

import {
  CONSOLIDATED_ROOT_AREA,
  areaOfPath,
  areasNamedIn,
  areasOfPathsIn,
  consolidate,
  consolidatedArea,
  consolidatedFactsFor,
  consolidatedFiles,
  mergeConsolidated,
  withoutConsolidatedFiles,
  type WorkingMemorySnapshot,
} from './index.js';

const AT = new Date('2026-10-26T12:00:00Z');

function snap(partial: Partial<WorkingMemorySnapshot>): WorkingMemorySnapshot {
  return { created: [], edited: [], deleted: [], read: [], searched: [], ...partial };
}

describe('consolidate — the minimum holds for every run', () => {
  it('skips a one- or two-file run that changed files', () => {
    expect(consolidate(snap({ edited: ['api/src/r/x.ts'] }))).toBeNull();
    expect(consolidate(snap({ edited: ['api/src/r/x.ts'], read: ['api/src/r/y.ts'] }))).toBeNull();
    expect(consolidate(snap({ created: ['api/src/r/x.ts'], deleted: ['api/src/r/old.ts'] }))).toBeNull();
  });

  it('counts changed and read files together', () => {
    const c = consolidate(snap({ edited: ['api/src/r/x.ts'], read: ['api/src/r/y.ts', 'api/src/r/z.ts'] }), { at: AT });
    expect(c?.key).toBe('codemap:api/src/r');
  });

  it('counts a file once however it was touched', () => {
    expect(consolidate(snap({ edited: ['a/b/x.ts'], read: ['a/b/x.ts', 'a/b/x.ts', 'a/b/y.ts'] }))).toBeNull();
  });

  it('does not count noise it drops', () => {
    expect(consolidate(snap({ edited: ['a/b/x.ts'], read: ['pnpm-lock.yaml', 'node_modules/p/i.js', 'a/b/y.ts'] }))).toBeNull();
  });
});

describe('consolidate — root files', () => {
  it('files a root-only run under the root area', () => {
    const c = consolidate(snap({ edited: ['DONE.md', 'ROADMAP.md'], read: ['package.json'] }), { at: AT })!;
    expect(c.key).toBe(`codemap:${CONSOLIDATED_ROOT_AREA}`);
  });

  it('maps a root file to the same area a reader resolves it to', () => {
    const c = consolidate(snap({ read: ['README.md', 'package.json', 'tsconfig.json'] }), { at: AT })!;
    expect(consolidatedArea(c.key)).toBe(areaOfPath('README.md'));
    expect(consolidatedFactsFor([c], [areaOfPath('package.json')])).toEqual([c]);
  });
});

describe('areaOfPath', () => {
  it('is a file\'s directory, a directory itself, or the root area', () => {
    expect(areaOfPath('web/src/a.ts')).toBe('web/src');
    expect(areaOfPath('.\\web\\src\\a.ts')).toBe('web/src');
    expect(areaOfPath('clients/vscode/src')).toBe('clients/vscode/src');
    expect(areaOfPath('DONE.md')).toBe(CONSOLIDATED_ROOT_AREA);
    expect(areaOfPath('  ')).toBe('');
  });
});

describe('areasOfPathsIn', () => {
  it('takes the areas of the repo-relative paths a text names', () => {
    expect(areasOfPathsIn('Touches `brain-embedded/src/runStore/dispatchTool.ts` and C:/abs/x.ts, see (docs/specs/35.md).')).toEqual([
      'brain-embedded/src/runStore',
      'docs/specs',
    ]);
  });

  it('maps a root file to the root area and skips what is not a repo path', () => {
    expect(areasOfPathsIn('bump package.json, e.g. now')).toEqual([CONSOLIDATED_ROOT_AREA]);
    expect(areasOfPathsIn('see https://x.dev/a/b.ts and /etc/hosts and ../up/x.ts')).toEqual([]);
  });
});

describe('reading a stored map back', () => {
  const map = consolidate(snap({ read: ['w/s/a.ts', 'w/s/b.ts'], edited: ['w/s/c.ts'] }), { at: AT })!;

  it('lists the files a map names, never a deletion', () => {
    expect(consolidatedFiles(map.content)).toEqual(['w/s/c.ts', 'w/s/a.ts', 'w/s/b.ts']);
    expect(consolidatedFiles('Code map\n- x/y.ts (deleted)\n- x/z.ts')).toEqual(['x/z.ts']);
  });

  it('drops a changed file from the map and keeps the rest', () => {
    const out = withoutConsolidatedFiles(map.content, ['W\\S\\A.ts'])!;
    expect(out).not.toContain('w/s/a.ts');
    expect(out).toContain('- w/s/c.ts (changed)');
    expect(out).toContain('- w/s/b.ts');
    expect(out.split('\n')[0]).toBe(map.content.split('\n')[0]);
  });

  it('returns null once no file is left', () => {
    expect(withoutConsolidatedFiles(map.content, ['w/s/a.ts', 'w/s/b.ts', 'w/s/c.ts'])).toBeNull();
  });

  it('leaves the map as it was when nothing named is in it', () => {
    expect(withoutConsolidatedFiles(map.content, ['other.ts'])).toBe(map.content);
  });

  it('reads a merged map the same way', () => {
    const next = consolidate(snap({ edited: ['w/s/d.ts', 'w/s/e.ts', 'w/s/f.ts'] }), { at: AT })!;
    expect(consolidatedFiles(mergeConsolidated(next.key, map.content, next.content))).toEqual([
      'w/s/d.ts', 'w/s/e.ts', 'w/s/f.ts', 'w/s/c.ts', 'w/s/a.ts', 'w/s/b.ts',
    ]);
  });
});

describe('which maps a run is handed', () => {
  const facts = [
    { key: 'codemap:brain-embedded/src', content: 'wide' },
    { key: 'learned:x', content: 'not a map' },
    { key: 'codemap:brain-embedded/src/runStore', content: 'narrow' },
    { key: 'codemap:api/src', content: 'elsewhere' },
    { key: 'codemap:(root)', content: 'root' },
  ];

  it('keeps overlapping maps only, most specific first, capped', () => {
    expect(consolidatedFactsFor(facts, ['brain-embedded/src/runStore']).map((f) => f.content)).toEqual(['narrow', 'wide']);
    expect(consolidatedFactsFor(facts, ['brain-embedded/src/runStore'], 1).map((f) => f.content)).toEqual(['narrow']);
    expect(consolidatedFactsFor(facts, ['(root)']).map((f) => f.content)).toEqual(['root']);
    expect(consolidatedFactsFor(facts, [])).toEqual([]);
  });

  it('finds the areas a request names by word', () => {
    const areas = facts.map((f) => consolidatedArea(f.key)).filter((a): a is string => !!a);
    expect(areasNamedIn('the run store drops a steer', areas)).toEqual(['brain-embedded/src/runStore']);
    expect(areasNamedIn('Why does runStore retry?', areas)).toEqual(['brain-embedded/src/runStore']);
    expect(areasNamedIn('brain-embedded build is slow', areas)).toEqual(['brain-embedded/src/runStore', 'brain-embedded/src']);
  });

  it('never matches on a generic or short directory name', () => {
    expect(areasNamedIn('fix the src api', ['api/src', 'web/src'])).toEqual([]);
    expect(areasNamedIn('', ['api/src'])).toEqual([]);
  });
});
