/**
 * src/knowledge/comparability.analysis.test.ts — the `normalizeAnalysis` / `compareAnalysis` cases ported
 * from Builderforce.ai agent-tools `codeAnalysis.test.ts`, assertions unchanged (the parity oracle).
 * Only the imports changed: the analyzer list comes from agent-tools' `codeAnalysisTool.ts` there, so its
 * names/versions are restated below; the analyzer-RUNNING cases stay in agent-tools.
 */

import { compareAnalysis, normalizeAnalysis, type CodeAnalysisSnapshot } from './comparability.js';

const ANALYZERS = [{ name: 'dependency-cruiser', version: '16.10.4' }, { name: 'jscpd', version: '4.0.5' }, { name: 'knip', version: '5.62.0' }];

const dependency = { modules: [{ source: 'src/a.ts', dependencies: [{ resolved: 'src/b.ts', circular: true }, { resolved: 'node:fs', coreModule: true }] }, { source: 'src/b.ts', dependencies: [] }] };
const duplicates = { duplicates: [{ fragment: 'return input + 1;', firstFile: { name: 'src/a.ts', start: 2, end: 7 }, secondFile: { name: 'src/b.ts', start: 5, end: 10 } }] };
const unused = { files: ['src/dead.ts'], issues: [{ file: 'src/b.ts', exports: [{ name: 'unused', line: 20 }], dependencies: [{ name: 'abandoned-library' }], unlisted: [{ name: 'missing-library' }] }] };
const snapshot = (): CodeAnalysisSnapshot => ({ version: 1, id: 'scan', capturedAt: '2026-10-18T12:00:00.000Z', source: { branch: 'main', commit: 'abc', dirty: false }, tools: ANALYZERS.map((a) => ({ name: a.name, version: a.version, status: 'complete' })), ...normalizeAnalysis(dependency, duplicates, unused) });

describe('measured code analysis', () => {
  it('normalizes edges, clone locations, unused files, exports and dependencies separately', () => {
    const value = snapshot();
    expect(value.edges).toEqual([{ from: 'src/a.ts', to: 'src/b.ts', circular: true }]);
    expect(value.issues.map((i) => i.kind)).toEqual(['duplicate', 'unused-file', 'unused-export', 'unused-dependency']);
    expect(value.issues[0]?.locations[1]).toEqual({ path: 'src/b.ts', startLine: 5, endLine: 10 });
    expect(value.issues[0]?.snippet).toBe('return input + 1;');
  });
  it('retains clone identity when code shifts lines', () => {
    const shifted = structuredClone(duplicates); shifted.duplicates[0]!.firstFile.start = 80;
    expect(normalizeAnalysis(dependency, shifted, unused).issues[0]?.id).toBe(snapshot().issues[0]?.id);
  });
  it('measures resolved and unchanged findings rather than assuming every missing result is fixed', () => {
    const baseline = snapshot(); const current = snapshot(); current.issues = current.issues.slice(1);
    expect(compareAnalysis(current, baseline)).toMatchObject({ comparable: true, resolved: [baseline.issues[0]], unchanged: current.issues });
    current.tools[1]!.status = 'failed';
    expect(compareAnalysis(current, baseline)).toMatchObject({ comparable: false, resolved: [] });
    expect(compareAnalysis(current)).toMatchObject({ comparable: false, resolved: [] });
  });
});

describe('compareAnalysis delegates to comparable()', () => {
  it('names why two snapshots are not comparable', () => {
    const baseline = snapshot(); const current = snapshot();
    expect(compareAnalysis(current, baseline).reasons).toEqual([]);
    current.tools[1]!.status = 'failed';
    expect(compareAnalysis(current, baseline).reasons).toEqual(['analyzer-failed:jscpd']);
    expect(compareAnalysis(current).reasons).toEqual(['no-baseline']);
    const fewer = snapshot(); fewer.tools = fewer.tools.slice(0, 2);
    expect(compareAnalysis(fewer, snapshot()).reasons).toEqual(['tool-missing:knip', 'tools-incomplete']);
    const scoped = snapshot(); scoped.source.scope = 'packages/app';
    expect(compareAnalysis(scoped, snapshot()).reasons).toEqual(['scope-changed']);
  });
});
