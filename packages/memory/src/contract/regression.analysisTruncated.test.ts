/**
 * src/contract/regression.analysisTruncated.test.ts — a truncated code analysis is never comparable (PRD 41 B2, C9).
 * Before `CodeAnalysisSnapshot.truncated` reached `analysisMeta`, a host had to wrap `compareAnalysis` to gate it; without
 * the wrapper, an issue missing from a partial scan was reported as resolved.
 */

import { compareAnalysis, normalizeAnalysis, isBinaryBytes, type CodeAnalysisSnapshot } from '../knowledge/index.js';

const TOOLS = [{ name: 'dependency-cruiser', version: '16.10.4' }, { name: 'jscpd', version: '4.0.5' }, { name: 'knip', version: '5.62.0' }];

const snapshot = (files: string[], truncated?: true): CodeAnalysisSnapshot => ({
    version: 1, id: 'scan', capturedAt: '2026-11-10T12:00:00.000Z', source: { branch: 'main', commit: 'abc', dirty: false },
    tools: TOOLS.map((t) => ({ ...t, status: 'complete' as const })),
    ...normalizeAnalysis({ modules: [] }, { duplicates: [] }, { files, issues: [] }),
    ...(truncated ? { truncated } : {}),
});

describe('regression: truncated code analysis', () => {
    it('regression: a truncated after-side against a full baseline reported the files it never reached as resolved', () => {
        const result = compareAnalysis(snapshot(['a.ts'], true), snapshot(['a.ts', 'b.ts']));
        expect(result).toMatchObject({ comparable: false, reasons: ['truncated'], added: [], resolved: [] });
        expect(result.unchanged.map((i) => i.id)).toEqual(['unused-file:a.ts']);
    });

    it('regression: the binary rule hosts restated (a NUL in the first 8192 bytes) is exported as isBinaryBytes', () => {
        const edge = new Uint8Array(9000).fill(0x61);
        edge[8191] = 0;
        expect(isBinaryBytes(edge)).toBe(true);
        edge[8191] = 0x61; edge[8192] = 0;
        expect(isBinaryBytes(edge)).toBe(false);
    });
});
