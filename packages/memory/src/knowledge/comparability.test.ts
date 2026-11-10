/**
 * src/knowledge/comparability.test.ts — `comparable()` (PRD 41 §C9): each stable reason code.
 */

import { comparable, type ComparabilityMeta } from './comparability.js';

const base: ComparabilityMeta = {
    extractor: 'tree-sitter@0.25', extractorVersion: 3, contentHashRule: 'sha256-lf', tools: { jscpd: '4.0.5', knip: '5.1.0' },
    scope: '.', repo: 'r', branch: 'main', truncated: false, failed: [],
};
const reasonsOf = (after: Partial<ComparabilityMeta>, before: Partial<ComparabilityMeta> = {}) =>
    comparable({ ...base, ...before }, { ...base, ...after }).reasons;

describe('comparable', () => {
    it('two equal sides are comparable with no reasons', () => {
        expect(comparable(base, { ...base })).toEqual({ comparable: true, reasons: [] });
        expect(comparable({}, {})).toEqual({ comparable: true, reasons: [] });
    });

    it('names each reason with a stable code', () => {
        expect(reasonsOf({ extractor: 'regex@1' })).toEqual(['extractor-mismatch']);
        expect(reasonsOf({ extractor: 'tree-sitter@0.26' })).toEqual(['extractor-version-changed']);
        expect(reasonsOf({ extractorVersion: 4 })).toEqual(['extractor-version-changed']);
        expect(reasonsOf({ contentHashRule: 'sha256-raw' })).toEqual(['content-hash-rule-changed']);
        expect(reasonsOf({ tools: { jscpd: '4.0.6', knip: '5.1.0' } })).toEqual(['tool-version-changed:jscpd']);
        expect(reasonsOf({ tools: { jscpd: '4.0.5' } })).toEqual(['tool-missing:knip']);
        expect(reasonsOf({ failed: ['knip'] }, { failed: ['jscpd'] })).toEqual(['analyzer-failed:jscpd', 'analyzer-failed:knip']);
        expect(reasonsOf({ scope: 'packages/app' })).toEqual(['scope-changed']);
        expect(reasonsOf({ repo: 'other' })).toEqual(['repo-changed']);
        expect(reasonsOf({ branch: 'dev' })).toEqual(['branch-changed']);
        expect(reasonsOf({}, { truncated: true })).toEqual(['truncated']);
    });

    it('collects every reason, sorted and deduped, and is not comparable', () => {
        const r = comparable({ ...base, failed: ['knip'] }, { ...base, failed: ['knip'], scope: 'x', truncated: true });
        expect(r).toEqual({ comparable: false, reasons: ['analyzer-failed:knip', 'scope-changed', 'truncated'] });
    });
});
