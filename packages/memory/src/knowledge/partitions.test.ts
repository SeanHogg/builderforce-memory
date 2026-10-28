/**
 * src/knowledge/partitions.test.ts
 * The verdict table, glob matching, top-level-folder partitions and override validation.
 */

import { globToRegExp, loadPartitions, matchGlob, partitionOf, verdictOf } from './partitions.js';
import type { KnowledgePack, PartitionRegistry } from './model.js';
import { DOCS_PACK } from './packs/docs.js';

const LAYERS: PartitionRegistry = {
    partitions: [
        { id: 'presentation', order: 0, match: ['**/routes/**', '**/components/**'] },
        { id: 'application', order: 1, match: ['**/application/**'] },
        { id: 'domain', order: 2, match: ['**/domain/**'] },
        { id: 'infrastructure', order: 3, match: ['**/infrastructure/**', '**/db/**'] },
    ],
    allow: {
        presentation: ['application', 'domain'],
        application: ['domain'],
        domain: [],
        infrastructure: ['domain', 'application'],
    },
};

describe('verdictOf — the table', () => {
    it.each([
        ['presentation', 'presentation', 'same'],
        ['presentation', 'application', 'allowed'],
        ['presentation', 'domain', 'allowed'],
        ['infrastructure', 'application', 'allowed'],
        ['domain', 'presentation', 'upward'],
        ['application', 'presentation', 'upward'],
        ['infrastructure', 'presentation', 'upward'],
        ['presentation', 'infrastructure', 'skips'],
        ['application', 'infrastructure', 'skips'],
        ['domain', 'infrastructure', 'skips'],
    ])('%s → %s is %s', (from, to, verdict) => {
        expect(verdictOf(from, to, LAYERS)).toBe(verdict);
    });

    it('gives no verdict without allow, or with an unpartitioned end', () => {
        expect(verdictOf('presentation', 'domain', { partitions: LAYERS.partitions })).toBeUndefined();
        expect(verdictOf(undefined, 'domain', LAYERS)).toBeUndefined();
        expect(verdictOf('domain', undefined, LAYERS)).toBeUndefined();
    });
});

describe('globs and partitionOf', () => {
    it('matches ** across folders and * within one', () => {
        expect(matchGlob('api/src/routes/x.ts', '**/routes/**')).toBe(true);
        expect(matchGlob('routes/x.ts', '**/routes/**')).toBe(true);
        expect(matchGlob('api/src/router/x.ts', '**/routes/**')).toBe(false);
        expect(matchGlob('a/b.md', '*')).toBe(false);
        expect(matchGlob('b.md', '*')).toBe(true);
        expect(matchGlob('x.ts', '*.{ts,tsx}')).toBe(true);
        expect(globToRegExp('a.b').test('aXb')).toBe(false);
    });

    it('takes the first matching rule', () => {
        expect(partitionOf('api/db/routes/q.ts', LAYERS)).toBe('presentation');
        expect(partitionOf('api/db/q.ts', LAYERS)).toBe('infrastructure');
        expect(partitionOf('README.md', LAYERS)).toBeUndefined();
    });

    it('the docs pack partitions by top-level folder, with no verdicts', () => {
        const reg = loadPartitions(DOCS_PACK);
        expect(partitionOf('contracts/lease.md', reg)).toBe('contracts');
        expect(partitionOf('contracts/deep/x.md', reg)).toBe('contracts');
        expect(partitionOf('README.md', reg)).toBe('(root)');
        expect(verdictOf('contracts', 'policy', reg)).toBeUndefined();
    });
});

describe('loadPartitions', () => {
    const pack: KnowledgePack = { id: 'code', parsers: [], partitions: LAYERS };

    it('returns the pack default without an override', () => {
        expect(loadPartitions(pack)).toBe(LAYERS);
        expect(loadPartitions({ id: 'x', parsers: [] })).toEqual({ partitions: [] });
    });

    it('accepts a valid override', () => {
        const reg = loadPartitions(pack, { partitions: [{ id: 'ui', order: 0, match: ['ui/**'] }, { id: 'core', order: 1, match: ['core/**'] }], allow: { ui: ['core'] } });
        expect(reg.allow).toEqual({ ui: ['core'] });
        expect(partitionOf('ui/a.ts', reg)).toBe('ui');
    });

    it.each([
        ['not an object', 'nope'],
        ['no partitions', {}],
        ['bad order', { partitions: [{ id: 'a', order: 'x', match: ['a/**'] }] }],
        ['bad match', { partitions: [{ id: 'a', order: 0, match: [1] }] }],
        ['duplicate ids', { partitions: [{ id: 'a', order: 0, match: ['a'] }, { id: 'a', order: 1, match: ['b'] }] }],
        ['unknown allow target', { partitions: [{ id: 'a', order: 0, match: ['a'] }], allow: { a: ['b'] } }],
    ])('rejects an override with %s', (_why, override) => {
        expect(() => loadPartitions(pack, override)).toThrow(TypeError);
    });
});
