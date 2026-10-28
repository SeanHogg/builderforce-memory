/**
 * src/knowledge/stores/localSnapshotStore.test.ts
 * Upserts on the platform's unique keys (fact key; relation from+to+kind; scope rootId),
 * markStale scoped to one source, canonical bytes, and the extracted guard.
 */

import type { ScopeRecord, StoredFact, StoredRelation } from '../model.js';
import { LocalSnapshotStore } from './localSnapshotStore.js';
import { memPersistence } from '../testkit.test-util.js';

const SCOPE: ScopeRecord = { rootId: 'r', title: 'R', packs: ['docs'] };
const fact = (key: string, content: string): StoredFact => ({
    key, content, source: 's', importance: 0.5, origin: 'model', fileAnchors: [{ path: 'a.md', contentHash: 'h1' }],
});
const rel = (to: string, kind: string, extra: Partial<StoredRelation> = {}): StoredRelation => ({
    from: 'fact:claim:1', to: to as StoredRelation['to'], kind, weight: 1, origin: 'model', sourceHash: 'h1', ...extra,
});

const store = () => new LocalSnapshotStore(memPersistence(), { now: () => 'T' });

describe('LocalSnapshotStore', () => {
    it('upserts facts by key and relations by (from, to, kind)', async () => {
        const s = store();
        await s.write({ scope: SCOPE, facts: [fact('claim:1', 'v1'), fact('claim:2', 'x')], relations: [rel('src:r::a.md#a', 'about')] });
        await s.write({
            scope: SCOPE,
            facts: [fact('claim:1', 'v2')],
            relations: [rel('src:r::a.md#a', 'about', { weight: 3 }), rel('src:r::a.md#a', 'mentions')],
        });
        const facts = await s.facts({ rootId: 'r' });
        expect(facts.map((f) => [f.key, f.content])).toEqual(expect.arrayContaining([['claim:1', 'v2'], ['claim:2', 'x']]));
        expect(facts).toHaveLength(2);
        const rels = await s.relations({ rootId: 'r' });
        expect(rels).toHaveLength(2);
        expect(rels.find((r) => r.kind === 'about')?.weight).toBe(3);
        expect(await s.facts({ rootId: 'other' })).toEqual([]);
    });

    it('upserts the scope by rootId without dropping fields a later write omits', async () => {
        const s = store();
        await s.write({ scope: { ...SCOPE, lastDigest: 'd1', lastReviewedAt: 'T1' }, facts: [], relations: [] });
        await s.write({ scope: SCOPE, facts: [], relations: [] });
        expect(await s.scopes()).toEqual([{ ...SCOPE, lastDigest: 'd1', lastReviewedAt: 'T1' }]);
    });

    it('markStale touches only model relations on that source judged from another hash', async () => {
        const s = store();
        await s.write({
            scope: SCOPE, facts: [],
            relations: [
                rel('src:r::a.md#a', 'about'),
                rel('src:r::a.md#b', 'about', { sourceHash: 'h2' }),
                rel('src:r::b.md#c', 'about'),
                rel('src:r::a.md#d', 'about', { origin: 'human', sourceHash: undefined }),
            ],
        });
        await s.markStale({ rootId: 'r' }, 'src:r::a.md', 'h2');
        const byTo = new Map((await s.relations({ rootId: 'r' })).map((r) => [r.to, r.staleAt] as const));
        expect(byTo.get('src:r::a.md#a')).toBe('T');
        expect(byTo.get('src:r::a.md#b')).toBeUndefined();
        expect(byTo.get('src:r::b.md#c')).toBeUndefined();
        expect(byTo.get('src:r::a.md#d')).toBeUndefined();
    });

    it('persists canonically: the same content in any order is the same bytes, and reloads', async () => {
        const p1 = memPersistence(), p2 = memPersistence();
        const a = new LocalSnapshotStore(p1), b = new LocalSnapshotStore(p2);
        await a.write({ scope: SCOPE, facts: [fact('k1', '1'), fact('k2', '2')], relations: [] });
        await b.write({ scope: SCOPE, facts: [fact('k2', '2'), fact('k1', '1')], relations: [] });
        expect(p1.value).toBe(p2.value);
        expect(await new LocalSnapshotStore(p1).facts({ rootId: 'r' })).toHaveLength(2);
    });

    it('caches extracted output by hash and refuses extracted items as memory', async () => {
        const s = store();
        await s.putExtracted('h', [{ sourceRef: 'src:r::a.md', sourceHash: 'h', nodes: [], edges: [], keys: [] }]);
        expect(await s.getExtracted('h')).toHaveLength(1);
        expect(await s.getExtracted('nope')).toBeUndefined();
        const bad = { ...fact('k', 'c'), origin: 'extracted' } as unknown as StoredFact;
        await expect(s.write({ scope: SCOPE, facts: [bad], relations: [] })).rejects.toThrow();
    });
});
