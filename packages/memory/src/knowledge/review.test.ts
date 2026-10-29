/**
 * src/knowledge/review.test.ts
 * planReview / reviewDirectory with fake ports: unchanged files are never read, extracted
 * output is never written as memory, a changed hash marks only that source stale, the
 * budget stops judgement (and counts the rest), invalid judgement is dropped and counted,
 * and abort keeps what was written consistent per file.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { hashText } from '../ingest/hash.js';
import type { JudgementRequest, KnowledgeStore, ReviewArgs, StoredFact, StoredRelation } from './model.js';
import { DOCS_PACK } from './packs/docs.js';
import { planReview } from './plan.js';
import { assertStoredBatch, reviewDirectory } from './review.js';
import { fakeJudge, graphOf, memSource, memStore, ROOT, type SpySource } from './testkit.test-util.js';

const BUDGET = { modelCalls: 200, ms: 60_000 };

function args(source: SpySource, store: KnowledgeStore, manifest: InMemoryIngestManifest, extra: Partial<ReviewArgs> = {}): ReviewArgs {
    return { source, store, manifest, packs: [DOCS_PACK], budget: BUDGET, ...extra };
}

const anchoredTo = (facts: StoredFact[], path: string) => facts.filter((f) => f.fileAnchors.some((a) => a.path === path));

describe('planReview', () => {
    it('reads only the ignore files — never an unchanged file — and reports skips', async () => {
        const source = memSource({
            '.gitignore': 'skip/\n', 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n', 'img.png': 'x', 'skip/x.md': '# X\n',
        });
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, memStore(), manifest));

        source.files['a.md'] = '# A\nalpha changed\n';
        source.reads.length = 0;
        const plan = await planReview(source, manifest, [DOCS_PACK]);
        expect(source.reads).toEqual(['.gitignore']);
        expect(plan.changed).toEqual(['a.md']);
        expect(plan.unchanged).toBe(1);
        expect(plan.byPack).toEqual({ docs: 2 });
        expect(plan.estimatedModelCalls).toBeGreaterThan(0);
        expect(plan.skipped).toEqual(expect.arrayContaining([
            { path: '.gitignore', reason: 'ignored' }, { path: 'img.png', reason: 'binary' }, { path: 'skip/x.md', reason: 'ignored' },
        ]));

        source.reads.length = 0;
        await reviewDirectory(args(source, memStore(), manifest));
        expect(source.reads).toEqual(['.gitignore', 'a.md']);
    });

    it('enforces the size cap and lists removed files', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n', 'big.md': 'x'.repeat(50) });
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, memStore(), manifest));
        delete source.files['a.md'];
        const plan = await planReview(source, manifest, [DOCS_PACK], { maxFileBytes: 10 });
        expect(plan.removed).toEqual(['a.md']);
        expect(plan.skipped).toContainEqual({ path: 'big.md', reason: 'too-large' });
    });
});

describe('reviewDirectory', () => {
    it('never writes extracted output as memory; caches it by content hash instead', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n' });
        const store = memStore();
        const batches: Array<{ facts: StoredFact[]; relations: StoredRelation[] }> = [];
        const write = store.write.bind(store);
        store.write = (b) => { batches.push(b); return write(b); };
        const judge = fakeJudge();
        const report = await reviewDirectory(args(source, store, new InMemoryIngestManifest(), {
            judgement: judge,
            similar: async (ids) => (ids.length >= 2 ? [{ a: ids[0]!, b: ids[1]!, similarity: 0.9 }] : []),
        }));

        const written = batches.flatMap((b) => [...b.facts, ...b.relations]);
        expect(written.length).toBeGreaterThan(0);
        expect(written.every((x) => x.origin === 'model')).toBe(true);
        expect(report.extracted).toBe(2);
        expect(report.judged).toBe(3); // two sections + one compare
        expect(report.factsWritten).toBeGreaterThan(0);
        expect(await store.getExtracted(hashText('# A\nalpha\n'))).toHaveLength(1);
        expect(await store.relations({ rootId: ROOT })).toContainEqual(expect.objectContaining({ kind: 'contradicts', confidence: 0.9 }));
        expect((await store.scopes())[0]).toMatchObject({ rootId: ROOT, lastDigest: report.digest, packs: ['docs'] });
    });

    it('the store guard rejects extracted items', async () => {
        const bad = { from: 'src:r::a', to: 'src:r::b', kind: 'links', weight: 1, origin: 'extracted' } as unknown as StoredRelation;
        expect(() => assertStoredBatch({ facts: [], relations: [bad] })).toThrow();
        await expect(memStore().write({ scope: { rootId: ROOT, title: 't', packs: [] }, facts: [], relations: [bad] })).rejects.toThrow();
    });

    it('a changed hash marks only that source stale; unchanged sections are re-anchored, not re-judged', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n# A2\nsame\n', 'b.md': '# B\nbeta\n' });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));

        source.files['a.md'] = '# A\nalpha two\n# A2\nsame\n';
        await reviewDirectory(args(source, store, manifest)); // extracted-only: no model
        let g = await graphOf(source, store);
        const facts = await store.facts({ rootId: ROOT });
        const claimsA = anchoredTo(facts, 'a.md').filter((f) => f.key.startsWith('claim:'));
        const claimsB = anchoredTo(facts, 'b.md').filter((f) => f.key.startsWith('claim:'));
        expect(claimsA).toHaveLength(2);
        for (const c of claimsA) expect(g.nodes.find((n) => n.ref === `fact:${c.key}`)?.stale).toBe(true);
        for (const c of claimsB) expect(g.nodes.find((n) => n.ref === `fact:${c.key}`)?.stale).toBeUndefined();
        const rels = await store.relations({ rootId: ROOT });
        expect(rels.filter((r) => r.to.startsWith(`src:${ROOT}::a.md#`) && r.kind === 'about').every((r) => r.staleAt)).toBe(true);
        expect(rels.filter((r) => r.to.startsWith(`src:${ROOT}::b.md#`)).every((r) => !r.staleAt)).toBe(true);

        // Re-judging: only the changed section costs a call; the unchanged one is fresh again.
        source.files['a.md'] = '# A\nalpha three\n# A2\nsame\n';
        const judge = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: judge }));
        expect(judge.requests.filter((r) => r.kind === 'extract').map((r) => r.items[0]!.text)).toEqual(['alpha three']);
        g = await graphOf(source, store);
        const same = (await store.facts({ rootId: ROOT })).find((f) => f.key.startsWith('claim:') && f.content.startsWith('Lease ends same'))!;
        expect(g.nodes.find((n) => n.ref === `fact:${same.key}`)?.stale).toBeUndefined();
    });

    it('the budget stops judgement; the rest is counted unjudged and stays changed', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n# B\nbeta\n', 'b.md': '# C\ngamma\n' });
        const manifest = new InMemoryIngestManifest();
        const judge = fakeJudge();
        const report = await reviewDirectory(args(source, memStore(), manifest, { judgement: judge, budget: { modelCalls: 1, ms: 60_000 } }));
        expect(judge.requests).toHaveLength(1);
        expect(report.judged).toBe(1);
        expect(report.unjudged).toBe(2);
        expect(report.extracted).toBe(2);
        expect((await planReview(source, manifest, [DOCS_PACK])).changed).toEqual(['a.md', 'b.md']);
    });

    it('drops invalid judgement output and counts it', async () => {
        const judge = fakeJudge((req: JudgementRequest) => ({
            extract: [{
                ref: req.items[0]!.ref,
                claims: [
                    { subject: 'Lease', predicate: 'ends', object: '2027', quote: 'ends 2027' },
                    { subject: 'Lease', object: 'x', quote: '' } as never,
                ],
                entities: [{ name: '', type: 'thing' }, { name: 'Lease', type: 'contract' }],
            }],
            invalid: 1,
        }));
        const store = memStore();
        const report = await reviewDirectory(args(memSource({ 'a.md': '# A\nalpha\n' }), store, new InMemoryIngestManifest(), { judgement: judge }));
        expect(report.invalidJudgements).toBe(3);
        const facts = await store.facts({ rootId: ROOT });
        expect(facts.filter((f) => f.key.startsWith('claim:'))).toHaveLength(1);
        expect(facts.map((f) => f.key)).toContain('ent:docs:lease');
    });

    it('abort between files keeps the written file whole and stops before the next', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n' });
        const store = memStore();
        const controller = new AbortController();
        const report = await reviewDirectory(args(source, store, new InMemoryIngestManifest(), {
            judgement: fakeJudge(),
            signal: controller.signal,
            onProgress: (p) => { if (p.phase === 'write') controller.abort(); },
        }));
        expect(report.aborted).toBe(true);
        const facts = await store.facts({ rootId: ROOT });
        expect(anchoredTo(facts, 'a.md').length).toBeGreaterThan(0);
        expect(anchoredTo(facts, 'b.md')).toHaveLength(0);
        expect(source.reads).not.toContain('b.md');
    });
});
