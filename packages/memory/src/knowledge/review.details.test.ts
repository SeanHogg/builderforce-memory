/**
 * src/knowledge/review.details.test.ts
 * A file found binary on read goes stale, a throwing judgement port leaves the file to be
 * retried, a contradiction names its sections, and `extractFile` is the one single-file extraction.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { hashText } from '../ingest/hash.js';
import { buildGraph } from './build.js';
import { extractFile } from './extractFile.js';
import type { NodeRef, ReviewArgs, StoredFact } from './model.js';
import { DOCS_PACK } from './packs/docs.js';
import { planReview } from './plan.js';
import { reviewDirectory } from './review.js';
import { EXTRACTOR_VERSION } from './sources.js';
import { extractedOf, fakeJudge, fileNode, memSource, memStore, ref, ROOT, type SpySource } from './testkit.test-util.js';

const args = (source: SpySource, store = memStore(), manifest = new InMemoryIngestManifest(), extra: Partial<ReviewArgs> = {}): ReviewArgs =>
    ({ source, store, manifest, packs: [DOCS_PACK], budget: { modelCalls: 200, ms: 60_000 }, ...extra });

describe('judgement and findings details', () => {
    it('a file found binary on read has its old relations marked stale', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n' });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        source.files['a.md'] = 'bin\u0000ary';
        await reviewDirectory(args(source, store, manifest));
        const about = (await store.relations({ rootId: ROOT })).filter((r) => r.kind === 'about');
        expect(about.length).toBeGreaterThan(0);
        expect(about.every((r) => r.staleAt)).toBe(true);
    });

    it('a judgement call that throws is unjudged, and the file is retried', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n' });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const broken = { async judge(): Promise<never> { throw new Error('port down'); } };
        const report = await reviewDirectory(args(source, store, manifest, { judgement: broken }));
        expect(report).toMatchObject({ judged: 0, unjudged: 1 });
        expect((await planReview(source, manifest, [DOCS_PACK])).changed).toEqual(['a.md']);

        const judge = fakeJudge();
        const again = await reviewDirectory(args(source, store, manifest, { judgement: judge }));
        expect(judge.requests.filter((r) => r.kind === 'extract')).toHaveLength(1);
        expect(again.judged).toBe(1);
    });

    it('a contradiction names the sections its claims are about', () => {
        const section = (path: string, slug: string, label: string) => ({
            ...fileNode(path), ref: `${ref(path)}#${slug}` as NodeRef, kind: 'section', label,
        });
        const claim = (key: string, path: string): StoredFact => ({
            key, content: key, source: 'review', importance: 0.5, origin: 'model', fileAnchors: [{ path, contentHash: `h:${path}` }],
        });
        const g = buildGraph({
            rootId: ROOT, packs: [DOCS_PACK],
            extracted: [
                { ...extractedOf(fileNode('a.md')), nodes: [fileNode('a.md'), section('a.md', 'rent', 'Rent')] },
                { ...extractedOf(fileNode('b.md')), nodes: [fileNode('b.md'), section('b.md', 'term', 'Term')] },
            ],
            facts: [claim('claim:a', 'a.md'), claim('claim:b', 'b.md')],
            relations: [
                { from: 'fact:claim:a', to: `${ref('a.md')}#rent` as NodeRef, kind: 'about', weight: 1, origin: 'model' },
                { from: 'fact:claim:b', to: `${ref('b.md')}#term` as NodeRef, kind: 'about', weight: 1, origin: 'model' },
                { from: 'fact:claim:a', to: 'fact:claim:b', kind: 'contradicts', weight: 1, origin: 'model', confidence: 0.8 },
            ],
        });
        const f = g.findings.find((x) => x.kind === 'contradiction');
        expect(f?.params).toMatchObject({ fileA: 'a.md', sectionA: 'Rent', fileB: 'b.md', sectionB: 'Term' });
    });
});

describe('extractFile', () => {
    it('extracts one file with the cache, the version and the stored entity keys', async () => {
        const store = memStore();
        const lease = { key: 'ent:docs:lease', content: 'Lease', source: 's', importance: 0.5, origin: 'model' as const, fileAnchors: [] };
        await store.write({ scope: { rootId: ROOT, title: 't', packs: [] }, facts: [lease], relations: [] });
        const text = '# A\nThe Lease ends.\n';
        const hash = hashText(text);
        const input = {
            rootId: ROOT, path: 'a.md', hash, packs: [DOCS_PACK], store,
            doc: { id: 'a.md', tenantId: 't', content: { kind: 'text' as const, text } },
        };
        const ex = await extractFile(input);
        expect(ex).toMatchObject({ sourceRef: ref('a.md'), sourceHash: hash, extractorVersion: EXTRACTOR_VERSION });
        expect(ex!.edges.some((e) => e.kind === 'mentions' && e.to === 'ent:docs:lease')).toBe(true);
        expect(await store.getExtracted(hash)).toEqual([ex]);
        expect(await extractFile(input)).toEqual(ex);
        const pdf = { ...input, path: 'a.pdf', doc: { ...input.doc, content: { kind: 'text' as const, text, mediaType: 'application/pdf' } } };
        expect(await extractFile(pdf)).toBeUndefined();
    });
});
