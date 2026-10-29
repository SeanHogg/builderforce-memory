/**
 * src/knowledge/review.gaps.test.ts
 * PRD 38 audit follow-ups: abort DURING judgement re-judges next time; stale relations and
 * facts on deleted sources show stale; the contradiction finding; nested `.gitignore`;
 * pdf/office only with a host extractor; files binary on read are not re-read; *Continue*
 * reuses the confirmed plan; wiki links by name, reference-style links, out-of-root links
 * and the pre-heading section.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { markdownParser } from '../ingest/parsers.js';
import type { SourceDocument } from '../ingest/types.js';
import { buildGraph } from './build.js';
import type { ExtractContext, KnowledgePack, KnowledgeParser, NodeRef, ReviewArgs, StoredFact } from './model.js';
import { DOCS_PACK } from './packs/docs.js';
import { planDetail, planReview } from './plan.js';
import { reviewDirectory } from './review.js';
import { asPathSource, resolveRelativePath } from './sources.js';
import { extractedOf, fakeJudge, fileNode, memSource, memStore, ref, ROOT, type SpySource } from './testkit.test-util.js';

const BUDGET = { modelCalls: 200, ms: 60_000 };
const args = (source: SpySource, store = memStore(), manifest = new InMemoryIngestManifest(), extra: Partial<ReviewArgs> = {}): ReviewArgs =>
    ({ source, store, manifest, packs: [DOCS_PACK], budget: BUDGET, ...extra });

describe('reviewDirectory — audit follow-ups', () => {
    it('an abort during judgement leaves that file to be judged again', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n## B\nbeta\n' });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const controller = new AbortController();
        let calls = 0;
        const judge = fakeJudge();
        const inner = judge.judge.bind(judge);
        judge.judge = async (req) => {
            calls += 1;
            const out = await inner(req);
            if (calls === 1) controller.abort();
            return out;
        };
        const first = await reviewDirectory(args(source, store, manifest, { judgement: judge, signal: controller.signal }));
        expect(first.aborted).toBe(true);
        expect((await store.facts({ rootId: ROOT })).filter((f) => f.key.startsWith('claim:'))).toHaveLength(0);

        const again = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: again }));
        expect(again.requests.filter((r) => r.kind === 'extract')).toHaveLength(2);
    });

    it('records a file found binary on read, so it is not re-read', async () => {
        const source = memSource({ 'a.txt': 'bin\u0000ary' });
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, memStore(), manifest));
        source.reads.length = 0;
        await reviewDirectory(args(source, memStore(), manifest));
        expect(source.reads).toEqual([]);
    });

    it('Continue reuses the confirmed plan without re-walking', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n' });
        const manifest = new InMemoryIngestManifest();
        const detail = await planDetail(source, manifest, [DOCS_PACK]);
        let walks = 0;
        const list = source.list.bind(source);
        source.list = () => { walks += 1; return list(); };
        const store = memStore();
        const report = await reviewDirectory(args(source, store, manifest, { plan: detail }));
        expect(walks).toBe(0);
        expect(report.extracted).toBe(2);

        // Run again with the same plan: what the first run finished is not reviewed (or read) again.
        source.reads.length = 0;
        const again = await reviewDirectory(args(source, store, manifest, { plan: detail }));
        expect(again.extracted).toBe(0);
        expect(source.reads).toEqual([]);
    });

    it('counts compare pairs skipped for budget as unjudged', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n' });
        const report = await reviewDirectory(args(source, memStore(), new InMemoryIngestManifest(), {
            judgement: fakeJudge(),
            budget: { modelCalls: 2, ms: 60_000 },
            similar: async (ids) => (ids.length >= 2 ? [{ a: ids[0]!, b: ids[1]!, similarity: 0.95 }] : []),
        }));
        expect(report.judged).toBe(2);
        expect(report.unjudged).toBe(1);
    });
});

describe('planReview — ignores and documents', () => {
    it('scopes a nested .gitignore to its folder', async () => {
        const source = memSource({ 'docs/.gitignore': 'draft.md\n', 'docs/draft.md': '# D\n', 'draft.md': '# Root\n' });
        const plan = await planReview(source, new InMemoryIngestManifest(), [DOCS_PACK]);
        expect(plan.skipped).toContainEqual({ path: 'docs/draft.md', reason: 'ignored' });
        expect(plan.changed).toContain('draft.md');
    });

    it('skips pdf/office as binary unless a host parser claims the media type', async () => {
        const source = memSource({ 'a.pdf': 'text', 'b.md': '# B\n' });
        expect((await planReview(source, new InMemoryIngestManifest(), [DOCS_PACK])).skipped)
            .toContainEqual({ path: 'a.pdf', reason: 'binary' });
        const pdf: KnowledgeParser = {
            id: 'pdf-text',
            accepts: (s) => s.content.kind === 'text' && s.content.mediaType === 'application/pdf',
            parse: (s) => [{ text: s.content.kind === 'text' ? s.content.text : '' }],
        };
        const withPdf: KnowledgePack = { ...DOCS_PACK, parsers: [pdf, ...DOCS_PACK.parsers] };
        const plan = await planReview(source, new InMemoryIngestManifest(), [withPdf]);
        expect(plan.changed).toContain('a.pdf');
    });
});

describe('buildGraph — stale knowledge', () => {
    const claim: StoredFact = {
        key: 'claim:x', content: 'The lease ends', source: 'review', importance: 0.5, origin: 'model',
        fileAnchors: [{ path: 'gone.md', contentHash: 'h:old' }],
    };

    it('marks a model fact anchored to a deleted source stale', () => {
        const g = buildGraph({ rootId: ROOT, packs: [DOCS_PACK], extracted: [extractedOf(fileNode('a.md'))], facts: [claim], relations: [] });
        expect(g.nodes.find((n) => n.ref === 'fact:claim:x')?.stale).toBe(true);
    });

    it('carries a stale relation as a stale edge', () => {
        const g = buildGraph({
            rootId: ROOT, packs: [DOCS_PACK], extracted: [extractedOf(fileNode('a.md')), extractedOf(fileNode('b.md'))], facts: [],
            relations: [{ from: ref('a.md'), to: ref('b.md'), kind: 'agrees', weight: 1, origin: 'model', staleAt: '2026-10-28' }],
        });
        expect(g.edges.find((e) => e.kind === 'agrees')?.stale).toBe(true);
        expect(g.stats.stale).toBe(0); // stale NODES only; the edge carries its own flag
    });

    it('reports a contradiction naming both sections', () => {
        const fact = (key: string, path: string): StoredFact => ({
            key, content: key, source: 'review', importance: 0.5, origin: 'model', fileAnchors: [{ path, contentHash: `h:${path}` }],
        });
        const g = buildGraph({
            rootId: ROOT, packs: [DOCS_PACK],
            extracted: [extractedOf(fileNode('a.md')), extractedOf(fileNode('b.md'))],
            facts: [fact('claim:a', 'a.md'), fact('claim:b', 'b.md')],
            relations: [{ from: 'fact:claim:a', to: 'fact:claim:b', kind: 'contradicts', weight: 1, origin: 'model', confidence: 0.8 }],
        });
        const f = g.findings.find((x) => x.kind === 'contradiction');
        expect(f?.params).toMatchObject({ fileA: 'a.md', fileB: 'b.md', confidence: 80 });
        expect(f?.requestKey).toBe('knowledgeGraph.fix.contradiction');
    });
});

describe('docs extract — links and sections', () => {
    const ctx: ExtractContext = { rootId: ROOT, knownKeys: new Map(), resolvePath: resolveRelativePath };
    const extract = (path: string, text: string) => {
        const src: SourceDocument = asPathSource({ id: path, tenantId: 't', content: { kind: 'text', text } }, path, `h:${path}`);
        return markdownParser.extract!(src, markdownParser.parse(src), ctx);
    };
    const links = (ex: ReturnType<typeof extract>) => ex.edges.filter((e) => e.kind === 'links').map((e) => e.to as NodeRef);

    it('extracts reference-style links and keeps out-of-root targets as broken', () => {
        const ex = extract('one.md', '# One\nSee [the guide][g] and [up](../outside.md).\n\n[g]: docs/guide.md\n');
        expect(links(ex).sort()).toEqual([ref('../outside.md'), ref('docs/guide.md')].sort());
    });

    it('makes text before the first heading a section, but not front matter alone', () => {
        const pre = extract('p.md', 'Intro before any heading.\n# H\nbody\n');
        expect(pre.nodes.some((n) => n.ref === `${ref('p.md')}#~preamble`)).toBe(true);
        const fm = extract('f.md', '---\ntitle: T\n---\n# H\nbody\n');
        expect(fm.nodes.some((n) => n.ref.endsWith('#body'))).toBe(false);
    });
});
