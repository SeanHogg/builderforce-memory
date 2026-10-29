/**
 * src/knowledge/review.fixes.test.ts
 * Review follow-ups: binary documents are claimed only by a host extractor (any pack order);
 * `[[Name]]` is resolved at build time, never in the cached extract; a fact outlives one of
 * its deleted sources; ignore precedence; out-of-root and root links; compare pairs survive
 * a skipped run; extractor versions; a stale plan.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { hashText } from '../ingest/hash.js';
import { htmlParser, isBinaryDocumentType, markdownParser, plainTextParser } from '../ingest/parsers.js';
import type { SourceDocument } from '../ingest/types.js';
import { buildGraph } from './build.js';
import type { ExtractContext, Extracted, KnowledgePack, KnowledgeParser, ReviewArgs } from './model.js';
import { DOCS_PACK } from './packs/docs.js';
import { planDetail, planReview } from './plan.js';
import { reviewDirectory } from './review.js';
import {
    asPathSource, compareManifestKey, composePacks, EXTRACTOR_VERSION, filesManifestKey, manifestMap, resolveRelativePath,
} from './sources.js';
import {
    extractedOf, fakeJudge, fileNode, graphOf, memSource, memStore, ref, ROOT, type SpySource,
} from './testkit.test-util.js';

const BUDGET = { modelCalls: 200, ms: 60_000 };
const args = (source: SpySource, store = memStore(), manifest = new InMemoryIngestManifest(), extra: Partial<ReviewArgs> = {}): ReviewArgs =>
    ({ source, store, manifest, packs: [DOCS_PACK], budget: BUDGET, ...extra });

const ctx: ExtractContext = { rootId: ROOT, knownKeys: new Map(), resolvePath: resolveRelativePath };
const doc = (path: string, text: string): SourceDocument =>
    asPathSource({ id: path, tenantId: 't', content: { kind: 'text', text } }, path, `h:${path}`);
const extractMd = (path: string, text: string): Extracted => {
    const src = doc(path, text);
    return markdownParser.extract!(src, markdownParser.parse(src), ctx);
};
const links = (ex: Extracted) => ex.edges.filter((e) => e.kind === 'links');

const pdfParser: KnowledgeParser = {
    id: 'pdf-text',
    accepts: (s) => s.content.kind === 'text' && s.content.mediaType === 'application/pdf',
    parse: (s) => [{ text: s.content.kind === 'text' ? s.content.text : '' }],
};
const pdfPack: KnowledgePack = { id: 'pdf', parsers: [pdfParser] };

describe('binary documents', () => {
    it('knowledge composition refuses an office or pdf media type to the builtin text parsers; ingest is unchanged', () => {
        const docx = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
        for (const type of [docx, 'application/pdf', 'application/msword', 'application/vnd.ms-excel']) {
            expect(isBinaryDocumentType(type)).toBe(true);
            const source: SourceDocument = { id: 'x', tenantId: 't', content: { kind: 'text', text: 't', mediaType: type } };
            expect(composePacks([DOCS_PACK]).resolve(source)).toBeUndefined();
            expect(composePacks([DOCS_PACK, pdfPack]).resolve(source)?.pack.id).toBe(type === 'application/pdf' ? 'pdf' : undefined);
            // The ingest pipeline's builtins accept exactly what they always did.
            expect(plainTextParser.accepts(source)).toBe(true);
        }
        const xml: SourceDocument = { id: 'x', tenantId: 't', content: { kind: 'text', text: 't', mediaType: docx } };
        expect(htmlParser.accepts(xml)).toBe(true);
        expect(isBinaryDocumentType('application/vnd.api+json')).toBe(false);
    });

    it('a docx with no host parser is skipped as binary', async () => {
        const plan = await planReview(memSource({ 'a.docx': 'x', 'b.md': '# B\n' }), new InMemoryIngestManifest(), [DOCS_PACK]);
        expect(plan.skipped).toContainEqual({ path: 'a.docx', reason: 'binary' });
    });

    it('a host pdf pack placed AFTER the docs pack still claims the pdf', async () => {
        const source = memSource({ 'a.pdf': 'text' });
        const plan = await planReview(source, new InMemoryIngestManifest(), [DOCS_PACK, pdfPack]);
        expect(plan.byPack).toEqual({ pdf: 1 });
        const report = await reviewDirectory(args(source, memStore(), new InMemoryIngestManifest(), { packs: [DOCS_PACK, pdfPack] }));
        expect(report.extracted).toBe(1);
    });
});

describe('[[Name]] links resolve at build time', () => {
    const one = extractMd('a/one.md', '# One\nSee [[Two]].\n');
    const build = (...paths: string[]) => buildGraph({
        rootId: ROOT, packs: [DOCS_PACK], extracted: [one, ...paths.map((p) => extractedOf(fileNode(p)))], facts: [], relations: [],
    });
    const targets = (g: ReturnType<typeof build>) => g.edges.filter((e) => e.kind === 'links').map((e) => e.to);

    it('the cached extract carries the fallback and the name, not the resolution', () => {
        expect(links(one)).toEqual([expect.objectContaining({ to: ref('a/Two.md'), byName: 'Two' })]);
    });

    it('points at the unique file of that name, even when it appeared after the extract', () => {
        const g = build('b/Two.md');
        expect(targets(g)).toEqual([ref('b/Two.md')]);
        expect(g.edges.every((e) => !('byName' in e))).toBe(true);
    });

    it('keeps the fallback when the name is ambiguous or absent', () => {
        expect(targets(build('b/Two.md', 'c/Two.md'))).toEqual([ref('a/Two.md')]);
        expect(targets(build())).toEqual([ref('a/Two.md')]);
    });
});

describe('a fact outlives one of its deleted sources', () => {
    it('drops the removed anchor from a fact other sources back; a fact anchored only there stays stale', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n' });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        const lease = (await store.facts({ rootId: ROOT })).find((f) => f.key === 'ent:docs:lease')!;
        expect(lease.fileAnchors.map((a) => a.path)).toEqual(['a.md', 'b.md']);

        delete source.files['b.md'];
        await reviewDirectory(args(source, store, manifest));
        const facts = await store.facts({ rootId: ROOT });
        expect(facts.find((f) => f.key === 'ent:docs:lease')!.fileAnchors.map((a) => a.path)).toEqual(['a.md']);
        const g = await graphOf(source, store);
        expect(g.nodes.find((n) => n.ref === 'ent:docs:lease')?.stale).toBeUndefined();
        const onlyB = facts.find((f) => f.key.startsWith('claim:') && f.fileAnchors.every((a) => a.path === 'b.md'))!;
        expect(g.nodes.find((n) => n.ref === `fact:${onlyB.key}`)?.stale).toBe(true);
    });
});

describe('ignore precedence', () => {
    it('.builderforce/ignore overrides .gitignore', async () => {
        const source = memSource({
            '.gitignore': 'docs/generated/\n', '.builderforce/ignore': '!docs/generated/\n', 'docs/generated/x.md': '# X\n',
        });
        const plan = await planReview(source, new InMemoryIngestManifest(), [DOCS_PACK]);
        expect(plan.changed).toEqual(['docs/generated/x.md']);
    });
});

describe('links out of and to the root', () => {
    it('a link to the root itself is no link', () => {
        expect(links(extractMd('a/b.md', '# T\n[x](/) [y](./) [z](.) [w](..) [v](../) [d](docs/)\n'))).toEqual([]);
    });

    it('an escape stays a broken link under a normalized path that never collapses different files', () => {
        const refs = [
            links(extractMd('a/x.md', '# T\n[u](../../y.md)\n')),
            links(extractMd('a/b/x.md', '# T\n[u](../../../y.md)\n')),
            links(extractMd('x.md', '# T\n[u](../../y.md)\n')),
        ].map((l) => l.map((e) => e.to));
        expect(refs).toEqual([[ref('../y.md')], [ref('../y.md')], [ref('../../y.md')]]);

        const g = buildGraph({ rootId: ROOT, packs: [DOCS_PACK], extracted: [extractMd('a/x.md', '# T\n[u](../../y.md)\n')], facts: [], relations: [] });
        const missing = g.nodes.find((n) => n.ref === ref('../y.md'));
        expect(missing).toMatchObject({ kind: 'missing' });
        expect(missing?.partition).toBeUndefined();
    });

    it('a reference definition may end in whitespace', () => {
        for (const eol of ['  \n', '\t\r\n']) {
            const ex = extractMd('one.md', `# One\nSee [the guide][g].\n\n[g]: docs/guide.md${eol}`);
            expect(links(ex).map((e) => e.to)).toEqual([ref('docs/guide.md')]);
        }
    });
});

describe('compare pairs skipped can continue', () => {
    const similar = async (ids: string[]) => (ids.length >= 2 ? [{ a: ids[0]!, b: ids[1]!, similarity: 0.95 }] : []);

    it('a pair skipped for budget is compared by the next run, once', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n' });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const first = await reviewDirectory(args(source, store, manifest, {
            judgement: fakeJudge(), similar, budget: { modelCalls: 2, ms: 60_000 },
        }));
        expect(first.unjudged).toBe(1);
        expect((await manifestMap(manifest, compareManifestKey(ROOT))).size).toBe(2);

        const judge = fakeJudge();
        const second = await reviewDirectory(args(source, store, manifest, { judgement: judge, similar }));
        expect(judge.requests.map((r) => r.kind)).toEqual(['compare']);
        expect(second.unjudged).toBe(0);
        expect(await store.relations({ rootId: ROOT })).toContainEqual(expect.objectContaining({ kind: 'contradicts' }));
        expect((await manifestMap(manifest, compareManifestKey(ROOT))).size).toBe(0);

        const third = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: third, similar }));
        expect(third.requests).toHaveLength(0);
    });

    it('aborted is set only when the abort landed before compare finished', async () => {
        const run = async (files: Record<string, string>, pairs: number) => {
            const controller = new AbortController();
            const judge = fakeJudge();
            const inner = judge.judge.bind(judge);
            judge.judge = async (req) => {
                const out = await inner(req);
                if (req.kind === 'compare') controller.abort();
                return out;
            };
            const manifest = new InMemoryIngestManifest();
            const report = await reviewDirectory(args(memSource(files), memStore(), manifest, {
                judgement: judge, signal: controller.signal,
                similar: async (ids) => ids.slice(1).slice(0, pairs).map((b, i) => ({ a: ids[i]!, b, similarity: 0.9 })),
            }));
            return { report, awaiting: (await manifestMap(manifest, compareManifestKey(ROOT))).size };
        };
        const last = await run({ 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n' }, 1);
        expect(last.report.aborted).toBe(false);
        expect(last.awaiting).toBe(0);
        const mid = await run({ 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n', 'c.md': '# C\ngamma\n' }, 2);
        expect(mid.report.aborted).toBe(true);
        expect(mid.awaiting).toBeGreaterThan(0);
    });
});

describe('extractor version and stale plans', () => {
    it('a cached Extracted of an older version is re-extracted', async () => {
        const text = '# A\nalpha\n';
        const store = memStore();
        const hash = hashText(text);
        await store.putExtracted(hash, [{ sourceRef: ref('a.md'), sourceHash: hash, extractorVersion: EXTRACTOR_VERSION - 1, nodes: [], edges: [], keys: [] }]);
        await reviewDirectory(args(memSource({ 'a.md': text }), store));
        const cached = (await store.getExtracted(hash))!;
        expect(cached).toHaveLength(1);
        expect(cached[0]).toMatchObject({ extractorVersion: EXTRACTOR_VERSION });
        expect(cached[0]!.nodes.length).toBeGreaterThan(0);
    });

    it('Continue with a plan that lists a file that can no longer be read skips it, without retiring it', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n' });
        const manifest = new InMemoryIngestManifest();
        const detail = await planDetail(source, manifest, [DOCS_PACK]);
        delete source.files['b.md'];
        const read = source.read.bind(source);
        source.read = async (p) => {
            if (!(p in source.files)) throw new Error('ENOENT');
            return read(p);
        };
        const report = await reviewDirectory(args(source, memStore(), manifest, { plan: detail }));
        expect(report.aborted).toBe(false);
        expect(report.extracted).toBe(1);
        expect(report.readErrors).toBe(1);
        expect([...(await manifestMap(manifest, filesManifestKey(ROOT))).keys()]).toEqual(['a.md']);
    });
});
