/**
 * src/knowledge/review.upgrade.test.ts
 * Upgrade and robustness follow-ups: the extractor version lives in its own manifest entry
 * (the files manifest keeps its key), *Re-review all*, a failed read retires nothing, the
 * compare phase without its ports, the reserved preamble ref, claim text and byName determinism.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { hashText } from '../ingest/hash.js';
import { markdownParser } from '../ingest/parsers.js';
import { buildGraph, graphDigest } from './build.js';
import type { ExtractContext, KnowledgePack } from './model.js';
import { DOCS_PACK } from './packs/docs.js';
import { planReview } from './plan.js';
import { reviewDirectory } from './review.js';
import { MAX_AWAITING_CLAIMS, saveAwaiting, type PendingClaims } from './reviewCompare.js';
import {
    asPathSource, compareManifestKey, EXTRACTOR_VERSION, extractorManifestKey, filesManifestKey, judgedManifestKey, manifestMap,
    resolveRelativePath,
} from './sources.js';
import {
    args, edge, extractedOf, fakeJudge, FILES, fileNode, memSource, memStore, quietPack, ref, ROOT, similar, stateOf,
} from './testkit.test-util.js';
import { canonicalJson, factRef, srcRef } from './refs.js';

describe('the extractor version has its own manifest entry', () => {
    it('the files manifest keeps the key consumers read', () => {
        expect(filesManifestKey(ROOT)).toBe(`knowledge:${ROOT}`);
        expect(extractorManifestKey(ROOT)).toBe(`knowledge:${ROOT}:extractor`);
    });

    it('an old manifest (files, no version entry) re-extracts once, still detects removals, and re-anchors unchanged sections', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(await stateOf(manifest)).toEqual({ version: EXTRACTOR_VERSION, owed: [] });
        expect((await manifest.get(filesManifestKey(ROOT)))!.map((f) => f.id)).toEqual(['a.md', 'b.md', 'c.md']);

        delete source.files['b.md'];
        source.files['c.md'] = '# C\ngamma changed\n';
        const withVersion = await planReview(source, manifest, [DOCS_PACK]);
        expect(withVersion.changed).toEqual(['c.md']);

        await manifest.set(extractorManifestKey(ROOT), []); // what a v2026.10.31 manifest looks like
        const old = await planReview(source, manifest, [DOCS_PACK]);
        expect(old.removed).toEqual(['b.md']);
        expect(old.changed).toEqual(['a.md', 'c.md']);

        const judge = fakeJudge();
        const report = await reviewDirectory(args(source, store, manifest, { judgement: judge }));
        expect(report.extracted).toBe(2);
        expect(judge.requests.filter((r) => r.kind === 'extract')).toHaveLength(1); // c.md only; a.md is re-anchored
        expect([...(await manifestMap(manifest, filesManifestKey(ROOT))).keys()]).toEqual(['a.md', 'c.md']);
        expect(await stateOf(manifest)).toEqual({ version: EXTRACTOR_VERSION, owed: [] });
        expect((await planReview(source, manifest, [DOCS_PACK])).changed).toEqual([]);
    });

    it('an aborted upgrade run records the version and owes the files it never reached', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        await manifest.delete(extractorManifestKey(ROOT));

        const controller = new AbortController();
        controller.abort();
        const report = await reviewDirectory(args(source, store, manifest, { signal: controller.signal }));
        expect(report.aborted).toBe(true);
        expect(await stateOf(manifest)).toEqual({ version: EXTRACTOR_VERSION, owed: ['a.md', 'b.md', 'c.md'] });
        expect((await planReview(source, manifest, [DOCS_PACK])).changed).toEqual(['a.md', 'b.md', 'c.md']);
    });
});

describe('re-review all', () => {
    it('marks every file changed, still finds removals and does not inflate the estimate', async () => {
        const source = memSource({ ...FILES });
        const manifest = new InMemoryIngestManifest();
        const store = memStore();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        delete source.files['b.md'];

        const plain = await planReview(source, manifest, [DOCS_PACK]);
        const forced = await planReview(source, manifest, [DOCS_PACK], { force: true });
        expect(plain.changed).toEqual([]);
        expect(forced.changed).toEqual(['a.md', 'c.md']);
        expect(forced.removed).toEqual(['b.md']);
        expect(forced.estimatedModelCalls).toBe(plain.estimatedModelCalls);

        const judge = fakeJudge();
        const report = await reviewDirectory(args(source, store, manifest, { judgement: judge, force: true }));
        expect(report.extracted).toBe(2);
        expect(judge.requests).toHaveLength(0);
        expect([...(await manifestMap(manifest, filesManifestKey(ROOT))).keys()]).toEqual(['a.md', 'c.md']);
    });
});

describe('a file that cannot be read', () => {
    it('keeps its facts, anchors and manifest hash, and is counted', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n' });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        const before = { facts: await store.facts({ rootId: ROOT }), relations: await store.relations({ rootId: ROOT }) };
        const hash = hashText('# A\nalpha\n');

        source.files['a.md'] = '# A\nalpha edited\n';
        source.read = async () => { throw Object.assign(new Error('permission denied'), { code: 'EACCES' }); };
        const report = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(report).toMatchObject({ readErrors: 1, extracted: 0, aborted: false });
        expect(await store.facts({ rootId: ROOT })).toEqual(before.facts);
        expect((await store.relations({ rootId: ROOT })).every((r) => !r.staleAt)).toBe(true);
        expect((await manifestMap(manifest, filesManifestKey(ROOT))).get('a.md')).toBe(hash);
        // Still changed for the next review.
        expect((await planReview(source, manifest, [DOCS_PACK])).changed).toEqual(['a.md']);
    });
});

describe('bookkeeping of removed and binary files', () => {
    it('counts them stale, prunes their judged sections and drops their anchor from shared entities', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect([...(await manifestMap(manifest, judgedManifestKey(ROOT))).keys()].some((k) => k.startsWith(`${srcRef(ROOT, 'b.md')}#`))).toBe(true);

        delete source.files['b.md'];
        delete source.files['c.md'];
        const removed = await reviewDirectory(args(source, store, manifest));
        expect(removed.staleMarked).toBe(2);
        const judged = [...(await manifestMap(manifest, judgedManifestKey(ROOT))).keys()];
        expect(judged.every((k) => k.startsWith(`${srcRef(ROOT, 'a.md')}#`))).toBe(true);
        const lease = (await store.facts({ rootId: ROOT })).find((f) => f.key === 'ent:docs:lease')!;
        expect(lease.fileAnchors.map((a) => a.path)).toEqual(['a.md']);

        source.files['a.md'] = 'bin\u0000ary';
        expect((await reviewDirectory(args(source, store, manifest))).staleMarked).toBe(1);
    });
});

describe('the compare phase', () => {
    it('keeps the claims awaiting when the host has no similar port, so a later run with one can compare them', async () => {
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(memSource({ ...FILES }), memStore(), manifest, { judgement: fakeJudge() }));
        expect((await manifestMap(manifest, compareManifestKey(ROOT))).size).toBeGreaterThan(0);
    });

    it('an extracted-only run (no judgement port) leaves the awaiting set intact', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        const before = await manifest.get(compareManifestKey(ROOT));
        expect(before!.length).toBeGreaterThan(0);

        await reviewDirectory(args(source, store, manifest));
        expect(await manifest.get(compareManifestKey(ROOT))).toEqual(before);
    });

    it('a pack that does not ask for contradictions never touches the awaiting set', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        const before = JSON.stringify(await manifest.get(compareManifestKey(ROOT)));
        expect(before).not.toBe('[]');

        // A new claim exists after this run, yet nothing is saved for it: no pack waits on a compare.
        source.files['a.md'] = '# A\nalphabet\n';
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), packs: [quietPack] }));
        expect((await store.facts({ rootId: ROOT })).some((f) => f.content.startsWith('Lease ends alphabet'))).toBe(true);
        expect(JSON.stringify(await manifest.get(compareManifestKey(ROOT)))).toBe(before);
    });

    it('bounds the awaiting set to the most recent claims', async () => {
        const manifest = new InMemoryIngestManifest();
        const pending: PendingClaims = new Map();
        for (let i = 0; i < MAX_AWAITING_CLAIMS + 50; i++) pending.set(`fact:claim:${String(i).padStart(5, '0')}`, { text: 't', hash: 'h' });
        await saveAwaiting(manifest, ROOT, pending, undefined);
        const kept = [...(await manifestMap(manifest, compareManifestKey(ROOT))).keys()];
        expect(kept).toHaveLength(MAX_AWAITING_CLAIMS);
        expect(kept).toContain(`fact:claim:${String(MAX_AWAITING_CLAIMS + 49).padStart(5, '0')}`);
        expect(kept).not.toContain('fact:claim:00000');
    });

    it('only a pack that asks for contradictions puts its claims up for comparison', async () => {
        const notes: KnowledgePack = {
            id: 'notes',
            parsers: [{ ...markdownParser, id: 'notes-md', accepts: (s) => s.id.endsWith('.txt') }],
            judgement: { nodeKinds: ['section'], claims: true, entities: true, contradictions: false },
        };
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const source = memSource({ 'a.md': '# A\nalpha\n', 'n.txt': '# N\nnotes\n' });
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), packs: [notes, DOCS_PACK] }));

        const claimRef = async (text: string) => {
            const fact = (await store.facts({ rootId: ROOT })).find((f) => f.content.startsWith(text));
            expect(fact).toBeDefined(); // judged either way
            return factRef(fact!.key);
        };
        const awaiting = [...(await manifestMap(manifest, compareManifestKey(ROOT))).keys()];
        expect(awaiting).toEqual([await claimRef('Lease ends alpha')]);
        expect(awaiting).not.toContain(await claimRef('Lease ends notes'));
    });

    it('hands the similar port the texts of the pending claims, as the compare reads them', async () => {
        const calls: Array<{ ids: string[]; texts: ReadonlyMap<string, string> | undefined }> = [];
        const spy = async (ids: string[], texts?: ReadonlyMap<string, string>) => {
            calls.push({ ids, texts });
            return similar(ids);
        };
        const judge = fakeJudge();
        await reviewDirectory(args(memSource({ 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n' }), memStore(), new InMemoryIngestManifest(), {
            judgement: judge, similar: spy,
        }));
        expect(calls).toHaveLength(1);
        expect(calls[0]!.ids).toHaveLength(2);
        expect([...calls[0]!.texts!.keys()].sort()).toEqual(calls[0]!.ids);
        expect([...calls[0]!.texts!.values()].sort()).toEqual(['Lease ends alpha', 'Lease ends beta']);
        // The same texts the compare call carries.
        expect(judge.requests.find((r) => r.kind === 'compare')!.items.map((i) => i.text).sort()).toEqual(['Lease ends alpha', 'Lease ends beta']);
    });

    it('compares a claim left awaiting by its claim text, the form a first run uses', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n', 'b.md': '# B\nbeta\n' });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), similar, budget: { modelCalls: 2, ms: 60_000 } }));
        const judge = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: judge, similar }));
        const compare = judge.requests.find((r) => r.kind === 'compare')!;
        expect(compare.items.map((i) => i.text).sort()).toEqual(['Lease ends alpha', 'Lease ends beta']);
    });
});

describe('docs extract determinism and refs', () => {
    const ctx: ExtractContext = { rootId: ROOT, knownKeys: new Map(), resolvePath: resolveRelativePath };

    it('a preamble and a top-level "# Body" heading keep distinct refs', () => {
        const text = 'Intro text.\n# Body\nunder the heading\n';
        const src = asPathSource({ id: 'p.md', tenantId: 't', content: { kind: 'text', text } }, 'p.md', 'h:p');
        const ex = markdownParser.extract!(src, markdownParser.parse(src), ctx);
        const sections = ex.nodes.filter((n) => n.kind === 'section');
        expect(sections.map((n) => n.ref).sort()).toEqual([`${ref('p.md')}#body`, `${ref('p.md')}#~preamble`].sort());
        expect(sections.find((n) => n.ref.endsWith('#body'))!.label).toBe('Body');
    });

    it('edges that differ only by byName digest and build the same in any input order', () => {
        const edges = [edge('a.md', 'x.md', 'links', { byName: 'X' }), edge('a.md', 'x.md', 'links', { byName: 'Y' })];
        const input = (es: typeof edges) => ({
            rootId: ROOT, packs: [DOCS_PACK], facts: [], relations: [], extracted: [extractedOf(fileNode('a.md'), es)],
        });
        expect(graphDigest(input(edges))).toBe(graphDigest(input([...edges].reverse())));
        expect(canonicalJson(buildGraph(input(edges)))).toBe(canonicalJson(buildGraph(input([...edges].reverse()))));
    });
});
