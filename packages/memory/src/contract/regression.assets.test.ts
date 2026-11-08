/**
 * src/contract/regression.assets.test.ts — assets (binary files stored as a file node, never read): how their asset-ness follows the
 * packs, how a cut treats them, and what the upgrade and the removal of one do. Each bug pinned by one test named `regression: …`.
 */

import { hashText } from '../ingest/hash.js';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { isBinaryDocumentType } from '../ingest/parserSupport.js';
import { assetExtracted } from '../knowledge/assetFile.js';
import { buildGraph } from '../knowledge/build.js';
import type { KnowledgePack, KnowledgeParser } from '../knowledge/model.js';
import { DOCS_PACK } from '../knowledge/packs/docs.js';
import { planDetail, planReview } from '../knowledge/plan.js';
import { srcRef } from '../knowledge/refs.js';
import { reviewDirectory } from '../knowledge/review.js';
import { extractJob, type ExtractPool } from '../knowledge/extractJob.js';
import { EXTRACTOR_VERSION, extractorManifestKey, filesManifestKey, isBinaryText, manifestMap, mediaTypeOfPath, saveManifestMap } from '../knowledge/sources.js';
import { args, edge, extractedOf, fileNode, memSource, memStore, ROOT, stateOf } from '../knowledge/testkit.test-util.js';

const PNG = 'PNGDATA';
const pdfParser: KnowledgeParser = {
    id: 'pdf-text',
    accepts: (s) => s.content.kind === 'text' && s.content.mediaType === 'application/pdf',
    parse: (s) => [{ text: s.content.kind === 'text' ? s.content.text : '' }],
};
const withPdf: KnowledgePack = { ...DOCS_PACK, parsers: [pdfParser, ...DOCS_PACK.parsers] };

describe('regression: asset-ness follows the packs', () => {
    it('regression: a pdf reviewed as an asset is re-processed when a host parser arrives, and an extracted one when it leaves', async () => {
        const source = memSource({ 'a.pdf': 'pdf text', 'b.md': '# B\n' });
        const manifest = new InMemoryIngestManifest();
        const store = memStore();
        const hash = hashText('pdf text');
        await reviewDirectory(args(source, store, manifest)); // no parser: a.pdf is an asset
        expect((await planReview(source, manifest, [DOCS_PACK])).changed).toEqual([]);

        // A parser arrives: the bytes did not move, the version did not move — the file is still owed.
        const arrived = await planReview(source, manifest, [withPdf]);
        expect(arrived).toMatchObject({ changed: ['a.pdf'], assets: 0 });
        const report = await reviewDirectory(args(source, store, manifest, { packs: [withPdf] }));
        expect(report.extracted).toBe(1);
        expect((await store.getExtracted(hash))![0]!.nodes[0]!.badges?.asset).toBeUndefined();
        expect((await planReview(source, manifest, [withPdf])).changed).toEqual([]);
        expect((await stateOf(manifest)).owed).toEqual([]);

        // The parser leaves: the extraction is replaced by the asset node.
        const gone = await planReview(source, manifest, [DOCS_PACK]);
        expect(gone).toMatchObject({ changed: ['a.pdf'], assets: 1 });
        await reviewDirectory(args(source, store, manifest, { packs: [DOCS_PACK] }));
        expect((await store.getExtracted(hash))![0]!.nodes).toMatchObject([{ kind: 'file', badges: { asset: 1, mediaType: 'application/pdf' } }]);
        expect((await planReview(source, manifest, [DOCS_PACK])).changed).toEqual([]);
    });

    it('regression: a plan handed to Continue keeps the flip, so the files it owes are still re-processed', async () => {
        const source = memSource({ 'a.pdf': 'pdf text' });
        const manifest = new InMemoryIngestManifest();
        const store = memStore(); // ONE store across the runs, so a stale asset entry in the extract cache cannot hide
        await reviewDirectory(args(source, store, manifest));
        const detail = await planDetail(source, manifest, [withPdf]);
        expect(detail.plan.changed).toEqual(['a.pdf']);
        const report = await reviewDirectory(args(source, store, manifest, { packs: [withPdf], plan: detail }));
        expect(report.extracted).toBe(1);
        expect((await store.getExtracted(hashText('pdf text')))![0]!.nodes[0]!.badges?.asset).toBeUndefined();
    });

    it('regression: an aborted run during a documents flip records the new types, and the files it did not finish stay owed for the next run', async () => {
        const files = Object.fromEntries(['a.pdf', 'b.pdf', 'c.pdf'].map((p) => [p, `text of ${p}`]));
        const source = memSource({ ...files });
        const manifest = new InMemoryIngestManifest();
        const store = memStore();
        await reviewDirectory(args(source, store, manifest)); // all three are assets
        const stop = new AbortController();
        const first = await reviewDirectory(args(source, store, manifest, {
            packs: [withPdf], signal: stop.signal, concurrency: 1, onProgress: (p) => { if (p.phase === 'write') stop.abort(); },
        }));
        expect(first.aborted).toBe(true);
        const owed = (await stateOf(manifest)).owed;
        expect(owed.length).toBeGreaterThan(0); // unfinished flipped files stay owed although `documents` is already 'pdf'
        expect(owed.every((p) => p in files)).toBe(true);

        const next = await reviewDirectory(args(source, store, manifest, { packs: [withPdf] }));
        expect(next.aborted).toBe(false);
        expect((await stateOf(manifest)).owed).toEqual([]);
        for (const text of Object.values(files)) expect((await store.getExtracted(hashText(text)))![0]!.nodes[0]!.badges?.asset).toBeUndefined();
    });
});

describe('regression: assets in a cut graph', () => {
    it('regression: a truncated graph never fills its slice with assets, nor keeps the ones beside a kept file', () => {
        const docs = ['d0.md', 'd1.md', 'd2.md', 'd3.md'];
        const icons = Array.from({ length: 6 }, (_, i) => `icons/i${i}.png`);
        const extracted = [
            extractedOf(fileNode('d0.md'), icons.map((p) => edge('d0.md', p))),
            ...docs.slice(1).map((p) => extractedOf(fileNode(p))),
            ...icons.map((p) => assetExtracted(ROOT, { path: p, hash: `h:${p}`, size: 1 })),
        ];
        // The control: uncut, the linked assets OUTRANK an isolated doc by importance (computed by PageRank, not the node's stored 0),
        // so only the assets-last order keeps them out of the slice.
        const whole = buildGraph({ rootId: ROOT, extracted, packs: [DOCS_PACK], facts: [], relations: [] });
        const importanceOf = (path: string) => whole.nodes.find((n) => n.ref === srcRef(ROOT, path))!.importance;
        expect(importanceOf('icons/i0.png')).toBeGreaterThan(importanceOf('d1.md'));

        const g = buildGraph({ rootId: ROOT, extracted, packs: [DOCS_PACK], facts: [], relations: [] }, { maxNodes: 3 });
        expect(g.stats.truncated).toBe(true);
        expect(g.nodes.filter((n) => n.badges?.['asset'] !== undefined)).toEqual([]);
        expect(g.nodes.filter((n) => n.kind === 'file').length).toBeGreaterThan(0);
        // The outline still lists every file, assets included, with their badges.
        expect(g.outline!.files.filter((f) => f.badges?.['asset'] === 1)).toHaveLength(icons.length);
    });
});

describe('regression: lockfiles and fonts', () => {
    it('regression: lockfiles are ignored (not assets, not extracted as text); a sqlite db stays an asset', async () => {
        const source = memSource({ 'yarn.lock': 'x', 'crate/Cargo.lock': 'y', 'app.db': 'z', 'a.md': '# A\n' });
        const plan = await planReview(source, new InMemoryIngestManifest(), [DOCS_PACK]);
        expect(plan.skipped).toEqual(expect.arrayContaining([{ path: 'yarn.lock', reason: 'ignored' }, { path: 'crate/Cargo.lock', reason: 'ignored' }]));
        expect(plan).toMatchObject({ files: 2, assets: 1 });
    });

    it('regression: a font object is not a binary document type (only pdf and office are), and an .eot is an asset', async () => {
        expect(mediaTypeOfPath('f.eot')).toBe('application/vnd.ms-fontobject');
        expect(isBinaryDocumentType('application/vnd.ms-fontobject')).toBe(false);
        for (const type of ['application/vnd.ms-excel', 'application/vnd.ms-powerpoint', 'application/msword', 'application/pdf']) expect(isBinaryDocumentType(type)).toBe(true);
        expect((await planReview(memSource({ 'f.eot': 'x' }), new InMemoryIngestManifest(), [DOCS_PACK])).assets).toBe(1);
    });
});

describe('regression: html references', () => {
    it('regression: img, source, link and svg image references are followed; external and protocol-relative ones are not', async () => {
        const html = [
            '<img src="icons/a.png"><source src="v/b.webm"><link rel="stylesheet" href="css/c.css">',
            '<svg><image xlink:href="icons/d.png"/></svg><a href="//cdn.example.com/x.png">x</a><img src="https://x.test/y.png">',
        ].join('\n');
        const source = memSource({ 'page.html': html });
        const store = memStore();
        await reviewDirectory(args(source, store, new InMemoryIngestManifest()));
        const ex = (await store.getExtracted(hashText(html)))!.find((e) => e.sourceRef === srcRef(ROOT, 'page.html'))!;
        expect(ex.edges.filter((e) => e.kind === 'links').map((e) => e.to)).toEqual(['css/c.css', 'icons/a.png', 'icons/d.png', 'v/b.webm'].map((p) => srcRef(ROOT, p)));
    });

    it('regression: a protocol-relative markdown link is external, not a missing file', async () => {
        const md = '# R\n[cdn](//cdn.example.com/x.png) and [local](icons/a.png)\n';
        const store = memStore();
        await reviewDirectory(args(memSource({ 'README.md': md }), store, new InMemoryIngestManifest()));
        const ex = (await store.getExtracted(hashText(md)))![0]!;
        expect(ex.edges.filter((e) => e.kind === 'links').map((e) => e.to)).toEqual([srcRef(ROOT, 'icons/a.png')]);
    });
});

describe('regression: the asset lifecycle', () => {
    it('regression: a manifest at the file\'s current hash with no stored body queues the asset, so its node is written without a read', async () => {
        const source = memSource({ 'icons/a.png': PNG, 'a.md': '# A\n' });
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, memStore(), manifest));
        const fresh = memStore(); // the manifest names the files; this store holds no body
        source.reads.length = 0;
        await reviewDirectory(args(source, fresh, manifest));
        expect(source.reads).toEqual([]); // the asset is never read
        expect((await fresh.getExtracted(hashText(PNG)))![0]!.nodes[0]).toMatchObject({ kind: 'file', badges: { asset: 1, mediaType: 'image/png' } });
    });

    it('regression: a v3 root whose binary files hold a legacy [] gets their asset nodes at v4', async () => {
        const source = memSource({ 'icons/a.png': PNG, 'a.md': '# A\n' });
        const manifest = new InMemoryIngestManifest();
        await saveManifestMap(manifest, filesManifestKey(ROOT), new Map([['icons/a.png', hashText(PNG)], ['a.md', hashText('# A\n')]]));
        await manifest.set(extractorManifestKey(ROOT), [{ id: 'version', hash: String(EXTRACTOR_VERSION - 1) }]);
        const store = memStore();
        await store.putExtracted(hashText(PNG), []); // what v3 stored for a binary file
        await reviewDirectory(args(source, store, manifest));
        expect((await store.getExtracted(hashText(PNG)))![0]!.nodes[0]).toMatchObject({ kind: 'file', badges: { asset: 1, size: PNG.length } });
        expect((await stateOf(manifest)).version).toBe(EXTRACTOR_VERSION);
    });

    it('regression: a removed asset is retired like any removed file', async () => {
        const source = memSource({ 'icons/a.png': PNG, 'a.md': '# A\n' });
        const manifest = new InMemoryIngestManifest();
        const store = memStore();
        await reviewDirectory(args(source, store, manifest));
        expect((await manifestMap(manifest, filesManifestKey(ROOT))).has('icons/a.png')).toBe(true);

        delete source.files['icons/a.png'];
        expect((await planReview(source, manifest, [DOCS_PACK])).removed).toEqual(['icons/a.png']);
        const report = await reviewDirectory(args(source, store, manifest));
        expect(report.staleMarked).toBe(1);
        expect((await manifestMap(manifest, filesManifestKey(ROOT))).has('icons/a.png')).toBe(false);
    });

    it('regression: with an extractPool that reads (deferRead) an asset is never sent to the pool, and a file the pool finds binary becomes an asset node', async () => {
        const nul = 'bin\u0000ary';
        const source = memSource({ 'a.md': '# A\nalpha\n', 'icons/a.png': PNG, 'blob.txt': nul });
        const sent: string[] = [];
        const pool: ExtractPool = async (job) => {
            sent.push(job.path);
            const doc = await source.read(job.path);
            if (doc.content.kind === 'text' && isBinaryText(doc.content.text)) return { binary: true, extracted: [] };
            return extractJob({ ...job, doc } as never, [DOCS_PACK]);
        };
        const store = memStore();
        const report = await reviewDirectory(args(source, store, new InMemoryIngestManifest(), { extractPool: pool, deferRead: true }));
        expect(report.failures).toEqual([]);
        expect(sent).not.toContain('icons/a.png');
        expect(sent).toContain('blob.txt');
        expect((await store.getExtracted(hashText(PNG)))![0]!.nodes[0]).toMatchObject({ badges: { asset: 1, mediaType: 'image/png' } });
        expect((await store.getExtracted(hashText(nul)))![0]!.nodes[0]).toMatchObject({ badges: { asset: 1, mediaType: 'text/plain' } });
    });
});
