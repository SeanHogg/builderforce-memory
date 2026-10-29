/**
 * src/knowledge/review.estimate.test.ts
 * The confirm chip's estimate against the calls a run makes — owed files and the awaiting
 * backlog count — and the awaiting set keeping its age order across a save/load.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import type { StoredFact } from './model.js';
import { DOCS_PACK } from './packs/docs.js';
import { planDetail } from './plan.js';
import { reviewDirectory } from './review.js';
import { MAX_AWAITING_CLAIMS, pendingClaims, saveAwaiting, type PendingClaims } from './reviewCompare.js';
import { compareManifestKey, extractorManifestKey, manifestMap } from './sources.js';
import { args, fakeJudge, FILES, memSource, memStore, quietPack, ROOT, similar } from './testkit.test-util.js';
import { compareStr } from './refs.js';

const estimateOf = async (source: ReturnType<typeof memSource>, manifest: InMemoryIngestManifest, packs = [DOCS_PACK]) =>
    (await planDetail(source, manifest, packs)).plan.estimatedModelCalls;

describe('the estimate', () => {
    it('counts the awaiting backlog a run without ports left, and equals the compare calls the first run with ports makes', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect((await manifestMap(manifest, compareManifestKey(ROOT))).size).toBe(3); // one claim per file, never compared

        const estimate = await estimateOf(source, manifest); // nothing changed: just ceil(3 awaiting / 10 per call)
        const judge = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: judge, similar }));
        expect(judge.requests.map((r) => r.kind)).toEqual(['compare']);
        expect(estimate).toBe(judge.requests.length);
    });

    it('adds the compare term only when the run will compare: the same backlog is free with compare:false', async () => {
        const source = memSource({ ...FILES });
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, memStore(), manifest, { judgement: fakeJudge() }));
        expect((await manifestMap(manifest, compareManifestKey(ROOT))).size).toBe(3);

        expect(await estimateOf(source, manifest)).toBe(1); // default: ceil(3 awaiting / 10)
        const noCompare = (await planDetail(source, manifest, [DOCS_PACK], { compare: false })).plan;
        expect(noCompare.estimatedModelCalls).toBe(0);

        // A content-changed file's compare sections drop out too; its own section stays.
        source.files['c.md'] = '# C\ngamma changed\n';
        const changed = await planDetail(source, manifest, [DOCS_PACK], { compare: false });
        expect(changed.plan.estimatedModelCalls).toBe(1);
        expect((await planDetail(source, manifest, [DOCS_PACK])).plan.estimatedModelCalls).toBe(1 + 1);
    });

    it('an upgrade that adds no section is estimated at one call per file, an upper bound that spends nothing', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), packs: [quietPack] }));
        expect(await estimateOf(source, manifest, [quietPack])).toBe(0);

        await manifest.delete(extractorManifestKey(ROOT));
        expect(await estimateOf(source, manifest, [quietPack])).toBe(3);
        const judge = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: judge, packs: [quietPack] }));
        expect(judge.requests).toHaveLength(0); // every section is re-anchored: the estimate is a ceiling
    });

    it('keeps the content-changed count beside the owed files', async () => {
        const source = memSource({ ...FILES });
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, memStore(), manifest, { judgement: fakeJudge() }));
        await manifest.delete(extractorManifestKey(ROOT));
        // 3 owed files, one call each; their 3 owed sections and the 3 claims already awaiting share compare
        // calls of ten: ceil((3 + 3) / 10) = 1.
        expect(await estimateOf(source, manifest)).toBe(3 + 1);

        // c.md also changed: a.md and b.md stay owed (2) and c.md's changed section replaces its owed call (1);
        // compare: ceil((2 owed + 1 changed section + 3 awaiting) / 10) = 1.
        source.files['c.md'] = '# C\ngamma changed\n';
        expect(await estimateOf(source, manifest)).toBe(2 + 1 + 1);
    });
});

describe('the awaiting set keeps its age order across a reload', () => {
    it('drops the oldest claims even though the manifest hands entries back in id order', async () => {
        const manifest = new InMemoryIngestManifest();
        // Ref order is the reverse of age order, so a reload that sorts by ref would get the age wrong.
        const refOf = (n: number) => `fact:claim:${String(99999 - n).padStart(5, '0')}`;
        const facts = new Map<string, StoredFact>();
        const fact = (r: string): StoredFact => ({
            key: r.slice(5), content: 'Lease ends x', source: 'review', importance: 0.5, origin: 'model',
            fileAnchors: [{ path: 'a.md', contentHash: 'h' }],
        });
        const first: PendingClaims = new Map();
        for (let n = 0; n < MAX_AWAITING_CLAIMS - 100; n++) {
            first.set(refOf(n), { text: 'Lease ends x', hash: 'h' });
            const f = fact(refOf(n));
            facts.set(f.key, f);
        }
        await saveAwaiting(manifest, ROOT, first, undefined);
        const stored = (await manifest.get(compareManifestKey(ROOT)))!;
        await manifest.set(compareManifestKey(ROOT), [...stored].sort((a, b) => compareStr(a.id, b.id)));

        const reloaded = await pendingClaims(manifest, ROOT, facts);
        expect([...reloaded.keys()]).toEqual([...first.keys()]);

        for (let n = 0; n < 300; n++) reloaded.set(refOf(MAX_AWAITING_CLAIMS + n), { text: 'Lease ends x', hash: 'h' });
        await saveAwaiting(manifest, ROOT, reloaded, undefined);
        const kept = [...(await manifestMap(manifest, compareManifestKey(ROOT))).keys()];
        expect(kept).toHaveLength(MAX_AWAITING_CLAIMS);
        // 2200 pending, 200 over the cap: the 200 oldest of the first run go, nothing newer.
        expect(kept).toEqual([...reloaded.keys()].slice(200));
    });
});
