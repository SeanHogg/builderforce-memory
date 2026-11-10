/**
 * src/knowledge/owedFiles.contentHash.test.ts — the content-hash rule in the extractor entry (PRD 41 §B2 step 0).
 * The rule never owes a file by its ABSENCE: a host switching to `contentHash` moves the hash of every CRLF file, and
 * those re-extract by content. Only a RECORDED rule that differs from today's owes every file one re-extraction.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { CONTENT_HASH_RULE } from './contentHash.js';
import { isOwed, loadExtractorState, owedCalls, saveExtractorState } from './owedFiles.js';
import { DOCS_PACK } from './packs/docs.js';
import { planDetail } from './plan.js';
import { reviewDirectory } from './review.js';
import { EXTRACTOR_VERSION, extractorManifestKey } from './sources.js';
import { ALL, args, fakeJudge, FILES, memSource, memStore, ROOT } from './testkit.test-util.js';

const key = extractorManifestKey(ROOT);

describe('the content-hash rule in the extractor entry', () => {
    it("an entry without a rule record owes nothing by it, at today's version", async () => {
        const manifest = new InMemoryIngestManifest();
        await manifest.set(key, [{ id: 'version', hash: String(EXTRACTOR_VERSION) }]);
        const state = await loadExtractorState(manifest, ROOT);
        expect(state.contentHashRule).toBeUndefined();
        expect(isOwed(state, 'a.md')).toBe(false);
        expect(owedCalls(state, 'a.md')).toBe(0);
        expect(isOwed({ version: EXTRACTOR_VERSION, owed: new Map() }, 'a.md')).toBe(false);
    });

    it("today's rule recorded is read back and owes nothing", async () => {
        const manifest = new InMemoryIngestManifest();
        await manifest.set(key, [{ id: 'version', hash: String(EXTRACTOR_VERSION) }, { id: 'contentHash', hash: CONTENT_HASH_RULE }]);
        const state = await loadExtractorState(manifest, ROOT);
        expect(state.contentHashRule).toBe(CONTENT_HASH_RULE);
        expect(isOwed(state, 'a.md')).toBe(false);
    });

    it("saving today's (first) rule writes no record, so the entry keeps its pre-rule shape", async () => {
        const manifest = new InMemoryIngestManifest();
        await saveExtractorState(manifest, ROOT, new Map());
        expect((await manifest.get(key))!.map((f) => f.id)).toEqual(['version']);
    });

    it("another recorded rule at today's version owes every file one re-extraction, once", async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        const owedPaths = async () => (await planDetail(source, manifest, [DOCS_PACK])).files.filter((f) => f.owed).map((f) => f.path);
        expect(await owedPaths()).toEqual([]);

        await manifest.set(key, [{ id: 'version', hash: String(EXTRACTOR_VERSION) }, { id: 'contentHash', hash: 'eol-lf@0' }]);
        expect(owedCalls(await loadExtractorState(manifest, ROOT), 'a.md')).toBe(1);
        expect(await owedPaths()).toEqual(ALL);
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(await owedPaths()).toEqual([]);
        expect((await manifest.get(key))!.map((f) => f.id)).toEqual(['version']);
    });
});
