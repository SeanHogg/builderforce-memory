/**
 * src/knowledge/owedFiles.contentHash.test.ts — the content-hash rule is part of the extractor identity (PRD 41 §B2 step 0):
 * an entry hashed under another rule owes every file one re-extraction, and the run that pays it records today's rule.
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
    it('a stored state without the rule, from before the rule, marks files owed once', async () => {
        const manifest = new InMemoryIngestManifest();
        await manifest.set(key, [{ id: 'version', hash: '4' }]);
        const before = await loadExtractorState(manifest, ROOT);
        expect(before.contentHashRule).toBeUndefined();
        expect(isOwed(before, 'a.md')).toBe(true);
        expect(owedCalls(before, 'a.md')).toBe(1);

        await saveExtractorState(manifest, ROOT, new Map());
        const after = await loadExtractorState(manifest, ROOT);
        expect(isOwed(after, 'a.md')).toBe(false);
        expect(owedCalls(after, 'a.md')).toBe(0);
    });

    it("a state with today's rule (recorded, or implied by today's version) is not owed", async () => {
        const manifest = new InMemoryIngestManifest();
        await manifest.set(key, [{ id: 'version', hash: String(EXTRACTOR_VERSION) }, { id: 'contentHash', hash: CONTENT_HASH_RULE }]);
        expect(isOwed(await loadExtractorState(manifest, ROOT), 'a.md')).toBe(false);
        await manifest.set(key, [{ id: 'version', hash: String(EXTRACTOR_VERSION) }]);
        expect(isOwed(await loadExtractorState(manifest, ROOT), 'a.md')).toBe(false);
        expect(isOwed({ version: EXTRACTOR_VERSION, owed: new Map() }, 'a.md')).toBe(false);
    });

    it('another recorded rule at today\'s version owes every file one re-extraction, once', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        const owedPaths = async () => (await planDetail(source, manifest, [DOCS_PACK])).files.filter((f) => f.owed).map((f) => f.path);
        expect(await owedPaths()).toEqual([]);

        await manifest.set(key, [{ id: 'version', hash: String(EXTRACTOR_VERSION) }, { id: 'contentHash', hash: 'raw' }]);
        expect(await owedPaths()).toEqual(ALL);
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(await owedPaths()).toEqual([]);
        // Today's rule is the one today's version implies, so the paid-off entry stays in its pre-rule shape.
        expect((await manifest.get(key))!.map((f) => f.id)).toEqual(['version']);
    });
});
