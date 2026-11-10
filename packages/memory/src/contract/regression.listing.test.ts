/**
 * src/contract/regression.listing.test.ts — the plan never throws over what a host lists (PRD 41 §B2 step 4, §B8.1).
 * Plan paths go through the ONE normalization (POSIX, NFC), so they are the Merkle paths; a listing that collides once
 * normalized, or names `.`/`..`, is skipped with a reason, and a re-spelled file is still read by the source's spelling.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { merkleTree } from '../knowledge/merkle.js';
import { DOCS_PACK } from '../knowledge/packs/docs.js';
import { planDetail } from '../knowledge/plan.js';
import { reviewDirectory } from '../knowledge/review.js';
import { args, fakeJudge, memSource, memStore } from '../knowledge/testkit.test-util.js';

const NFC = 'café.md', NFD = 'café.md';

describe('regression: a host listing cannot break the plan', () => {
    it('regression: an NFC/NFD pair of one name with different content plans (one kept, one skipped) instead of throwing', async () => {
        const source = memSource({ [NFC]: '# Café\nnfc\n', [NFD]: '# Café\nnfd\n', 'ok.md': '# OK\n' });
        const detail = await planDetail(source, new InMemoryIngestManifest(), [DOCS_PACK]);
        expect(detail.files.map((f) => f.path)).toEqual([NFC, 'ok.md']);
        expect(detail.files[0]!.listedPath).toBe(NFD); // the first by (normalized, listed) bytewise
        expect(detail.plan.skipped).toContainEqual({ path: NFC, reason: 'duplicate-path' });
        expect(detail.plan.merkleRoot).toBe(merkleTree(detail.files).root);
    });

    it('regression: a `.`/`..` path or a path listed twice is skipped with a reason, never thrown', async () => {
        const source = memSource({ 'a.md': '# A\n', './a.md': '# A2\n', '../up.md': '# Up\n', 'd/./x.md': '# X\n' });
        const detail = await planDetail(source, new InMemoryIngestManifest(), [DOCS_PACK]);
        expect(detail.files.map((f) => f.path)).toEqual(['a.md']);
        expect(detail.plan.skipped).toEqual(expect.arrayContaining([
            { path: '../up.md', reason: 'invalid-path' }, { path: 'd/./x.md', reason: 'invalid-path' }, { path: 'a.md', reason: 'duplicate-path' },
        ]));
    });

    it('regression: a file the source lists in NFD is read by that spelling and stored under its NFC path', async () => {
        const source = memSource({ [NFD]: '# Café\nbody\n' });
        const store = memStore();
        const report = await reviewDirectory(args(source, store, new InMemoryIngestManifest(), { judgement: fakeJudge() }));
        expect(report.readErrors).toBe(0);
        expect(source.reads).toContain(NFD);
        expect(source.reads).not.toContain(NFC);
    });
});
