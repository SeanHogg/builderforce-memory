/**
 * knowledge/pack/pack.size.test.ts — the size promise (PRD 41 A11 Accept): a first sync of a 20,000-file
 * fixture is at most 10% of the same rows as JSON, and a re-review of 10 files is under 50 KB.
 */

import { encodePack, jsonBytesOf, MAX_PART_BYTES, splitParts } from './index.js';
import { genPack } from './pack.testkit.test-util.js';

describe('BFKP size', () => {
    it('packs 20,000 files into at most 10% of the same rows as JSON, in 1 + ceil(size / 8 MB) requests at most', async () => {
        const pack = genPack(11, { files: 20_000, symbols: 4, edgesPerFile: 3, facts: 4_000, rich: false });
        const json = jsonBytesOf(pack);
        const bytes = await encodePack(pack);
        expect(bytes.length / json).toBeLessThanOrEqual(0.1);
        expect(splitParts(bytes).length).toBeLessThanOrEqual(1 + Math.ceil(bytes.length / MAX_PART_BYTES));
    }, 120_000);

    it('a re-review that changes 10 files is one small pack', async () => {
        const full = genPack(12, { files: 10, symbols: 4 });
        expect((await encodePack(full)).length).toBeLessThan(50 * 1024);
    });
});
