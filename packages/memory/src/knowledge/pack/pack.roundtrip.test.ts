/**
 * knowledge/pack/pack.roundtrip.test.ts — BFKP v1 is lossless and deterministic (PRD 41 A11 Accept):
 * a property-style round trip over seeded generated graphs, and equal input gives equal bytes.
 */

import { decodePack, encodePack, normalizePack, ZERO_DIGEST, type Pack } from './index.js';
import { genPack } from './pack.testkit.test-util.js';

describe('BFKP round trip', () => {
    it('decode(encode(x)) deep-equals the normalized input over 40 generated graphs', async () => {
        for (let seed = 1; seed <= 40; seed++) {
            const pack = genPack(seed, { files: seed % 11 === 0 ? 0 : (seed * 7) % 45, rich: seed % 2 === 0, hashes: seed % 5 === 0 });
            const decoded = await decodePack(await encodePack(pack));
            expect(decoded).toEqual(normalizePack(pack));
        }
    });

    it('round-trips an uncompressed frame the same as a deflated one', async () => {
        const pack = genPack(3, { files: 20 });
        expect(await decodePack(await encodePack(pack, { deflate: false }))).toEqual(await decodePack(await encodePack(pack)));
    });

    it('is deterministic: equal input, in any row order, gives equal bytes', async () => {
        const pack = genPack(7, { files: 30 });
        const shuffled: Pack = { ...pack, files: [...pack.files!].reverse(), facts: [...(pack.facts ?? [])].reverse(), relations: [...(pack.relations ?? [])].reverse() };
        const a = await encodePack(pack), b = await encodePack(pack), c = await encodePack(shuffled);
        expect(Buffer.from(a).equals(Buffer.from(b))).toBe(true);
        expect(Buffer.from(a).equals(Buffer.from(c))).toBe(true);
    });

    it('carries an empty pack (frame only)', async () => {
        const empty: Pack = { rootId: 'r', baseGeneration: 0, targetDigest: ZERO_DIGEST };
        expect(await decodePack(await encodePack(empty))).toEqual(empty);
    });

    it('keeps explicit hashes (a have request) apart from referenced ones', async () => {
        const have: Pack = { rootId: 'r', baseGeneration: 0, targetDigest: ZERO_DIGEST, hashes: ['a'.repeat(64), '~tag:x', 'b'.repeat(64)] };
        const back = await decodePack(await encodePack(have));
        expect(back.hashes).toEqual(['a'.repeat(64), 'b'.repeat(64), '~tag:x']);
    });

    it('round-trips awkward numbers exactly', async () => {
        const g = genPack(1, { files: 1 });
        g.facts = [0, 1, 0.5, 1e-9, 0.1234567, 2 ** 40, 1e15 + 0.5, 123456789.123456789].map((importance, i) => ({
            key: `k${i}`, content: 'c', source: 's', importance, origin: 'model' as const, fileAnchors: [],
        }));
        const back = await decodePack(await encodePack(g));
        expect(back.facts!.map((f) => f.importance).sort()).toEqual(g.facts.map((f) => f.importance).sort());
    });
});
