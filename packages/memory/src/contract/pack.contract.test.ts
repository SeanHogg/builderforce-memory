/**
 * src/contract/pack.contract.test.ts — the cross-cutting determinism contract of `./knowledge/pack`, through its
 * PUBLIC entry only. The server compares pack bytes and digests, so two encodes of one pack must be the same
 * bytes, and a pack that went through the wire must encode back to what was sent. (Codec internals are pinned
 * in knowledge/pack/*.test.ts.)
 */

import { decodePack, encodePack, normalizePack } from '../knowledge/pack/index.js';
import { genPack } from '../knowledge/pack/pack.testkit.test-util.js';

const sameBytes = (a: Uint8Array, b: Uint8Array) => Buffer.from(a).equals(Buffer.from(b));

describe('pack contract', () => {
    const packs = [1, 2, 3, 4, 5].map((seed) => genPack(seed, { files: seed * 6, rich: seed % 2 === 0, hashes: seed % 2 === 1 }));

    it('encoding the same pack twice gives equal bytes', async () => {
        for (const pack of packs) {
            expect(sameBytes(await encodePack(pack), await encodePack(pack))).toBe(true);
            expect(sameBytes(await encodePack(pack, { deflate: false }), await encodePack(pack, { deflate: false }))).toBe(true);
        }
    });

    it('decode-encode round trips: decode(encode(p)) equals normalize(p), and re-encoding it gives the original bytes', async () => {
        for (const pack of packs) {
            const bytes = await encodePack(pack);
            const back = await decodePack(bytes);
            expect(back).toEqual(normalizePack(pack));
            expect(sameBytes(await encodePack(back), bytes)).toBe(true);
        }
    });
});
