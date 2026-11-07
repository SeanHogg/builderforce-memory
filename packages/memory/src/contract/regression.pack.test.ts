/**
 * src/contract/regression.pack.test.ts — bugs fixed in the BFKP codec (2026.10.48), each pinned by one test named
 * `regression: <what broke>`, through the PUBLIC entry only.
 */

import {
    bucketDigest, bucketOf, decodePack, DEFAULT_PACK_LIMITS, encodePack, fileLeafKey, factLeafKey, joinParts, manifestTree, MANIFEST_BUCKETS,
    parseRelationLeafKey, relationLeafKey, rootDigestOfBuckets, splitParts, peekPack, ZERO_DIGEST,
} from '../knowledge/pack/index.js';
import { genPack } from '../knowledge/pack/pack.testkit.test-util.js';

// eslint-disable-next-line no-control-regex
const CONTROL = /[\u0000-\u001f\u007f]/;

describe('regression: relation leaf keys', () => {
    it('regression: a relation leaf key holds no NUL or other control character (Postgres text and jsonb reject NUL)', () => {
        const key = relationLeafKey({ from: 'src:r::a.ts', to: 'fact:claim:1', kind: 'about' });
        expect(CONTROL.test(key)).toBe(false);
        expect(key).toBe('r:["src:r::a.ts","fact:claim:1","about"]');
    });

    it('regression: the other key builders emit no control character for printable input', () => {
        expect(CONTROL.test(fileLeafKey('src/a.ts', 'a'.repeat(64)))).toBe(false);
        expect(CONTROL.test(fileLeafKey('src/a.ts', '~tag:x'))).toBe(false);
        expect(CONTROL.test(factLeafKey('claim:1'))).toBe(false);
    });

    it('regression: parseRelationLeafKey inverts relationLeafKey, including refs that hold separators, quotes and unicode', () => {
        for (const r of [
            { from: 'src:r::a|b.ts', to: 'fact:"quoted"', kind: 'about' },
            { from: 'src:r::\u00e9\u4e2d.ts', to: 'ent:x,y', kind: 'k' },
            { from: 'a', to: 'b', kind: '' },
        ]) expect(parseRelationLeafKey(relationLeafKey(r))).toEqual(r);
    });

    it('regression: parseRelationLeafKey is null for anything that is not a relation key', () => {
        for (const k of ['a:claim', 'f:x', 'r:', 'r:not json', 'r:["a","b"]', 'r:["a","b",3]', 'r:{"a":1}', 'r:\u0000x\u0000y\u0000z']) expect(parseRelationLeafKey(k)).toBeNull();
    });
});

describe('regression: incremental bucket digests', () => {
    it('regression: manifestTree is the bucket digests of its buckets, and the root is the digest of those', () => {
        const leaves = Array.from({ length: 300 }, (_, i) => ({ key: i % 3 ? `a:k${i}` : relationLeafKey({ from: `src:r::${i}`, to: 'fact:x', kind: 'about' }), digest: `d${i}` }));
        const tree = manifestTree(leaves);
        for (const [b, list] of tree.leaves) expect(bucketDigest([...list].reverse())).toBe(tree.buckets[b]);
        expect(bucketDigest([])).toBe(tree.buckets.find((_, b) => !tree.leaves.has(b)));
        expect(rootDigestOfBuckets(tree.buckets)).toBe(tree.root);
        // A pack that touches one bucket recomputes only that bucket and gets the same root as a full rebuild.
        const touched = bucketOf(leaves[0]!.key);
        const next = [...leaves.filter((l) => l.key !== leaves[0]!.key), { key: leaves[0]!.key, digest: 'changed' }];
        const full = manifestTree(next);
        const patched = tree.buckets.slice();
        patched[touched] = bucketDigest(next.filter((l) => bucketOf(l.key) === touched));
        expect(rootDigestOfBuckets(patched)).toBe(full.root);
    });

    it('regression: rootDigestOfBuckets refuses a list that is not all buckets', () => {
        expect(() => rootDigestOfBuckets(['x'])).toThrow();
        expect(rootDigestOfBuckets(new Array(MANIFEST_BUCKETS).fill(bucketDigest([])))).toMatch(/^[0-9a-f]{64}$/);
    });
});

describe('regression: pack memory bounds and parts', () => {
    it('regression: the default decode caps fit a 128 MB Worker (24 MB decoded, 200k rows)', () => {
        expect(DEFAULT_PACK_LIMITS.maxDecodedBytes).toBe(24 * 1024 * 1024);
        expect(DEFAULT_PACK_LIMITS.maxRows).toBe(200_000);
    });

    it('regression: a pack split into parts decodes from the parts, without being joined, to the same pack', async () => {
        const pack = { ...genPack(3, { files: 400, rich: true }), baseGeneration: 4 };
        const bytes = await encodePack(pack);
        const whole = await decodePack(bytes);
        const parts = splitParts(bytes, Math.max(4096, Math.ceil(bytes.length / 3)));
        expect(parts.length).toBeGreaterThan(1);
        expect(await decodePack(parts)).toEqual(whole);
        expect(peekPack(parts)).toEqual(peekPack(bytes));
        expect(Buffer.from(joinParts(parts)).equals(Buffer.from(bytes))).toBe(true);
    });

    it('regression: parts too small to hold the header and trailer still decode (they are joined as a fallback)', async () => {
        const pack = genPack(2, { files: 10 });
        const bytes = await encodePack(pack, { deflate: false });
        expect(await decodePack(splitParts(bytes, 100))).toEqual(await decodePack(bytes));
        const targetless = { rootId: 'r', baseGeneration: 0, targetDigest: ZERO_DIGEST };
        expect(await decodePack(splitParts(await encodePack(targetless), 20))).toEqual(await decodePack(await encodePack(targetless)));
    });

    it('regression: a flipped byte in a later part still fails the CRC when the parts are decoded as a stream', async () => {
        const bytes = await encodePack(genPack(2, { files: 300 }));
        const parts = splitParts(bytes, Math.max(4096, Math.ceil(bytes.length / 3))).map((p) => p.slice());
        parts[1]![Math.floor(parts[1]!.length / 2)]! ^= 0xff;
        await expect(decodePack(parts)).rejects.toThrow();
    });
});
