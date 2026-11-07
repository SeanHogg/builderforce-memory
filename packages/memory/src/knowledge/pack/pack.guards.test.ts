/**
 * knowledge/pack/pack.guards.test.ts — what the server's guards rely on (PRD 41 A11 Accept): a flipped
 * byte fails the CRC, an unknown section is skipped, a bomb stops at the cap, parts re-join.
 */

import { deflateRaw } from './deflate.js';
import { FLAG_DEFLATED, readFrame, writeFrame } from './frame.js';
import { decodePack, encodePack, joinParts, PackFormatError, PackTooLargeError, peekPack, splitParts, ZERO_DIGEST } from './index.js';
import { genPack } from './pack.testkit.test-util.js';

describe('BFKP guards', () => {
    it('rejects a flipped byte of an uncompressed body with a CRC failure', async () => {
        const bytes = await encodePack(genPack(2, { files: 10 }), { deflate: false });
        const bad = bytes.slice();
        bad[Math.floor(bytes.length / 2)]! ^= 0x01;
        await expect(decodePack(bad)).rejects.toThrow(PackFormatError);
    });

    it('rejects a flipped byte of a deflated body (corrupt stream or CRC)', async () => {
        const bytes = await encodePack(genPack(2, { files: 10 }));
        const bad = bytes.slice();
        bad[Math.floor(bytes.length / 2)]! ^= 0xff;
        await expect(decodePack(bad)).rejects.toThrow(PackFormatError);
    });

    it('rejects a bad magic, a truncated frame and an unknown version', async () => {
        const bytes = await encodePack(genPack(2, { files: 3 }));
        await expect(decodePack(new Uint8Array([1, 2, 3]))).rejects.toThrow(PackFormatError);
        const magic = bytes.slice(); magic[0] = 0x00;
        await expect(decodePack(magic)).rejects.toThrow(/magic/);
        const version = bytes.slice(); version[4] = 9;
        await expect(decodePack(version)).rejects.toThrow(/version/);
    });

    it('skips a section tag it does not know', async () => {
        const pack = genPack(4, { files: 8 });
        const { header, stored } = readFrame(await encodePack(pack, { deflate: false }));
        const body = new Uint8Array(stored.length + 5);
        body.set(stored);
        body.set([0x7f, 3, 9, 9, 9], stored.length);
        const decoded = await decodePack(writeFrame({ ...header, flags: 0 }, body, body));
        expect(decoded).toEqual(await decodePack(await encodePack(pack)));
    });

    it('stops inflating at the cap: a small bomb never holds more than the limit', async () => {
        const body = new Uint8Array(8 * 1024 * 1024);
        const bomb = writeFrame({ flags: FLAG_DEFLATED, rootId: 'r', baseGeneration: 0, targetDigest: ZERO_DIGEST }, body, await deflateRaw(body));
        expect(bomb.length).toBeLessThan(100_000);
        await expect(decodePack(bomb, { maxDecodedBytes: 1024 * 1024 })).rejects.toThrow(PackTooLargeError);
    });

    it('applies the 64 MB default cap', async () => {
        const body = new Uint8Array(65 * 1024 * 1024);
        const bomb = writeFrame({ flags: FLAG_DEFLATED, rootId: 'r', baseGeneration: 0, targetDigest: ZERO_DIGEST }, body, await deflateRaw(body));
        await expect(decodePack(bomb)).rejects.toThrow(PackTooLargeError);
    }, 60_000);

    it('bounds counts before allocating for them', async () => {
        const bytes = await encodePack(genPack(5, { files: 20 }));
        await expect(decodePack(bytes, { maxRows: 5 })).rejects.toThrow(PackFormatError);
        await expect(decodePack(bytes, { maxStrings: 5 })).rejects.toThrow(PackFormatError);
    });

    it('reads the generation and digest without inflating the body', async () => {
        const pack = { ...genPack(6, { files: 4 }), baseGeneration: 7 };
        expect(peekPack(await encodePack(pack))).toEqual({ rootId: pack.rootId, baseGeneration: 7, targetDigest: pack.targetDigest });
    });

    it('splits into ordered parts and joins them back byte for byte', async () => {
        const bytes = await encodePack(genPack(8, { files: 30 }), { deflate: false });
        const parts = splitParts(bytes, 1000);
        expect(parts.length).toBeGreaterThan(1);
        expect(parts.every((p) => p.length <= 1000)).toBe(true);
        expect(Buffer.from(joinParts(parts)).equals(Buffer.from(bytes))).toBe(true);
        expect(splitParts(bytes, bytes.length + 1)).toHaveLength(1);
    });
});
