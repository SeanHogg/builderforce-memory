/**
 * src/contract/regression.packWire.test.ts — a BFKP pack is untrusted client input. Bugs fixed in how the decoder reads it, each
 * pinned by one test named `regression: <what broke>`. Packs are hand-built from the frame and section primitives and decoded
 * through the PUBLIC `decodePack`.
 */

import { readFrame, writeFrame, writeSection } from '../knowledge/pack/frame.js';
import { TAG } from '../knowledge/pack/sections/index.js';
import { writeEdges, writeKeys, writeNodes } from '../knowledge/pack/sections/extrLists.js';
import { collector, tablesOf, writeHashTable, writeStringTable, type Tables } from '../knowledge/pack/strings.js';
import { ByteReader, ByteWriter, MAX_VARINT } from '../knowledge/pack/varint.js';
import {
    decodePack, encodePack, factDigest, keyHash, PackFormatError, PackTooLargeError, relationLeafKey, rootDigestOf, ZERO_DIGEST, type Pack,
} from '../knowledge/pack/index.js';
import type { KNode, NodeRef } from '../knowledge/model.js';

const HEAD = { flags: 0, rootId: 'r', baseGeneration: 0, targetDigest: ZERO_DIGEST };
const H1 = 'a'.repeat(64), H2 = 'b'.repeat(64);
const NODE = (ref: string): KNode => ({ ref: ref as NodeRef, kind: 'file', label: 'a', pack: 'docs', importance: 0, origin: 'extracted' });

/** A frame whose body is these sections, in order, written as given. */
function frameOf(sections: ReadonlyArray<readonly [number, Uint8Array]>): Uint8Array {
    const w = new ByteWriter();
    for (const [tag, bytes] of sections) writeSection(w, tag, bytes);
    const body = w.finish();
    return writeFrame(HEAD, body, body);
}

/**
 * One EXTR hash, `kind` byte as given (0 full, 1 diff); for a diff, its node list says `positions` add positions but carries
 * `nodes` items. `vflags` is the entry's version flag byte.
 */
function extrPack(o: { kind?: number; positions?: number; nodes?: number; vflags?: number } = {}): Uint8Array {
    const kind = o.kind ?? 1, positions = o.positions ?? 1, nodeCount = o.nodes ?? 1, vflags = o.vflags ?? 0;
    const nodes = Array.from({ length: nodeCount }, (_, i) => NODE(`src:r::n${i}.ts`));
    const write = (t: Tables): Uint8Array => {
        const w = new ByteWriter();
        w.varint(1);
        w.u8(kind);
        w.varint(t.hash(H2));
        if (kind === 1) w.varint(t.hash(H1));
        w.varint(1);
        w.varint(t.str('src:r::a.ts'));
        w.varint(t.hash(H2));
        w.u8(vflags);
        if (vflags & 1) w.varint(2);
        if (kind === 1) {
            w.varint(0); w.varint(positions);
            for (let i = 0; i < positions; i++) w.varint(i === 0 ? 0 : 1);
            writeNodes(w, t, nodes);
            w.varint(0); w.varint(0); writeEdges(w, t, []);
            w.varint(0); w.varint(0); writeKeys(w, t, []);
        } else {
            writeNodes(w, t, nodes); writeEdges(w, t, []); writeKeys(w, t, []);
        }
        return w.finish();
    };
    const col = collector();
    write(col);
    const t = tablesOf(col.strings, col.hashes);
    return frameOf([[TAG.STR, writeStringTable(t.strings)], [TAG.HASH, writeHashTable(t.hashes)], [TAG.EXTR, write(t)]]);
}

/** The body sections of a real (uncompressed) pack, so one can be altered. */
async function sectionsOf(pack: Pack): Promise<Array<[number, Uint8Array]>> {
    const { stored } = readFrame(await encodePack(pack, { deflate: false }));
    const r = new ByteReader(stored);
    const out: Array<[number, Uint8Array]> = [];
    while (!r.done) { const tag = r.u8(); out.push([tag, r.bytes(r.varint())]); }
    return out;
}

const FILES_PACK: Pack = { ...HEAD, files: [{ path: 'a.ts', hash: H1, bytes: 3 }] };

describe('regression: EXTR diff counts', () => {
    it('regression: a well-formed diff (positions = items) still decodes', async () => {
        const pack = await decodePack(extrPack());
        const x = pack.extracted![0]!;
        expect(x.kind).toBe('diff');
        if (x.kind === 'diff') expect(x.entries[0]!.nodes.add).toEqual([{ at: 0, item: NODE('src:r::n0.ts') }]);
    });

    it('regression: more add positions than added items is malformed (it stored `undefined` items)', async () => {
        await expect(decodePack(extrPack({ positions: 2, nodes: 1 }))).rejects.toThrow(/2 add positions but 1 added items/);
    });

    it('regression: more added items than add positions is malformed (the extras were dropped)', async () => {
        await expect(decodePack(extrPack({ positions: 1, nodes: 2 }))).rejects.toThrow(/1 add positions but 2 added items/);
    });
});

describe('regression: unknown flags, kinds and versions are refused', () => {
    it('regression: an unknown frame flag bit is refused, not ignored', async () => {
        const bytes = (await encodePack(FILES_PACK, { deflate: false })).slice();
        bytes[5] = bytes[5]! | 0x02;
        await expect(decodePack(bytes)).rejects.toThrow(/unknown flag bits/);
    });

    it('regression: an EXTR kind byte other than full (0) or diff (1) is refused (2-255 were read as full)', async () => {
        await expect(decodePack(extrPack({ kind: 0 }))).resolves.toBeDefined();
        await expect(decodePack(extrPack({ kind: 2 }))).rejects.toThrow(/unknown EXTR entry kind 2/);
        await expect(decodePack(extrPack({ kind: 255 }))).rejects.toThrow(PackFormatError);
    });

    it('regression: an unknown bit in an EXTR entry version byte is refused', async () => {
        await expect(decodePack(extrPack({ vflags: 1 }))).resolves.toBeDefined();
        await expect(decodePack(extrPack({ vflags: 3 }))).rejects.toThrow(/unknown flag bits/);
    });
});

describe('regression: section framing', () => {
    it('regression: a repeated section tag is refused (the last copy silently won)', async () => {
        const sections = await sectionsOf(FILES_PACK);
        const file = sections.find(([tag]) => tag === TAG.FILE)!;
        await expect(decodePack(frameOf([...sections, file]))).rejects.toThrow(/appears twice/);
    });

    it('regression: a section with trailing bytes is refused (it was read short and the rest ignored)', async () => {
        const sections = await sectionsOf(FILES_PACK);
        const padded = sections.map(([tag, b]): [number, Uint8Array] => (tag === TAG.FILE ? [tag, new Uint8Array([...b, 0])] : [tag, b]));
        await expect(decodePack(frameOf(padded))).rejects.toThrow(/FILE section: 1 trailing bytes/);
        const str = sections.map(([tag, b]): [number, Uint8Array] => (tag === TAG.STR ? [tag, new Uint8Array([...b, 7])] : [tag, b]));
        await expect(decodePack(frameOf(str))).rejects.toThrow(/string table: 1 trailing bytes/);
        await expect(decodePack(frameOf(sections))).resolves.toEqual(await decodePack(await encodePack(FILES_PACK)));
    });
});

describe('regression: lone surrogates on the wire', () => {
    const lone = 'dir/a\uD800.ts';
    const wire = 'dir/a�.ts';

    it('regression: a path with a lone surrogate gets the same root digest on the client and after decoding on the server', async () => {
        const client = { files: [{ path: lone, hash: H1 }], facts: [], relations: [] };
        const decoded = await decodePack(await encodePack({ ...HEAD, files: [{ path: lone, hash: H1, bytes: 1 }] }));
        expect(decoded.files![0]!.path).toBe(wire);
        const server = { files: decoded.files!.map((f) => ({ path: f.path, hash: f.hash })), facts: [], relations: [] };
        expect(rootDigestOf('r', client)).toBe(rootDigestOf('r', server));
        expect(keyHash(lone)).toBe(keyHash(wire));
        expect(relationLeafKey({ from: `src:r::${lone}`, to: 'fact:x', kind: 'about' })).toBe(relationLeafKey({ from: `src:r::${wire}`, to: 'fact:x', kind: 'about' }));
        expect(factDigest({ key: lone, content: lone, origin: 'model' })).toBe(factDigest({ key: wire, content: wire, origin: 'model' }));
    });

    it('regression: a well-formed key (including a real surrogate pair) hashes exactly as before', () => {
        const fnvUtf16 = (k: string) => { let h = 0x811c9dc5; for (let i = 0; i < k.length; i++) { h ^= k.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h >>> 0; };
        for (const k of ['f:src/a.ts', 'a:claim:日本語', 'f:emoji😀.md', '']) expect(keyHash(k)).toBe(fnvUtf16(k));
    });
});

describe('regression: canonical varints', () => {
    const read = (bytes: number[]) => new ByteReader(new Uint8Array(bytes)).varint();

    it('regression: an overlong (non-canonical) varint is malformed', () => {
        expect(() => read([0x80, 0x00])).toThrow(/non-canonical/);
        expect(() => read([0x81, 0x80, 0x00])).toThrow(/non-canonical/);
        expect(read([0x00])).toBe(0);
        expect(read([0x80, 0x01])).toBe(128);
    });

    it('regression: a varint past Number.MAX_SAFE_INTEGER is malformed, and MAX_SAFE_INTEGER itself round-trips', () => {
        expect(() => read([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f])).toThrow(/safe integer/);
        expect(() => read([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x10])).toThrow(/safe integer/); // exactly 2^53
        const w = new ByteWriter();
        w.varint(MAX_VARINT);
        expect(new ByteReader(w.finish()).varint()).toBe(Number.MAX_SAFE_INTEGER);
    });

    it('regression: a string table suffix longer than the section is malformed (422), not "too large" (413)', async () => {
        const w = new ByteWriter();
        w.varint(1); w.varint(0); w.varint(10_000_000); w.u8(0x61);
        const err = await decodePack(frameOf([[TAG.STR, w.finish()]]), { maxDecodedBytes: 1024 * 1024 }).catch((e: unknown) => e);
        expect(err).toBeInstanceOf(PackFormatError);
        expect(err).not.toBeInstanceOf(PackTooLargeError);
    });
});
