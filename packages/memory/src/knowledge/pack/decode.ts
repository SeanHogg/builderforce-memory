/**
 * knowledge/pack/decode.ts — `decodePack` (PRD 41 A11). Guards run before any row is built: the
 * magic and version, the CRC-32 of the header and the decoded body, the inflate cap (`PackTooLargeError` past
 * `maxDecodedBytes`) and the string/row counts. A malformed pack is a `PackFormatError`.
 */

import { crc32 } from './crc32.js';
import { inflateRawChunks } from './deflate.js';
import { FLAG_DEFLATED, joinParts, readFrameParts, readSections } from './frame.js';
import { ROW_SECTIONS, TAG } from './sections/index.js';
import { readHashTable, readStringTable, readTables } from './strings.js';
import { DEFAULT_PACK_LIMITS, PackTooLargeError, type Pack, type PackLimits, type ReadLimits } from './types.js';
import { PackFormatError } from './varint.js';

const NONE = new Uint8Array([0]);
const partsOf = (input: Uint8Array | readonly Uint8Array[]): readonly Uint8Array[] => (input instanceof Uint8Array ? [input] : input);

/** The frame's header, without inflating the body (cheap: for routing and the generation check). */
export function peekPack(input: Uint8Array | readonly Uint8Array[]): { rootId: string; baseGeneration: number; targetDigest: string } {
    const { header } = readFrameParts(partsOf(input));
    return { rootId: header.rootId, baseGeneration: header.baseGeneration, targetDigest: header.targetDigest };
}

/**
 * Decodes one pack, given whole or as the ordered parts it travelled in. Parts of a deflated pack are inflated as one
 * stream, so the compressed pack is never copied into a second joined buffer.
 */
export async function decodePack(input: Uint8Array | readonly Uint8Array[], limits: Partial<PackLimits> = {}): Promise<Pack> {
    const base0: PackLimits = { ...DEFAULT_PACK_LIMITS, ...limits };
    // ONE row budget for the whole pack: every list of every section draws from it.
    const lim: ReadLimits = { ...base0, budget: { left: base0.maxRows } };
    const { header, headerBytes, stored, crc } = readFrameParts(partsOf(input));
    let body: Uint8Array;
    if (header.flags & FLAG_DEFLATED) {
        try {
            body = await inflateRawChunks(stored, lim.maxDecodedBytes);
        } catch (error) {
            if (error instanceof PackTooLargeError) throw error;
            throw new PackFormatError('corrupt deflate stream');
        }
    } else {
        const size = stored.reduce((n, c) => n + c.length, 0);
        if (size > lim.maxDecodedBytes) throw new PackTooLargeError(lim.maxDecodedBytes);
        body = stored.length === 1 ? stored[0]! : joinParts(stored);
    }
    if (crc32(body, crc32(headerBytes)) !== crc) throw new PackFormatError('CRC-32 mismatch');
    try {
        const sections = readSections(body);
        const strings = readStringTable(sections.get(TAG.STR) ?? NONE, lim.maxStrings, lim.maxDecodedBytes);
        const hashes = readHashTable(sections.get(TAG.HASH) ?? NONE, lim.maxStrings);
        const used = new Set<string>();
        const base = readTables(strings, hashes);
        const tables = { str: base.str, hash: (i: number) => { const h = base.hash(i); used.add(h); return h; } };
        const pack: Pack = { rootId: header.rootId, baseGeneration: header.baseGeneration, targetDigest: header.targetDigest };
        for (const s of ROW_SECTIONS) {
            const bytes = sections.get(s.tag);
            if (bytes) s.read(bytes, tables, lim, pack);
        }
        const extra = hashes.filter((h) => !used.has(h));
        if (extra.length > 0) pack.hashes = extra;
        return pack;
    } catch (error) {
        if (error instanceof PackFormatError || error instanceof PackTooLargeError) throw error;
        throw new PackFormatError(error instanceof Error ? error.message : 'malformed pack');
    }
}
