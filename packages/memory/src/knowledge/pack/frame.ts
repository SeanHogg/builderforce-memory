/**
 * knowledge/pack/frame.ts — the BFKP v1 frame (PRD 41 A11):
 *
 *   magic "BFKP" · version u8 · flags u8 (bit 0 = body deflated; any other bit is refused) · rootId · baseGeneration varint ·
 *   targetDigest (32 raw bytes) · body · trailer: CRC-32 (u32 LE) of the DECODED body
 *
 * The body is sections `{ tag u8, length varint, bytes }`; an unknown tag is skipped, a repeated tag is refused, and a known
 * section must be read to its last byte.
 *
 * EVOLVING THE FORMAT: known sections are CLOSED. A reader refuses trailing bytes in a known section and any flag bit, kind or
 * version byte it does not know, so a writer can never append a field to an existing section or set a new bit in it. A new field
 * goes in a NEW section tag (older readers skip it); a change to how the body itself is read (a new compression) is a new frame
 * VERSION. A new value in an existing closed vocabulary needs every reader upgraded first.
 * Also: the multi-part split a large pack travels in.
 */

import { crc32 } from './crc32.js';
import { fromHex64, HEX64, toHex } from './hex.js';
import { assertFlagsKnown, ByteReader, ByteWriter, PackFormatError } from './varint.js';

export const MAGIC = [0x42, 0x46, 0x4b, 0x50] as const; // "BFKP"
export const VERSION = 1;
export const FLAG_DEFLATED = 1;
const KNOWN_FLAGS = FLAG_DEFLATED;
export const KPACK_CONTENT_TYPE = 'application/vnd.builderforce.kpack';
/** A compressed pack larger than this travels as several parts. */
export const MAX_PART_BYTES = 8 * 1024 * 1024;

export interface FrameHeader { flags: number; rootId: string; baseGeneration: number; targetDigest: string }

/**
 * The CRC-32 in the trailer covers the HEADER (magic through digest) and the DECODED body, so a flipped
 * root id, generation or digest fails the same check a flipped row does.
 */
export function writeFrame(h: FrameHeader, body: Uint8Array, storedBody: Uint8Array): Uint8Array {
    if (!HEX64.test(h.targetDigest)) throw new PackFormatError('targetDigest is 64 lowercase hex characters');
    const w = new ByteWriter();
    for (const m of MAGIC) w.u8(m);
    w.u8(VERSION);
    w.u8(h.flags);
    w.str(h.rootId);
    w.varint(h.baseGeneration);
    w.bytes(fromHex64(h.targetDigest));
    const head = w.finish();
    const out = new ByteWriter();
    out.bytes(head);
    out.bytes(storedBody);
    out.u32le(crc32(body, crc32(head)));
    return out.finish();
}

const MIN_FRAME = 4 + 2 + 1 + 1 + 32 + 4;

/** The header alone: magic through digest. `bytes` may be just the first part of a split pack. */
export function readHeader(bytes: Uint8Array): { header: FrameHeader; headerBytes: Uint8Array } {
    const r = new ByteReader(bytes);
    for (const m of MAGIC) if (r.u8() !== m) throw new PackFormatError('not a BFKP pack (bad magic)');
    const version = r.u8();
    if (version !== VERSION) throw new PackFormatError(`unsupported BFKP version ${version}`);
    const flags = r.u8();
    // A flag this reader does not know changes how the body must be read (a new compression, say): refuse, never guess.
    assertFlagsKnown(flags, KNOWN_FLAGS);
    const rootId = r.str();
    const baseGeneration = r.varint();
    const targetDigest = toHex(r.bytes(32));
    return { header: { flags, rootId, baseGeneration, targetDigest }, headerBytes: bytes.subarray(0, r.offset) };
}

/**
 * Splits a frame into header fields, the stored (possibly deflated) body, the header's bytes, and the CRC the
 * trailer claims. The caller checks `crc32(decodedBody, crc32(headerBytes)) === crc` once it has inflated.
 */
export function readFrame(bytes: Uint8Array): { header: FrameHeader; headerBytes: Uint8Array; stored: Uint8Array; crc: number } {
    if (bytes.length < MIN_FRAME) throw new PackFormatError('truncated frame');
    const { header, headerBytes } = readHeader(bytes);
    const r = new ByteReader(bytes);
    r.bytes(headerBytes.length);
    if (r.remaining < 4) throw new PackFormatError('truncated frame');
    const stored = r.bytes(r.remaining - 4);
    return { header, headerBytes, stored, crc: r.u32le() };
}

/** Parts that can be read as a frame without joining: a first part that holds the whole header and a last part that holds the trailer. */
const HEADER_ROOM = 4096;
export function readFrameParts(parts: readonly Uint8Array[]): { header: FrameHeader; headerBytes: Uint8Array; stored: Uint8Array[]; crc: number } {
    const first = parts[0], last = parts[parts.length - 1];
    if (parts.length < 2 || !first || !last || first.length < HEADER_ROOM || last.length < 4) return singleFrame(parts);
    const { header, headerBytes } = readHeader(first);
    const trailer = last.subarray(last.length - 4);
    const stored = parts.map((p, i) => {
        const from = i === 0 ? headerBytes.length : 0;
        const to = i === parts.length - 1 ? p.length - 4 : p.length;
        return p.subarray(from, to);
    });
    const crc = (trailer[0]! | (trailer[1]! << 8) | (trailer[2]! << 16) | (trailer[3]! << 24)) >>> 0;
    return { header, headerBytes, stored, crc };
}

function singleFrame(parts: readonly Uint8Array[]): { header: FrameHeader; headerBytes: Uint8Array; stored: Uint8Array[]; crc: number } {
    const f = readFrame(joinParts(parts));
    return { ...f, stored: [f.stored] };
}

export function writeSection(w: ByteWriter, tag: number, bytes: Uint8Array): void {
    w.u8(tag);
    w.varint(bytes.length);
    w.bytes(bytes);
}

/**
 * The sections of a decoded body by tag; unknown tags are returned too, for the caller to skip. A tag that appears twice is a
 * malformed pack: which copy wins would be a guess, and the writer never repeats one.
 */
export function readSections(body: Uint8Array): Map<number, Uint8Array> {
    const r = new ByteReader(body);
    const out = new Map<number, Uint8Array>();
    while (!r.done) {
        const tag = r.u8();
        if (out.has(tag)) throw new PackFormatError(`section tag ${tag} appears twice`);
        out.set(tag, r.bytes(r.varint()));
    }
    return out;
}

/** Cuts a pack into ordered parts of at most `max` bytes. A pack within `max` is one part. */
export function splitParts(pack: Uint8Array, max: number = MAX_PART_BYTES): Uint8Array[] {
    if (pack.length <= max) return [pack];
    const parts: Uint8Array[] = [];
    for (let at = 0; at < pack.length; at += max) parts.push(pack.subarray(at, Math.min(at + max, pack.length)));
    return parts;
}

/** The inverse of {@link splitParts}. */
export function joinParts(parts: readonly Uint8Array[]): Uint8Array {
    if (parts.length === 1) return parts[0]!;
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) { out.set(p, at); at += p.length; }
    return out;
}
