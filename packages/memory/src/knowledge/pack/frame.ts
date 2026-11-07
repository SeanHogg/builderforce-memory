/**
 * knowledge/pack/frame.ts — the BFKP v1 frame (PRD 41 A11):
 *
 *   magic "BFKP" · version u8 · flags u8 (bit 0 = body deflated) · rootId · baseGeneration varint ·
 *   targetDigest (32 raw bytes) · body · trailer: CRC-32 (u32 LE) of the DECODED body
 *
 * The body is sections `{ tag u8, length varint, bytes }`; an unknown tag is skipped.
 * Also: the multi-part split a large pack travels in.
 */

import { crc32 } from './crc32.js';
import { ByteReader, ByteWriter, PackFormatError } from './varint.js';

export const MAGIC = [0x42, 0x46, 0x4b, 0x50] as const; // "BFKP"
export const VERSION = 1;
export const FLAG_DEFLATED = 1;
export const KPACK_CONTENT_TYPE = 'application/vnd.builderforce.kpack';
/** A compressed pack larger than this travels as several parts. */
export const MAX_PART_BYTES = 8 * 1024 * 1024;

export interface FrameHeader { flags: number; rootId: string; baseGeneration: number; targetDigest: string }

const hexOf = (b: Uint8Array): string => { let s = ''; for (const x of b) s += (x < 16 ? '0' : '') + x.toString(16); return s; };

export function writeFrame(h: FrameHeader, body: Uint8Array, storedBody: Uint8Array): Uint8Array {
    if (!/^[0-9a-f]{64}$/.test(h.targetDigest)) throw new PackFormatError('targetDigest is 64 lowercase hex characters');
    const w = new ByteWriter();
    for (const m of MAGIC) w.u8(m);
    w.u8(VERSION);
    w.u8(h.flags);
    w.str(h.rootId);
    w.varint(h.baseGeneration);
    const raw = new Uint8Array(32);
    for (let i = 0; i < 32; i++) raw[i] = parseInt(h.targetDigest.slice(i * 2, i * 2 + 2), 16);
    w.bytes(raw);
    w.bytes(storedBody);
    w.u32le(crc32(body));
    return w.finish();
}

/** Splits a frame into header fields, the stored (possibly deflated) body, and the CRC the trailer claims. */
export function readFrame(bytes: Uint8Array): { header: FrameHeader; stored: Uint8Array; crc: number } {
    if (bytes.length < 4 + 2 + 1 + 1 + 32 + 4) throw new PackFormatError('truncated frame');
    const r = new ByteReader(bytes);
    for (const m of MAGIC) if (r.u8() !== m) throw new PackFormatError('not a BFKP pack (bad magic)');
    const version = r.u8();
    if (version !== VERSION) throw new PackFormatError(`unsupported BFKP version ${version}`);
    const flags = r.u8();
    const rootId = r.str();
    const baseGeneration = r.varint();
    const targetDigest = hexOf(r.bytes(32));
    if (r.remaining < 4) throw new PackFormatError('truncated frame');
    const stored = r.bytes(r.remaining - 4);
    return { header: { flags, rootId, baseGeneration, targetDigest }, stored, crc: r.u32le() };
}

export function writeSection(w: ByteWriter, tag: number, bytes: Uint8Array): void {
    w.u8(tag);
    w.varint(bytes.length);
    w.bytes(bytes);
}

/** The sections of a decoded body by tag (a repeated tag keeps the last); unknown tags are returned too, for the caller to skip. */
export function readSections(body: Uint8Array): Map<number, Uint8Array> {
    const r = new ByteReader(body);
    const out = new Map<number, Uint8Array>();
    while (!r.done) { const tag = r.u8(); out.set(tag, r.bytes(r.varint())); }
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
