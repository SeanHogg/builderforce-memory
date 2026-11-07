/**
 * knowledge/pack/strings.ts — the STR and HASH sections of BFKP v1 (PRD 41 A11).
 *
 *   STR   every path, key, ref, symbol name and label once, sorted and front-coded: for each
 *         string the length of the byte prefix it shares with the one before, then the suffix.
 *   HASH  every content hash once, sorted. A 64-character lowercase hex hash is stored as its raw
 *         32 bytes (flag 0); any other id (a root-level `~tag:…` output) as UTF-8 (flag 1).
 *
 * Sections refer to both by index. Encoding runs each section encoder twice (see encode.ts): a
 * first pass with {@link collector} gathers what is used, a second with {@link tablesOf} writes real indexes.
 */

import { boundedCount, ByteReader, ByteWriter, PackFormatError, utf8 } from './varint.js';

/** What a section encoder needs: the index of a string and of a hash. */
export interface Tables {
    str(s: string): number;
    hash(h: string): number;
}

/** What a section decoder needs: the string and hash at an index. */
export interface ReadTables {
    str(i: number): string;
    hash(i: number): string;
}

const HEX64 = /^[0-9a-f]{64}$/;

/** A pass-1 `Tables` that records every string and hash it is asked for. */
export function collector(): Tables & { strings: Set<string>; hashes: Set<string> } {
    const strings = new Set<string>(), hashes = new Set<string>();
    return { strings, hashes, str: (s) => { strings.add(s); return 0; }, hash: (h) => { hashes.add(h); return 0; } };
}

export interface BuiltTables extends Tables { strings: string[]; hashes: string[] }

/** The real tables over what pass 1 collected: sorted, so equal input gives equal bytes. */
export function tablesOf(strings: Iterable<string>, hashes: Iterable<string>): BuiltTables {
    const s = [...new Set(strings)].sort(), h = [...new Set(hashes)].sort();
    const si = new Map(s.map((v, i) => [v, i])), hi = new Map(h.map((v, i) => [v, i]));
    return { strings: s, hashes: h, str: (v) => si.get(v)!, hash: (v) => hi.get(v)! };
}

function commonPrefix(a: Uint8Array, b: Uint8Array): number {
    const n = Math.min(a.length, b.length);
    let i = 0;
    while (i < n && a[i] === b[i]) i++;
    return i;
}

export function writeStringTable(strings: readonly string[]): Uint8Array {
    const w = new ByteWriter();
    w.varint(strings.length);
    let prev: Uint8Array = new Uint8Array(0);
    for (const s of strings) {
        const b = utf8(s);
        const shared = commonPrefix(prev, b);
        w.varint(shared);
        w.varint(b.length - shared);
        w.bytes(b.subarray(shared));
        prev = b;
    }
    return w.finish();
}

export function readStringTable(bytes: Uint8Array, maxStrings: number): string[] {
    const r = new ByteReader(bytes);
    const n = boundedCount(r, maxStrings, 'string table');
    const dec = new TextDecoder('utf-8', { fatal: true });
    const out: string[] = [];
    let prev: Uint8Array = new Uint8Array(0);
    for (let i = 0; i < n; i++) {
        const shared = r.varint(), rest = r.varint();
        if (shared > prev.length) throw new PackFormatError('string table: prefix longer than the previous string');
        const b = new Uint8Array(shared + rest);
        b.set(prev.subarray(0, shared));
        b.set(r.bytes(rest), shared);
        try { out.push(dec.decode(b)); } catch { throw new PackFormatError('string table: invalid UTF-8'); }
        prev = b;
    }
    return out;
}

export function writeHashTable(hashes: readonly string[]): Uint8Array {
    const w = new ByteWriter();
    w.varint(hashes.length);
    for (const h of hashes) {
        if (HEX64.test(h)) {
            w.u8(0);
            const raw = new Uint8Array(32);
            for (let i = 0; i < 32; i++) raw[i] = parseInt(h.slice(i * 2, i * 2 + 2), 16);
            w.bytes(raw);
        } else {
            w.u8(1);
            w.str(h);
        }
    }
    return w.finish();
}

export function readHashTable(bytes: Uint8Array, maxHashes: number): string[] {
    const r = new ByteReader(bytes);
    const n = boundedCount(r, maxHashes, 'hash table');
    const out: string[] = [];
    for (let i = 0; i < n; i++) {
        const flag = r.u8();
        if (flag === 0) {
            let hex = '';
            for (const b of r.bytes(32)) hex += (b < 16 ? '0' : '') + b.toString(16);
            out.push(hex);
        } else if (flag === 1) out.push(r.str());
        else throw new PackFormatError('hash table: unknown entry flag');
    }
    return out;
}

/** The read-side lookup over decoded tables; an out-of-range index is a malformed pack. */
export function readTables(strings: readonly string[], hashes: readonly string[]): ReadTables {
    return {
        str: (i) => { const v = strings[i]; if (v === undefined) throw new PackFormatError('string index out of range'); return v; },
        hash: (i) => { const v = hashes[i]; if (v === undefined) throw new PackFormatError('hash index out of range'); return v; },
    };
}
