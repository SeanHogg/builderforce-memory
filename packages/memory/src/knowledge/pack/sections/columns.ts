/**
 * knowledge/pack/sections/columns.ts — helpers the BFKP sections share: enums that turn the model's
 * small closed vocabularies into one byte, zigzag deltas for index runs, and the section
 * descriptor type (PRD 41 A11).
 */

import type { Pack, PackLimits } from '../types.js';
import type { ReadTables, Tables } from '../strings.js';
import { PackFormatError, type ByteReader, type ByteWriter } from '../varint.js';

/** One section: its tag, how it is written from a pack (undefined = nothing to write) and read back into one. */
export interface SectionCodec {
    tag: number;
    write(pack: Pack, t: Tables): Uint8Array | undefined;
    read(bytes: Uint8Array, t: ReadTables, limits: PackLimits, into: Pack): void;
}

export const ORIGINS = ['extracted', 'model', 'human', 'agent'] as const;
export const STORED_ORIGINS = ['model', 'human', 'agent'] as const;
export const VERDICTS = ['allowed', 'same', 'skips', 'upward'] as const;
/** The relation/edge kinds the model names; anything else is stored as 0 + a string. */
export const EDGE_KINDS = [
    'references', 'links', 'contains', 'mentions', 'duplicate', 'near-duplicate', 'contradicts', 'agrees',
    'supersedes', 'about', 'co-changed',
] as const;
/** The node kinds the packs name. */
export const NODE_KINDS = ['file', 'section', 'symbol', 'area', 'entity', 'claim', 'fact'] as const;

export const zigzag = (n: number): number => (n >= 0 ? n * 2 : -n * 2 - 1);
export const unzigzag = (z: number): number => (z % 2 === 0 ? z / 2 : -(z + 1) / 2);

/** A closed vocabulary: the index + 1 of a known word, else 0 followed by the word's string index. */
export function writeWord(w: ByteWriter, t: Tables, vocab: readonly string[], word: string): void {
    const at = vocab.indexOf(word);
    if (at >= 0) w.varint(at + 1);
    else { w.varint(0); w.varint(t.str(word)); }
}

export function readWord(r: ByteReader, t: ReadTables, vocab: readonly string[]): string {
    const code = r.varint();
    if (code === 0) return t.str(r.varint());
    const word = vocab[code - 1];
    if (word === undefined) throw new PackFormatError('unknown vocabulary code');
    return word;
}

/** A closed enum that must be one of `vocab`. */
export function writeEnum(w: ByteWriter, vocab: readonly string[], value: string): void {
    const at = vocab.indexOf(value);
    if (at < 0) throw new PackFormatError(`value '${value}' is not one of ${vocab.join('|')}`);
    w.u8(at);
}

export function readEnum<T extends string>(r: ByteReader, vocab: readonly T[]): T {
    const v = vocab[r.u8()];
    if (v === undefined) throw new PackFormatError('unknown enum code');
    return v;
}

/** Strings sort the way the string table does, so a sorted run of rows has non-decreasing indexes. */
export const compare = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
