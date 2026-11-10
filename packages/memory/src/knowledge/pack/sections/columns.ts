/**
 * knowledge/pack/sections/columns.ts — helpers the BFKP sections share: enums that turn the model's
 * small closed vocabularies into one byte, zigzag deltas for index runs, and the section
 * descriptor type (PRD 41 A11).
 */

import type { Pack, ReadLimits } from '../types.js';
import type { ReadTables, Tables } from '../strings.js';
import { PackFormatError, type ByteReader, type ByteWriter } from '../varint.js';

/** One section: its tag, how it is written from a pack (undefined = nothing to write) and read back into one. */
export interface SectionCodec {
    tag: number;
    write(pack: Pack, t: Tables): Uint8Array | undefined;
    read(bytes: Uint8Array, t: ReadTables, limits: ReadLimits, into: Pack): void;
}

export const ORIGINS = ['extracted', 'model', 'human', 'agent'] as const;
export const STORED_ORIGINS = ['model', 'human', 'agent'] as const;
export const VERDICTS = ['allowed', 'same', 'skips', 'upward'] as const;
/**
 * The relation/edge kinds a READER knows as one-byte codes; anything else is stored as 0 + a string. APPEND ONLY: a
 * code is its index + 1, so an existing word never moves.
 *
 * Expand/contract across releases (frame.ts: "a new value in an existing closed vocabulary needs every reader
 * upgraded first"): a reader before 2026-11-10 throws on a code past 11, so the symbol kinds (PRD 43 P0b) are KNOWN
 * here but not yet WRITTEN as codes — see {@link WRITTEN_EDGE_KINDS}.
 */
export const EDGE_KINDS = [
    'references', 'links', 'contains', 'mentions', 'duplicate', 'near-duplicate', 'contradicts', 'agrees',
    'supersedes', 'about', 'co-changed',
    // PRD 43 P0b (2026-11-10): the symbol-level kinds, in SYMBOL_EDGE_KINDS order.
    'extends', 'implements', 'composes', 'uses', 'calls',
] as const;
/**
 * The prefix of {@link EDGE_KINDS} a WRITER codes: what every deployed reader decodes. A later kind is written as 0 + its
 * string, which every reader (old or new) decodes to the same value. Grow it to the whole list once no reader older
 * than this release remains (the contract step) — the bytes change, the decoded pack does not.
 */
export const WRITTEN_EDGE_KINDS: readonly string[] = EDGE_KINDS.slice(0, 11);
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
