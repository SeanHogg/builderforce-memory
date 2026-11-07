/**
 * knowledge/pack/encode.ts — `encodePack` and `normalizePack` (PRD 41 A11).
 *
 * Deterministic: rows are put in canonical order first, the tables are sorted, and every section
 * is written from the same normalized pack, so equal input (in any row order) gives equal bytes.
 * Two passes over the section writers: the first collects every string and hash they use, the
 * second writes real table indexes.
 *
 * NORMALISATION (lossless for every value the model produces, documented where it is not): rows are sorted and
 * empty parts dropped; object keys of badges and of the analysis document are written in sorted order; a
 * negative zero is written as zero; a string holding a lone UTF-16 surrogate is written with U+FFFD in its place
 * (UTF-8 cannot carry it). Equal normalised input gives equal bytes.
 */

import { deflateRaw } from './deflate.js';
import { FLAG_DEFLATED, writeFrame, writeSection } from './frame.js';
import { ROW_SECTIONS, TAG } from './sections/index.js';
import { compare } from './sections/columns.js';
import { collector, tablesOf, writeHashTable, writeStringTable } from './strings.js';
import type { Pack, RelationId } from './types.js';
import { ByteWriter } from './varint.js';

const nonEmpty = <T>(xs: readonly T[] | undefined): T[] | undefined => (xs && xs.length > 0 ? [...xs] : undefined);
const byRelation = (a: RelationId, b: RelationId): number => compare(a.from, b.from) || compare(a.to, b.to) || compare(a.kind, b.kind);

/** The hashes the row sections reference (a `have` request's explicit hashes are not among them). */
function referencedHashes(pack: Pack): Set<string> {
    const col = collector();
    for (const s of ROW_SECTIONS) s.write(pack, col);
    return col.hashes;
}

/** The pack in canonical form: rows sorted, empty parts dropped, `hashes` reduced to those no section references. */
export function normalizePack(pack: Pack): Pack {
    const out: Pack = { rootId: pack.rootId, baseGeneration: pack.baseGeneration, targetDigest: pack.targetDigest };
    const files = nonEmpty(pack.files)?.sort((a, b) => compare(a.path, b.path));
    if (files) out.files = files;
    const extracted = nonEmpty(pack.extracted)?.sort((a, b) => compare(a.hash, b.hash));
    if (extracted) out.extracted = extracted;
    const facts = nonEmpty(pack.facts)?.sort((a, b) => compare(a.key, b.key));
    if (facts) out.facts = facts;
    const relations = nonEmpty(pack.relations)?.sort((a, b) => byRelation(a, b));
    if (relations) out.relations = relations;
    const stale = nonEmpty(pack.stale)?.sort((a, b) => compare(a.sourceRef, b.sourceRef));
    if (stale) out.stale = stale;
    const d = pack.deletes;
    if (d && d.files.length + d.facts.length + d.relations.length > 0) {
        out.deletes = { files: [...d.files].sort(compare), facts: [...d.facts].sort(compare), relations: [...d.relations].sort(byRelation) };
    }
    if (pack.scope) out.scope = pack.scope;
    if (pack.analysis !== undefined) out.analysis = pack.analysis;
    const leaves = nonEmpty(pack.leaves)?.sort((a, b) => compare(a.key, b.key));
    if (leaves) out.leaves = leaves;
    if (pack.buckets) out.buckets = pack.buckets;
    const used = referencedHashes(out);
    const extra = [...new Set(pack.hashes ?? [])].filter((h) => !used.has(h)).sort(compare);
    if (extra.length > 0) out.hashes = extra;
    return out;
}

export interface EncodeOptions {
    /** Deflate the body (default true). Off, the bytes are the plain columnar form (tests, size comparisons). */
    deflate?: boolean;
    /** Filled with the size of the DECODED body: a client cuts a pack by it before the server's cap refuses it. */
    stats?: { decodedBytes: number };
}

export async function encodePack(input: Pack, opts: EncodeOptions = {}): Promise<Uint8Array> {
    const pack = normalizePack(input);
    const col = collector();
    for (const s of ROW_SECTIONS) s.write(pack, col);
    const tables = tablesOf(col.strings, [...col.hashes, ...(pack.hashes ?? [])]);

    const body = new ByteWriter();
    writeSection(body, TAG.STR, writeStringTable(tables.strings));
    writeSection(body, TAG.HASH, writeHashTable(tables.hashes));
    for (const s of ROW_SECTIONS) {
        const bytes = s.write(pack, tables);
        if (bytes) writeSection(body, s.tag, bytes);
    }
    const plain = body.finish();
    if (opts.stats) opts.stats.decodedBytes = plain.length;
    const deflate = opts.deflate ?? true;
    return writeFrame(
        { flags: deflate ? FLAG_DEFLATED : 0, rootId: pack.rootId, baseGeneration: pack.baseGeneration, targetDigest: pack.targetDigest },
        plain,
        deflate ? await deflateRaw(plain) : plain,
    );
}
