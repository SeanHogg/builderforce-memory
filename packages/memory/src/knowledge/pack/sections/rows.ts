/**
 * knowledge/pack/sections/rows.ts — the small row sections of BFKP v1 (PRD 41 A11): FILE
 * `(path#, hash#, bytes)`, STALE `(sourceRef#, hash#)`, DEL tombstones, SCOPE, ANALYSIS, and the
 * manifest sections LEAF (a reconcile answer) and BKT (the 4,096 bucket digests).
 * Rows arrive sorted (see `normalizePack`), so index runs are delta-encoded.
 */

import { assertConsumed, assertFlagsKnown, boundedCount, ByteReader, ByteWriter, PackFormatError } from '../varint.js';
import { fromHex64, toHex } from '../hex.js';
import type { Pack } from '../types.js';
import type { SectionCodec } from './columns.js';
import { MANIFEST_BUCKETS } from '../manifest.js';

export const TAG = { STR: 1, HASH: 2, FILE: 3, EXTR: 4, FACT: 5, REL: 6, STALE: 7, DEL: 8, SCOPE: 9, ANALYSIS: 10, LEAF: 11, BKT: 12, SITE: 13 } as const;

export const fileSection: SectionCodec = {
    tag: TAG.FILE,
    write(pack, t) {
        const rows = pack.files;
        if (!rows?.length) return undefined;
        const w = new ByteWriter();
        w.varint(rows.length);
        let prev = 0;
        for (const f of rows) { const i = t.str(f.path); w.varint(i - prev); prev = i; }
        for (const f of rows) w.varint(t.hash(f.hash));
        for (const f of rows) w.varint(f.bytes);
        return w.finish();
    },
    read(bytes, t, limits, into) {
        const r = new ByteReader(bytes);
        const n = boundedCount(r, limits.budget, 'files');
        const paths: string[] = [];
        let prev = 0;
        for (let i = 0; i < n; i++) { prev += r.varint(); paths.push(t.str(prev)); }
        const hashes = paths.map(() => t.hash(r.varint()));
        into.files = paths.map((path, i) => ({ path, hash: hashes[i]!, bytes: r.varint() }));
        assertConsumed(r, 'FILE section');
    },
};

export const staleSection: SectionCodec = {
    tag: TAG.STALE,
    write(pack, t) {
        const rows = pack.stale;
        if (!rows?.length) return undefined;
        const w = new ByteWriter();
        w.varint(rows.length);
        let prev = 0;
        for (const s of rows) { const i = t.str(s.sourceRef); w.varint(i - prev); prev = i; }
        for (const s of rows) w.varint(t.hash(s.hash));
        return w.finish();
    },
    read(bytes, t, limits, into) {
        const r = new ByteReader(bytes);
        const n = boundedCount(r, limits.budget, 'stale');
        const refs: string[] = [];
        let prev = 0;
        for (let i = 0; i < n; i++) { prev += r.varint(); refs.push(t.str(prev)); }
        into.stale = refs.map((sourceRef) => ({ sourceRef, hash: t.hash(r.varint()) }));
        assertConsumed(r, 'STALE section');
    },
};

export const deleteSection: SectionCodec = {
    tag: TAG.DEL,
    write(pack, t) {
        const d = pack.deletes;
        if (!d || d.files.length + d.facts.length + d.relations.length === 0) return undefined;
        const w = new ByteWriter();
        for (const list of [d.files, d.facts]) {
            w.varint(list.length);
            let prev = 0;
            for (const s of list) { const i = t.str(s); w.varint(i - prev); prev = i; }
        }
        w.varint(d.relations.length);
        for (const x of d.relations) w.varint(t.str(x.from));
        for (const x of d.relations) w.varint(t.str(x.to));
        for (const x of d.relations) w.varint(t.str(x.kind));
        return w.finish();
    },
    read(bytes, t, limits, into) {
        const r = new ByteReader(bytes);
        const lists: string[][] = [];
        for (let k = 0; k < 2; k++) {
            const n = boundedCount(r, limits.budget, 'tombstones');
            const out: string[] = [];
            let prev = 0;
            for (let i = 0; i < n; i++) { prev += r.varint(); out.push(t.str(prev)); }
            lists.push(out);
        }
        const n = boundedCount(r, limits.budget, 'relation tombstones');
        const from = Array.from({ length: n }, () => t.str(r.varint()));
        const to = Array.from({ length: n }, () => t.str(r.varint()));
        const kind = Array.from({ length: n }, () => t.str(r.varint()));
        assertConsumed(r, 'DEL section');
        into.deletes = { files: lists[0]!, facts: lists[1]!, relations: from.map((f, i) => ({ from: f, to: to[i]!, kind: kind[i]! })) };
    },
};

export const scopeSection: SectionCodec = {
    tag: TAG.SCOPE,
    write(pack, t) {
        const s = pack.scope;
        if (!s) return undefined;
        const w = new ByteWriter();
        w.varint(t.str(s.rootId));
        w.varint(t.str(s.title));
        w.varint(s.packs.length);
        for (const p of s.packs) w.varint(t.str(p));
        w.u8((s.lastDigest !== undefined ? 1 : 0) | (s.lastReviewedAt !== undefined ? 2 : 0));
        if (s.lastDigest !== undefined) w.varint(t.str(s.lastDigest));
        if (s.lastReviewedAt !== undefined) w.varint(t.str(s.lastReviewedAt));
        return w.finish();
    },
    read(bytes, t, limits, into) {
        const r = new ByteReader(bytes);
        const scope: NonNullable<Pack['scope']> = { rootId: t.str(r.varint()), title: t.str(r.varint()), packs: [] };
        const n = boundedCount(r, limits.budget, 'packs');
        for (let i = 0; i < n; i++) scope.packs.push(t.str(r.varint()));
        const flags = r.u8();
        assertFlagsKnown(flags, 3);
        if (flags & 1) scope.lastDigest = t.str(r.varint());
        if (flags & 2) scope.lastReviewedAt = t.str(r.varint());
        assertConsumed(r, 'SCOPE section');
        into.scope = scope;
    },
};

/** Object keys in sorted order, recursively: equal documents become equal bytes however their keys were built. */
const canonical = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(canonical);
    if (v && typeof v === 'object') return Object.fromEntries(Object.keys(v).sort().map((k) => [k, canonical((v as Record<string, unknown>)[k])]));
    return v;
};

/** The analysis snapshot is one JSON-compatible document with no row structure; it travels as one string (deflate does the rest). */
export const analysisSection: SectionCodec = {
    tag: TAG.ANALYSIS,
    write(pack, t) {
        if (pack.analysis === undefined) return undefined;
        const w = new ByteWriter();
        w.varint(t.str(JSON.stringify(canonical(pack.analysis))));
        return w.finish();
    },
    read(bytes, t, _limits, into) {
        const r = new ByteReader(bytes);
        const text = t.str(r.varint());
        assertConsumed(r, 'ANALYSIS section');
        into.analysis = JSON.parse(text) as unknown;
    },
};

export const leafSection: SectionCodec = {
    tag: TAG.LEAF,
    write(pack, t) {
        const rows = pack.leaves;
        if (!rows?.length) return undefined;
        const w = new ByteWriter();
        w.varint(rows.length);
        let prev = 0;
        for (const l of rows) { const i = t.str(l.key); w.varint(i - prev); prev = i; }
        for (const l of rows) w.varint(t.hash(l.digest));
        return w.finish();
    },
    read(bytes, t, limits, into) {
        const r = new ByteReader(bytes);
        const n = boundedCount(r, limits.budget, 'leaves');
        const keys: string[] = [];
        let prev = 0;
        for (let i = 0; i < n; i++) { prev += r.varint(); keys.push(t.str(prev)); }
        into.leaves = keys.map((key) => ({ key, digest: t.hash(r.varint()) }));
        assertConsumed(r, 'LEAF section');
    },
};

/** The bucket digests, raw: 4,096 x 32 bytes. */
export const bucketSection: SectionCodec = {
    tag: TAG.BKT,
    write(pack) {
        const b = pack.buckets;
        if (!b) return undefined;
        if (b.length !== MANIFEST_BUCKETS) throw new PackFormatError(`a bucket list holds ${MANIFEST_BUCKETS} digests`);
        const w = new ByteWriter();
        for (const hex of b) w.bytes(fromHex64(hex));
        return w.finish();
    },
    read(bytes, _t, _limits, into) {
        if (bytes.length !== MANIFEST_BUCKETS * 32) throw new PackFormatError('bucket section has the wrong size');
        const out: string[] = [];
        for (let i = 0; i < MANIFEST_BUCKETS; i++) out.push(toHex(bytes.subarray(i * 32, i * 32 + 32)));
        into.buckets = out;
    },
};
