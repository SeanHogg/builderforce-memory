/**
 * knowledge/pack/sections/facts.ts — the FACT and REL sections of BFKP v1 (PRD 41 A11): the stored
 * rows, columnar, keys as STR indexes (sorted, so delta-coded), relation kind as an enum.
 */

import type { NodeRef, StoredFact, StoredRelation } from '../types.js';
import { assertConsumed, assertFlagsKnown, boundedCount, ByteReader, ByteWriter } from '../varint.js';
import { EDGE_KINDS, STORED_ORIGINS, readEnum, readWord, writeEnum, writeWord, type SectionCodec } from './columns.js';
import { TAG } from './rows.js';

const F_UPDATED = 1, F_BY_NULL = 2, F_BY = 4;
const A_LINE = 1, A_HASH = 2;

export const factSection: SectionCodec = {
    tag: TAG.FACT,
    write(pack, t) {
        const rows = pack.facts;
        if (!rows?.length) return undefined;
        const w = new ByteWriter();
        w.varint(rows.length);
        let prev = 0;
        for (const f of rows) { const i = t.str(f.key); w.varint(i - prev); prev = i; }
        for (const f of rows) w.varint(t.str(f.content));
        for (const f of rows) w.varint(t.str(f.source));
        for (const f of rows) w.num(f.importance);
        for (const f of rows) writeEnum(w, STORED_ORIGINS, f.origin);
        const flags = rows.map((f) => (f.updatedAt !== undefined ? F_UPDATED : 0) | (f.createdBy === null ? F_BY_NULL : 0) | (typeof f.createdBy === 'string' ? F_BY : 0));
        for (const f of flags) w.u8(f);
        rows.forEach((f, i) => {
            if (flags[i]! & F_UPDATED) w.varint(t.str(f.updatedAt!));
            if (flags[i]! & F_BY) w.varint(t.str(f.createdBy!));
        });
        for (const f of rows) w.varint(f.fileAnchors.length);
        for (const f of rows) for (const a of f.fileAnchors) w.varint(t.str(a.path));
        for (const f of rows) for (const a of f.fileAnchors) w.u8((a.line !== undefined ? A_LINE : 0) | (a.contentHash !== undefined ? A_HASH : 0));
        for (const f of rows) for (const a of f.fileAnchors) {
            if (a.line !== undefined) w.varint(a.line);
            if (a.contentHash !== undefined) w.varint(t.str(a.contentHash));
        }
        return w.finish();
    },
    read(bytes, t, limits, into) {
        const r = new ByteReader(bytes);
        const n = boundedCount(r, limits.budget, 'facts');
        const keys: string[] = [];
        let prev = 0;
        for (let i = 0; i < n; i++) { prev += r.varint(); keys.push(t.str(prev)); }
        const content = keys.map(() => t.str(r.varint()));
        const source = keys.map(() => t.str(r.varint()));
        const importance = keys.map(() => r.num());
        const origin = keys.map(() => readEnum(r, STORED_ORIGINS));
        const flags = keys.map(() => r.u8());
        for (const f of flags) assertFlagsKnown(f, F_UPDATED | F_BY_NULL | F_BY);
        const facts: StoredFact[] = keys.map((key, i) => ({ key, content: content[i]!, source: source[i]!, importance: importance[i]!, origin: origin[i]!, fileAnchors: [] }));
        facts.forEach((f, i) => {
            if (flags[i]! & F_UPDATED) f.updatedAt = t.str(r.varint());
            if (flags[i]! & F_BY_NULL) f.createdBy = null;
            if (flags[i]! & F_BY) f.createdBy = t.str(r.varint());
        });
        const counts = facts.map(() => boundedCount(r, limits.budget, 'anchors'));
        const total = counts.reduce((a, b) => a + b, 0);
        const paths = Array.from({ length: total }, () => t.str(r.varint()));
        const aflags = Array.from({ length: total }, () => r.u8());
        for (const f of aflags) assertFlagsKnown(f, A_LINE | A_HASH);
        let at = 0;
        facts.forEach((f, i) => {
            for (let k = 0; k < counts[i]!; k++, at++) {
                const a: StoredFact['fileAnchors'][number] = { path: paths[at]! };
                if (aflags[at]! & A_LINE) a.line = r.varint();
                if (aflags[at]! & A_HASH) a.contentHash = t.str(r.varint());
                f.fileAnchors.push(a);
            }
        });
        assertConsumed(r, 'facts section');
        into.facts = facts;
    },
};

const R_CONF = 1, R_HASH = 2, R_STALE = 4, R_STALE_NULL = 8;

export const relationSection: SectionCodec = {
    tag: TAG.REL,
    write(pack, t) {
        const rows = pack.relations;
        if (!rows?.length) return undefined;
        const w = new ByteWriter();
        w.varint(rows.length);
        let prev = 0;
        for (const x of rows) { const i = t.str(x.from); w.varint(i - prev); prev = i; }
        for (const x of rows) w.varint(t.str(x.to));
        for (const x of rows) writeWord(w, t, EDGE_KINDS, x.kind);
        for (const x of rows) w.num(x.weight);
        for (const x of rows) writeEnum(w, STORED_ORIGINS, x.origin);
        const flags = rows.map((x) => (x.confidence !== undefined ? R_CONF : 0) | (x.sourceHash !== undefined ? R_HASH : 0)
            | (typeof x.staleAt === 'string' ? R_STALE : 0) | (x.staleAt === null ? R_STALE_NULL : 0));
        for (const f of flags) w.u8(f);
        rows.forEach((x, i) => {
            if (flags[i]! & R_CONF) w.num(x.confidence!);
            if (flags[i]! & R_HASH) w.varint(t.hash(x.sourceHash!));
            if (flags[i]! & R_STALE) w.varint(t.str(x.staleAt!));
        });
        return w.finish();
    },
    read(bytes, t, limits, into) {
        const r = new ByteReader(bytes);
        const n = boundedCount(r, limits.budget, 'relations');
        const from: string[] = [];
        let prev = 0;
        for (let i = 0; i < n; i++) { prev += r.varint(); from.push(t.str(prev)); }
        const to = from.map(() => t.str(r.varint()));
        const kind = from.map(() => readWord(r, t, EDGE_KINDS));
        const weight = from.map(() => r.num());
        const origin = from.map(() => readEnum(r, STORED_ORIGINS));
        const flags = from.map(() => r.u8());
        for (const f of flags) assertFlagsKnown(f, R_CONF | R_HASH | R_STALE | R_STALE_NULL);
        into.relations = from.map((f0, i): StoredRelation => {
            const x: StoredRelation = { from: f0 as NodeRef, to: to[i]! as NodeRef, kind: kind[i]!, weight: weight[i]!, origin: origin[i]! };
            const f = flags[i]!;
            if (f & R_CONF) x.confidence = r.num();
            if (f & R_HASH) x.sourceHash = t.hash(r.varint());
            if (f & R_STALE) x.staleAt = t.str(r.varint());
            if (f & R_STALE_NULL) x.staleAt = null;
            return x;
        });
        assertConsumed(r, 'relations section');
    },
};
