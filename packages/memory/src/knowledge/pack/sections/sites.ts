/**
 * knowledge/pack/sections/sites.ts — the SITE section of BFKP v1 (PRD 43 P0b, B1a rule 10): the call sites of the
 * `calls` edges inside EXTR. EXTR is a CLOSED section (frame.ts), so `KEdge.sites` cannot be a new edge flag there:
 * it travels here, as a NEW tag that a reader older than this release skips — it then decodes every edge, kind and
 * weight unchanged and only the sites are absent.
 *
 *   count varint · per edge that has sites: row varint · entry varint · edge varint · n varint · n × (path STR · line · column · order)
 *
 * `row` indexes the (normalized) `extracted` list, `entry` that row's entries, `edge` the entry's edges — for a diff
 * entry, its `edges.add` items. Positions strictly increase in (row, entry, edge) order, so a site group is attached
 * once. Read after EXTR (ROW_SECTIONS order).
 *
 * ADVISORY, never fatal: sites only draw a sequence, so a bad SITE section must not cost the pack it rides in. A read
 * decodes the whole section into scratch first and attaches nothing when any part of it is malformed (a position EXTR
 * did not decode, a repeat, an empty list, a bad string index, trailing bytes, the row budget) — the pack decodes as an
 * older reader would see it, without sites. A write drops an invalid site (not a non-negative integer line/column/order,
 * no string path) instead of refusing the pack. An empty `sites` list is not written (and so is absent after a decode).
 */

import type { CallSite, ExtrEntryDiff, Extracted, KEdge, Pack } from '../types.js';
import { boundedCount, ByteReader, ByteWriter } from '../varint.js';
import type { SectionCodec } from './columns.js';
import { TAG } from './rows.js';

/** The edges of one entry, in wire order: a full entry's list, or a diff entry's added items. */
function edgesOf(row: NonNullable<Pack['extracted']>[number], entry: Extracted | ExtrEntryDiff): KEdge[] {
    return row.kind === 'full' ? (entry as Extracted).edges : (entry as ExtrEntryDiff).edges.add.map((a) => a.item);
}

const isCount = (n: unknown): n is number => typeof n === 'number' && Number.isInteger(n) && n >= 0 && n <= Number.MAX_SAFE_INTEGER;

/** A site the wire can carry. */
export const isValidSite = (s: CallSite): boolean =>
    !!s && typeof s.path === 'string' && isCount(s.line) && isCount(s.column) && isCount(s.order);

export const siteSection: SectionCodec = {
    tag: TAG.SITE,
    write(pack, t) {
        const groups: Array<{ row: number; entry: number; edge: number; sites: readonly CallSite[] }> = [];
        (pack.extracted ?? []).forEach((row, ri) => {
            row.entries.forEach((entry, ei) => {
                edgesOf(row, entry).forEach((e, xi) => {
                    const sites = e.sites?.filter(isValidSite);
                    if (sites?.length) groups.push({ row: ri, entry: ei, edge: xi, sites });
                });
            });
        });
        if (groups.length === 0) return undefined;
        const w = new ByteWriter();
        w.varint(groups.length);
        for (const g of groups) {
            w.varint(g.row); w.varint(g.entry); w.varint(g.edge);
            w.varint(g.sites.length);
            for (const s of g.sites) { w.varint(t.str(s.path)); w.varint(s.line); w.varint(s.column); w.varint(s.order); }
        }
        return w.finish();
    },
    read(bytes, t, limits, into) {
        const scratch: Array<{ edge: KEdge; sites: CallSite[] }> = [];
        try {
            const r = new ByteReader(bytes);
            const n = boundedCount(r, limits.budget, 'call-site groups');
            const rows = into.extracted ?? [];
            let last: [number, number, number] = [-1, -1, -1];
            for (let i = 0; i < n; i++) {
                const ri = r.varint(), ei = r.varint(), xi = r.varint();
                // (row, entry, edge) strictly increasing: a group is attached once, in order.
                const after = ri > last[0] || (ri === last[0] && (ei > last[1] || (ei === last[1] && xi > last[2])));
                if (!after) return;
                last = [ri, ei, xi];
                const row = rows[ri];
                const entry = row?.entries[ei];
                const edge = row && entry ? edgesOf(row, entry)[xi] : undefined;
                if (!edge) return;
                const count = boundedCount(r, limits.budget, 'call sites');
                if (count === 0) return;
                const sites: CallSite[] = [];
                for (let s = 0; s < count; s++) sites.push({ path: t.str(r.varint()), line: r.varint(), column: r.varint(), order: r.varint() });
                scratch.push({ edge, sites });
            }
            if (!r.done) return;
        } catch {
            return; // malformed: the pack decodes without sites
        }
        for (const { edge, sites } of scratch) edge.sites = sites;
    },
};
