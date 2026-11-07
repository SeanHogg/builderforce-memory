/**
 * knowledge/pack/sections/extr.ts — the EXTR section of BFKP v1 (PRD 41 A11): the Extracted output per
 * content hash. A hash is either whole (`full`: nodes, edges and keys columnar) or a `diff` against the
 * prior hash's extraction of the same source (`prior hash#` plus the added and removed items).
 */

import type { Extracted, ExtrEntryDiff, ListDiff, PackExtr } from '../types.js';
import type { ReadTables, Tables } from '../strings.js';
import { boundedCount, ByteReader, ByteWriter, type Budget } from '../varint.js';
import type { SectionCodec } from './columns.js';
import { readEdges, readKeys, readNodes, writeEdges, writeKeys, writeNodes } from './extrLists.js';
import { TAG } from './rows.js';

const HAS_VERSION = 1;

function writeDiff<T>(w: ByteWriter, t: Tables, d: ListDiff<T>, items: (w: ByteWriter, t: Tables, xs: T[]) => void): void {
    w.varint(d.remove.length);
    for (const id of d.remove) w.varint(t.str(id));
    let prev = 0;
    w.varint(d.add.length);
    for (const a of d.add) { w.varint(a.at - prev); prev = a.at; }
    items(w, t, d.add.map((a) => a.item));
}

function readDiff<T>(r: ByteReader, t: ReadTables, budget: Budget, items: (r: ByteReader, t: ReadTables, budget: Budget) => T[]): ListDiff<T> {
    const remove = Array.from({ length: boundedCount(r, budget, 'diff removals') }, () => t.str(r.varint()));
    const n = boundedCount(r, budget, 'diff additions');
    const ats: number[] = [];
    let prev = 0;
    for (let i = 0; i < n; i++) { prev += r.varint(); ats.push(prev); }
    const list = items(r, t, budget);
    return { remove, add: ats.map((at, i) => ({ at, item: list[i]! })) };
}

export const extrSection: SectionCodec = {
    tag: TAG.EXTR,
    write(pack, t) {
        const rows = pack.extracted;
        if (!rows?.length) return undefined;
        const w = new ByteWriter();
        w.varint(rows.length);
        for (const x of rows) {
            w.u8(x.kind === 'full' ? 0 : 1);
            w.varint(t.hash(x.hash));
            if (x.kind === 'diff') w.varint(t.hash(x.prior));
            w.varint(x.entries.length);
            for (const e of x.entries) {
                w.varint(t.str(e.sourceRef));
                w.varint(t.hash(e.sourceHash));
                w.u8(e.extractorVersion !== undefined ? HAS_VERSION : 0);
                if (e.extractorVersion !== undefined) w.varint(e.extractorVersion);
                if (x.kind === 'full') {
                    const f = e as Extracted;
                    writeNodes(w, t, f.nodes); writeEdges(w, t, f.edges); writeKeys(w, t, f.keys);
                } else {
                    const d = e as ExtrEntryDiff;
                    writeDiff(w, t, d.nodes, writeNodes); writeDiff(w, t, d.edges, writeEdges); writeDiff(w, t, d.keys, writeKeys);
                }
            }
        }
        return w.finish();
    },
    read(bytes, t, limits, into) {
        const r = new ByteReader(bytes);
        const n = boundedCount(r, limits.budget, 'extracted hashes');
        const out: PackExtr[] = [];
        for (let i = 0; i < n; i++) {
            const diff = r.u8() === 1;
            const hash = t.hash(r.varint());
            const prior = diff ? t.hash(r.varint()) : '';
            const count = boundedCount(r, limits.budget, 'extracted entries');
            const entries: Array<Extracted | ExtrEntryDiff> = [];
            for (let k = 0; k < count; k++) {
                const sourceRef = t.str(r.varint());
                const sourceHash = t.hash(r.varint());
                const version = r.u8() & HAS_VERSION ? { extractorVersion: r.varint() } : {};
                if (!diff) {
                    entries.push({ sourceRef: sourceRef as Extracted['sourceRef'], sourceHash, ...version, nodes: readNodes(r, t, limits.budget), edges: readEdges(r, t, limits.budget), keys: readKeys(r, t, limits.budget) });
                } else {
                    entries.push({
                        sourceRef, sourceHash, ...version,
                        nodes: readDiff(r, t, limits.budget, readNodes), edges: readDiff(r, t, limits.budget, readEdges), keys: readDiff(r, t, limits.budget, readKeys),
                    });
                }
            }
            out.push(diff ? { kind: 'diff', hash, prior, entries: entries as ExtrEntryDiff[] } : { kind: 'full', hash, entries: entries as Extracted[] });
        }
        into.extracted = out;
    },
};
