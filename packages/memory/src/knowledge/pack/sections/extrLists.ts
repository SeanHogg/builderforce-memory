/**
 * knowledge/pack/sections/extrLists.ts — the three lists an Extracted entry holds (nodes, edges,
 * keys), written column by column (PRD 41 A11): kinds as small codes, names as STR indexes, the
 * ref index run zigzag-delta coded, optional fields behind a flag word.
 */

import type { KEdge, KNode, NodeRef } from '../types.js';
import type { ReadTables, Tables } from '../strings.js';
import { assertFlagsKnown, boundedCount, ByteReader, ByteWriter, PackFormatError, type Budget } from '../varint.js';
import { EDGE_KINDS, NODE_KINDS, ORIGINS, VERDICTS, readEnum, readWord, unzigzag, writeEnum, writeWord, zigzag } from './columns.js';

const N_PARTITION = 1, N_AREA = 2, N_STALE = 4, N_STALE_V = 8, N_BADGES = 16, N_SYSTEM = 32, N_APP = 64, N_COPY = 128, N_HASH = 256;
const N_KNOWN = 0x1ff;

export function writeNodes(w: ByteWriter, t: Tables, nodes: readonly KNode[]): void {
    w.varint(nodes.length);
    let prev = 0;
    for (const n of nodes) { const i = t.str(n.ref); w.varint(zigzag(i - prev)); prev = i; }
    for (const n of nodes) writeWord(w, t, NODE_KINDS, n.kind);
    for (const n of nodes) w.varint(t.str(n.label));
    for (const n of nodes) w.varint(t.str(n.pack));
    for (const n of nodes) w.num(n.importance);
    for (const n of nodes) writeEnum(w, ORIGINS, n.origin);
    const flags = nodes.map((n) =>
        (n.partition !== undefined ? N_PARTITION : 0) | (n.area !== undefined ? N_AREA : 0)
        | (n.stale !== undefined ? N_STALE : 0) | (n.stale ? N_STALE_V : 0) | (n.badges !== undefined ? N_BADGES : 0)
        | (n.system !== undefined ? N_SYSTEM : 0) | (n.app !== undefined ? N_APP : 0) | (n.copyOf !== undefined ? N_COPY : 0)
        | (n.sourceHash !== undefined ? N_HASH : 0));
    for (const f of flags) w.varint(f);
    nodes.forEach((n, i) => {
        const f = flags[i]!;
        if (f & N_PARTITION) w.varint(t.str(n.partition!));
        if (f & N_AREA) w.varint(t.str(n.area!));
        if (f & N_SYSTEM) w.varint(t.str(n.system!));
        if (f & N_APP) w.varint(t.str(n.app!));
        if (f & N_COPY) w.varint(t.str(n.copyOf!));
        if (f & N_HASH) w.varint(t.hash(n.sourceHash!));
        if (f & N_BADGES) {
            const keys = Object.keys(n.badges!).sort(); // canonical order: equal badges, equal bytes
            w.varint(keys.length);
            for (const k of keys) {
                w.varint(t.str(k));
                const v = n.badges![k]!;
                if (typeof v === 'number') { w.u8(0); w.num(v); } else { w.u8(1); w.varint(t.str(v)); }
            }
        }
    });
}

export function readNodes(r: ByteReader, t: ReadTables, budget: Budget): KNode[] {
    const n = boundedCount(r, budget, 'nodes');
    const refs: string[] = [];
    let prev = 0;
    for (let i = 0; i < n; i++) { prev += unzigzag(r.varint()); refs.push(t.str(prev)); }
    const kinds = Array.from({ length: n }, () => readWord(r, t, NODE_KINDS));
    const labels = Array.from({ length: n }, () => t.str(r.varint()));
    const packs = Array.from({ length: n }, () => t.str(r.varint()));
    const importance = Array.from({ length: n }, () => r.num());
    const origins = Array.from({ length: n }, () => readEnum(r, ORIGINS));
    const flags = Array.from({ length: n }, () => r.varint());
    for (const f of flags) assertFlagsKnown(f, N_KNOWN);
    return refs.map((ref, i): KNode => {
        const f = flags[i]!;
        const node: KNode = { ref: ref as NodeRef, kind: kinds[i]!, label: labels[i]!, pack: packs[i]!, importance: importance[i]!, origin: origins[i]! };
        if (f & N_PARTITION) node.partition = t.str(r.varint());
        if (f & N_AREA) node.area = t.str(r.varint()) as NodeRef;
        if (f & N_SYSTEM) node.system = t.str(r.varint());
        if (f & N_APP) node.app = t.str(r.varint());
        if (f & N_COPY) node.copyOf = t.str(r.varint());
        if (f & N_HASH) node.sourceHash = t.hash(r.varint());
        if (f & N_STALE) node.stale = (f & N_STALE_V) !== 0;
        if (f & N_BADGES) {
            const badges: Record<string, number | string> = {};
            const count = boundedCount(r, budget, 'badges');
            for (let b = 0; b < count; b++) {
                const key = t.str(r.varint());
                const type = r.u8();
                if (type > 1) throw new PackFormatError('unknown badge value type');
                badges[key] = type === 0 ? r.num() : t.str(r.varint());
            }
            node.badges = badges;
        }
        return node;
    });
}

const E_CONF = 1, E_VERDICT = 2, E_CYCLE = 4, E_CYCLE_V = 8, E_STALE = 16, E_STALE_V = 32, E_BYNAME = 64;
const E_KNOWN = 0x7f;

export function writeEdges(w: ByteWriter, t: Tables, edges: readonly KEdge[]): void {
    w.varint(edges.length);
    let prev = 0;
    for (const e of edges) { const i = t.str(e.from); w.varint(zigzag(i - prev)); prev = i; }
    for (const e of edges) w.varint(t.str(e.to));
    for (const e of edges) writeWord(w, t, EDGE_KINDS, e.kind);
    for (const e of edges) w.num(e.weight);
    for (const e of edges) writeEnum(w, ORIGINS, e.origin);
    const flags = edges.map((e) =>
        (e.confidence !== undefined ? E_CONF : 0) | (e.verdict !== undefined ? E_VERDICT : 0)
        | (e.cycle !== undefined ? E_CYCLE : 0) | (e.cycle ? E_CYCLE_V : 0)
        | (e.stale !== undefined ? E_STALE : 0) | (e.stale ? E_STALE_V : 0) | (e.byName !== undefined ? E_BYNAME : 0));
    for (const f of flags) w.varint(f);
    edges.forEach((e, i) => {
        const f = flags[i]!;
        if (f & E_CONF) w.num(e.confidence!);
        if (f & E_VERDICT) writeEnum(w, VERDICTS, e.verdict!);
        if (f & E_BYNAME) w.varint(t.str(e.byName!));
    });
}

export function readEdges(r: ByteReader, t: ReadTables, budget: Budget): KEdge[] {
    const n = boundedCount(r, budget, 'edges');
    const from: string[] = [];
    let prev = 0;
    for (let i = 0; i < n; i++) { prev += unzigzag(r.varint()); from.push(t.str(prev)); }
    const to = Array.from({ length: n }, () => t.str(r.varint()));
    const kinds = Array.from({ length: n }, () => readWord(r, t, EDGE_KINDS));
    const weights = Array.from({ length: n }, () => r.num());
    const origins = Array.from({ length: n }, () => readEnum(r, ORIGINS));
    const flags = Array.from({ length: n }, () => r.varint());
    for (const f of flags) assertFlagsKnown(f, E_KNOWN);
    return from.map((f0, i): KEdge => {
        const f = flags[i]!;
        const edge: KEdge = { from: f0 as NodeRef, to: to[i]! as NodeRef, kind: kinds[i]!, weight: weights[i]!, origin: origins[i]! };
        if (f & E_CONF) edge.confidence = r.num();
        if (f & E_VERDICT) edge.verdict = readEnum(r, VERDICTS);
        if (f & E_CYCLE) edge.cycle = (f & E_CYCLE_V) !== 0;
        if (f & E_STALE) edge.stale = (f & E_STALE_V) !== 0;
        if (f & E_BYNAME) edge.byName = t.str(r.varint());
        return edge;
    });
}

export type KeyRow = { ref: string; key: string };

export function writeKeys(w: ByteWriter, t: Tables, keys: readonly KeyRow[]): void {
    w.varint(keys.length);
    let prev = 0;
    for (const k of keys) { const i = t.str(k.ref); w.varint(zigzag(i - prev)); prev = i; }
    for (const k of keys) w.varint(t.str(k.key));
}

export function readKeys(r: ByteReader, t: ReadTables, budget: Budget): Array<{ ref: NodeRef; key: string }> {
    const n = boundedCount(r, budget, 'keys');
    const refs: string[] = [];
    let prev = 0;
    for (let i = 0; i < n; i++) { prev += unzigzag(r.varint()); refs.push(t.str(prev)); }
    return refs.map((ref) => ({ ref: ref as NodeRef, key: t.str(r.varint()) }));
}
