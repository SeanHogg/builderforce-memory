/**
 * src/contract/regression.symbolEdges.test.ts — the symbol-level edge kinds (PRD 43 P0b): `extends`, `implements`,
 * `composes`, `uses`, `calls` (with call sites). One `regression:` test per kind, the pack round trip, and the
 * compatibility proof in both directions:
 *
 *   - BACKWARD: a pack written before this change (edge kinds as the 11 legacy codes, no SITE section) decodes to the
 *     same pack, and re-encodes to the same bytes;
 *   - FORWARD: a pack written now decodes under a reader of before this change — the new kinds are written as strings
 *     (never as codes 12+ that reader refuses) and the SITE section is a new tag that reader skips.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { readFrame, writeFrame, writeSection } from '../knowledge/pack/frame.js';
import { TAG } from '../knowledge/pack/sections/index.js';
import { EDGE_KINDS, WRITTEN_EDGE_KINDS } from '../knowledge/pack/sections/columns.js';
import { writeKeys, writeNodes } from '../knowledge/pack/sections/extrLists.js';
import { collector, readStringTable, tablesOf, writeHashTable, writeStringTable, type Tables } from '../knowledge/pack/strings.js';
import { ByteReader, ByteWriter } from '../knowledge/pack/varint.js';
import { applyEntryDiff, decodePack, diffEntry, encodePack, ZERO_DIGEST, type Pack } from '../knowledge/pack/index.js';
import { buildGraph, truncateGraph, dedupeSymbolEdges, SYMBOL_EDGE_KINDS, type CallSite, type Extracted, type KEdge, type NodeRef } from '../knowledge/index.js';

const H1 = 'a'.repeat(64), H2 = 'b'.repeat(64);
const HEAD = { rootId: 'r', baseGeneration: 0, targetDigest: ZERO_DIGEST };
const FILE = 'src:r::src/a.ts' as NodeRef;
const sym = (name: string, line: number, path = 'src/a.ts') => `src:r::${path}#${name}@${line}` as NodeRef;
const site = (line: number, column: number, order: number, path = 'src/a.ts'): CallSite => ({ path, line, column, order });

/** The 11 kinds a reader of before 2026-11-10 knows as codes (index + 1); a higher code makes it throw. */
const LEGACY_EDGE_KINDS = [
    'references', 'links', 'contains', 'mentions', 'duplicate', 'near-duplicate', 'contradicts', 'agrees', 'supersedes', 'about', 'co-changed',
];

function extraction(edges: KEdge[]): Extracted {
    return {
        sourceRef: FILE, sourceHash: H1,
        nodes: [{ ref: FILE, kind: 'file', label: 'a.ts', pack: 'code', importance: 0.5, origin: 'extracted', sourceHash: H1 }],
        edges, keys: [{ ref: FILE, key: 'a.ts' }],
    };
}
const packOf = (edges: KEdge[]): Pack => ({ ...HEAD, extracted: [{ kind: 'full', hash: H1, entries: [extraction(edges)] }] });
const edge = (from: NodeRef, to: NodeRef, kind: string, sites?: CallSite[]): KEdge =>
    ({ from, to, kind, weight: 1, origin: 'extracted', ...(sites ? { sites } : {}) });

async function sectionsOf(bytes: Uint8Array): Promise<Map<number, Uint8Array>> {
    const { stored } = readFrame(bytes);
    const r = new ByteReader(stored);
    const out = new Map<number, Uint8Array>();
    while (!r.done) { const tag = r.u8(); out.set(tag, r.bytes(r.varint())); }
    return out;
}
const stringsOf = (sections: Map<number, Uint8Array>): string[] => readStringTable(sections.get(TAG.STR)!, 1e5, 1e7);

async function roundTrip(pack: Pack): Promise<Pack> {
    return decodePack(await encodePack(pack));
}

describe('regression: symbol-level edge kinds', () => {
    it('regression: an `extends` edge (class → its base) survives the pack and is written as a string an older reader decodes', async () => {
        const e = edge(sym('Repo', 3), sym('BaseRepo', 1, 'src/base.ts'), 'extends');
        const back = await roundTrip(packOf([e]));
        expect((back.extracted![0]!.entries[0] as Extracted).edges).toEqual([e]);
        expect(stringsOf(await sectionsOf(await encodePack(packOf([e]), { deflate: false })))).toContain('extends');
    });

    it('regression: an `implements` edge (class → interface) survives the pack', async () => {
        const e = edge(sym('Repo', 3), sym('Store', 1, 'src/store.ts'), 'implements');
        expect(((await roundTrip(packOf([e]))).extracted![0]!.entries[0] as Extracted).edges).toEqual([e]);
    });

    it('regression: a `composes` edge (class → the class of a field) survives the pack', async () => {
        const e = edge(sym('Service', 5), sym('Repo', 3), 'composes');
        expect(((await roundTrip(packOf([e]))).extracted![0]!.entries[0] as Extracted).edges).toEqual([e]);
    });

    it('regression: a `uses` edge (symbol → a type it names) survives the pack', async () => {
        const e = edge(sym('Service.load', 9), sym('Row', 1, 'src/types.ts'), 'uses');
        expect(((await roundTrip(packOf([e]))).extracted![0]!.entries[0] as Extracted).edges).toEqual([e]);
    });

    it('regression: a `calls` edge keeps EVERY call site — repeats included — with path, line, column and order', async () => {
        const sites = [site(10, 5, 0), site(12, 9, 2), site(14, 9, 3)];
        const other = [site(11, 5, 1)];
        const edges = [
            edge(sym('Service.load', 9), sym('Repo.get', 4), 'calls', sites),
            edge(sym('Service.load', 9), sym('missing', 0, 'node_modules/zod'), 'calls', other),
            edge(sym('Service.load', 9), sym('Row', 1, 'src/types.ts'), 'uses'),
        ];
        const back = (await roundTrip(packOf(edges))).extracted![0]!.entries[0] as Extracted;
        expect(back.edges).toEqual(edges);
        // The caller's calls in source order: the order index interleaves the two callees.
        const ordered = back.edges.flatMap((e) => (e.sites ?? []).map((s) => [s.order, e.to] as const)).sort((a, b) => a[0] - b[0]);
        expect(ordered.map(([o]) => o)).toEqual([0, 1, 2, 3]);
    });
});

describe('symbol edges: pack round trip', () => {
    const edges = SYMBOL_EDGE_KINDS.map((kind, i) => edge(sym(`S${i}`, i + 1), sym(`T${i}`, 1, 'src/t.ts'), kind, kind === 'calls' ? [site(3, 1, 0), site(4, 7, 1)] : undefined));

    it('encode → decode keeps every new kind and the call sites, deflated or plain, and re-encodes to the same bytes', async () => {
        const pack = packOf([edge(FILE, 'src:r::src/t.ts' as NodeRef, 'references'), ...edges]);
        for (const deflate of [true, false]) {
            const bytes = await encodePack(pack, { deflate });
            const back = await decodePack(bytes);
            expect((back.extracted![0]!.entries[0] as Extracted).edges).toEqual(pack.extracted![0]!.entries[0]!.edges);
            expect(Buffer.from(await encodePack(back, { deflate })).equals(Buffer.from(bytes))).toBe(true);
        }
    });

    it('a diff entry carries the sites of the edges it adds', async () => {
        const prior = extraction(edges.slice(0, 2));
        const next: Extracted = { ...extraction([...edges.slice(0, 2), edge(sym('S9', 9), sym('T9', 2), 'calls', [site(9, 3, 0)])]), sourceHash: H2 };
        const diff = diffEntry(prior, next)!;
        const pack: Pack = { ...HEAD, extracted: [{ kind: 'full', hash: H1, entries: [prior] }, { kind: 'diff', hash: H2, prior: H1, entries: [diff] }] };
        const back = await roundTrip(pack);
        const backDiff = back.extracted!.find((x) => x.kind === 'diff')!.entries[0]!;
        expect(applyEntryDiff(prior, backDiff as never)).toEqual(applyEntryDiff(prior, diff));
        expect(applyEntryDiff(prior, backDiff as never).edges.at(-1)!.sites).toEqual([site(9, 3, 0)]);
    });

    it('regression: a malformed SITE section never rejects the pack — it decodes whole, without any sites', async () => {
        const good = edge(sym('A', 1), sym('B', 2), 'calls', [site(1, 1, 0)]);
        const other = edge(sym('A', 1), sym('C', 3), 'calls', [site(2, 1, 1)]);
        const bytes = await encodePack(packOf([good, other]), { deflate: false });
        const withSite = async (site: Uint8Array): Promise<Pack> => {
            const sections = await sectionsOf(bytes);
            sections.set(TAG.SITE, site);
            const body = new ByteWriter();
            for (const [tag, b] of sections) writeSection(body, tag, b);
            const plain = body.finish();
            return decodePack(writeFrame({ flags: 0, ...HEAD }, plain, plain));
        };
        const bad = (fill: (w: ByteWriter) => void): Uint8Array => { const w = new ByteWriter(); fill(w); return w.finish(); };
        const variants = [
            bad((w) => { w.varint(1); w.varint(0); w.varint(0); w.varint(7); w.varint(1); w.varint(0); w.varint(1); w.varint(1); w.varint(0); }), // no edge 7
            // a VALID first group, then a repeat: nothing may be attached, not even the first
            bad((w) => { w.varint(2); for (let g = 0; g < 2; g++) { w.varint(0); w.varint(0); w.varint(0); w.varint(1); w.varint(0); w.varint(1); w.varint(1); w.varint(0); } }),
            bad((w) => { w.varint(1); w.varint(0); w.varint(0); w.varint(0); w.varint(0); }), // empty list
            bad((w) => { w.varint(1); w.varint(0); w.varint(0); w.varint(0); w.varint(1); w.varint(99_999); w.varint(1); w.varint(1); w.varint(0); }), // bad STR index
            bad((w) => { w.varint(5); }), // truncated
            bad((w) => { w.varint(1); w.varint(0); w.varint(0); w.varint(0); w.varint(1); w.varint(0); w.varint(1); w.varint(1); w.varint(0); w.u8(7); }), // trailing byte
        ];
        for (const v of variants) {
            const back = await withSite(v);
            expect((back.extracted![0]!.entries[0] as Extracted).edges).toEqual([good, other].map(({ sites: _, ...e }) => e));
        }
    });

    it('regression: an invalid site is dropped on write instead of refusing the pack', async () => {
        const e = edge(sym('A', 1), sym('B', 2), 'calls', [site(1, 1, 0), { path: 'src/a.ts', line: -1, column: 1, order: 1 }, { path: 'src/a.ts', line: 2, column: 1.5, order: 2 }]);
        const back = await roundTrip(packOf([e]));
        expect((back.extracted![0]!.entries[0] as Extracted).edges[0]!.sites).toEqual([site(1, 1, 0)]);
        const allBad = edge(sym('A', 1), sym('B', 2), 'calls', [{ path: 'src/a.ts', line: Number.NaN, column: 1, order: 0 }]);
        expect((((await roundTrip(packOf([allBad]))).extracted![0]!.entries[0]) as Extracted).edges[0]!.sites).toBeUndefined();
    });

    it('regression: a diff where only an edge\'s sites change carries the new sites (SITE in a diff entry)', async () => {
        const before = edge(sym('A.run', 3), sym('B.get', 2), 'calls', [site(4, 5, 0)]);
        const afterSites = [site(4, 5, 0), site(6, 9, 2)];
        const prior = extraction([edge(sym('A', 1), sym('B', 1), 'extends'), before]);
        const next: Extracted = { ...extraction([edge(sym('A', 1), sym('B', 1), 'extends'), { ...before, sites: afterSites }]), sourceHash: H2 };
        const diff = diffEntry(prior, next)!;
        expect(diff.edges.remove).toHaveLength(1);
        expect(diff.edges.add.map((a) => a.item.sites)).toEqual([afterSites]);
        const back = await roundTrip({ ...HEAD, extracted: [{ kind: 'full', hash: H1, entries: [prior] }, { kind: 'diff', hash: H2, prior: H1, entries: [diff] }] });
        const backDiff = back.extracted!.find((x) => x.kind === 'diff')!.entries[0]!;
        expect(applyEntryDiff(prior, backDiff as never).edges).toEqual(next.edges);
    });

    it('regression: `sites: []` and absent `sites` are the same edge to diffEntry (no spurious remove/add)', () => {
        const a = edge(sym('A', 1), sym('B', 2), 'calls');
        const prior = extraction([{ ...a, sites: [] }]);
        const next: Extracted = { ...extraction([a]), sourceHash: H2 };
        const diff = diffEntry(prior, next)!;
        expect(diff.edges).toEqual({ remove: [], add: [] });
    });
});

describe('symbol edges: compatibility', () => {
    it('the writer codes only the 11 legacy kinds; the reader knows the symbol kinds after them, append-only', () => {
        expect(WRITTEN_EDGE_KINDS).toEqual(LEGACY_EDGE_KINDS);
        expect(EDGE_KINDS.slice(0, 11)).toEqual(LEGACY_EDGE_KINDS);
        expect(EDGE_KINDS.slice(11)).toEqual([...SYMBOL_EDGE_KINDS]);
    });

    it('BACKWARD: a pack written before this change (legacy codes, no SITE section) decodes identically and re-encodes to its bytes', async () => {
        const legacy = [edge(FILE, 'src:r::src/b.ts' as NodeRef, 'references'), edge(FILE, 'src:r::src/c.ts' as NodeRef, 'co-changed'), edge(FILE, 'src:r::x' as NodeRef, 'custom-kind')];
        const x = extraction(legacy);
        // EXTR as a writer of before 2026-11-10 wrote it, edge columns by hand from the legacy vocabulary.
        const write = (t: Tables): Uint8Array => {
            const w = new ByteWriter();
            w.varint(1); w.u8(0); w.varint(t.hash(H1)); w.varint(1);
            w.varint(t.str(x.sourceRef)); w.varint(t.hash(x.sourceHash)); w.u8(0);
            writeNodes(w, t, x.nodes);
            w.varint(legacy.length);
            let prev = 0;
            for (const e of legacy) { const i = t.str(e.from); const d = i - prev; w.varint(d >= 0 ? d * 2 : -d * 2 - 1); prev = i; }
            for (const e of legacy) w.varint(t.str(e.to));
            for (const e of legacy) { const at = LEGACY_EDGE_KINDS.indexOf(e.kind); if (at >= 0) w.varint(at + 1); else { w.varint(0); w.varint(t.str(e.kind)); } }
            for (const e of legacy) w.num(e.weight);
            for (const _ of legacy) w.u8(0); // origin 'extracted'
            for (const _ of legacy) w.varint(0); // no optional fields
            writeKeys(w, t, x.keys);
            return w.finish();
        };
        const col = collector(); write(col);
        const t = tablesOf(col.strings, col.hashes);
        const body = new ByteWriter();
        writeSection(body, TAG.STR, writeStringTable(t.strings));
        writeSection(body, TAG.HASH, writeHashTable(t.hashes));
        writeSection(body, TAG.EXTR, write(t));
        const plain = body.finish();
        const old = writeFrame({ flags: 0, ...HEAD }, plain, plain);

        const back = await decodePack(old);
        expect(back).toEqual(packOf(legacy));
        expect(Buffer.from(await encodePack(back, { deflate: false })).equals(Buffer.from(old))).toBe(true);
    });

    it('FORWARD: a pack written now holds the symbol kinds as strings and its sites in a separate tag an older reader skips', async () => {
        const edges = SYMBOL_EDGE_KINDS.map((kind, i) => edge(sym(`S${i}`, i + 1), sym(`T${i}`, 1), kind, kind === 'calls' ? [site(2, 4, 0)] : undefined));
        const sections = await sectionsOf(await encodePack(packOf(edges), { deflate: false }));
        expect(stringsOf(sections)).toEqual(expect.arrayContaining([...SYMBOL_EDGE_KINDS]));
        expect(sections.has(TAG.SITE)).toBe(true);
        // An older reader skips the SITE tag it does not know: the same pack, every edge and kind, without the sites.
        sections.delete(TAG.SITE);
        sections.set(99, new Uint8Array([1, 2, 3])); // and any other unknown tag, as it always has
        const body = new ByteWriter();
        for (const [tag, b] of sections) writeSection(body, tag, b);
        const plain = body.finish();
        const back = await decodePack(writeFrame({ flags: 0, ...HEAD }, plain, plain));
        expect((back.extracted![0]!.entries[0] as Extracted).edges).toEqual(edges.map(({ sites: _, ...e }) => e));
    });
});

/**
 * Goldens produced by the REAL previous release (@seanhogg/builderforce-memory@2026.10.68, installed under the alias
 * `memory-prev`): a pack its writer wrote and what its reader decoded; this release's pack and what 2026.10.68's reader
 * decoded from those exact bytes. Regenerate only with that script, never by hand.
 */
describe('symbol edges: compatibility against the published 2026.10.68 (goldens)', () => {
    interface Golden {
        prevVersion: string; prevPackHex: string; prevPackDecodedByPrev: Pack;
        newPack: Pack; newPackHex: string; newPackDecodedByPrev: Pack;
    }
    const golden = JSON.parse(readFileSync(join(process.cwd(), 'src', 'contract', 'symbolEdges.golden.json'), 'utf8')) as Golden;
    const bytes = (hex: string): Uint8Array => new Uint8Array(Buffer.from(hex, 'hex'));

    it('BACKWARD (real): bytes written by 2026.10.68 decode here exactly as 2026.10.68 decoded them, and re-encode to the same bytes', async () => {
        expect(golden.prevVersion).toBe('2026.10.68');
        const back = await decodePack(bytes(golden.prevPackHex));
        expect(back).toEqual(golden.prevPackDecodedByPrev);
        expect(Buffer.from(await encodePack(back, { deflate: false })).toString('hex')).toBe(golden.prevPackHex);
    });

    it('FORWARD (real): this release still writes the exact bytes 2026.10.68 was proven to read — every edge and kind, the sites skipped', async () => {
        expect(Buffer.from(await encodePack(golden.newPack, { deflate: false })).toString('hex')).toBe(golden.newPackHex);
        // What the previous reader decoded is this release's decode with the sites (an unknown tag to it) left out.
        const stripSites = (p: Pack): unknown => JSON.parse(JSON.stringify(p, (k, v) => (k === 'sites' ? undefined : v)));
        expect(golden.newPackDecodedByPrev).toEqual(stripSites(await decodePack(bytes(golden.newPackHex))));
        // And this release reads its own golden with the sites, including the diff entry whose only change is an edge's sites.
        const diff = (await decodePack(bytes(golden.newPackHex))).extracted!.find((x) => x.kind === 'diff')!.entries[0] as { edges: { add: Array<{ item: KEdge }> } };
        expect(diff.edges.add[0]!.item.sites).toHaveLength(2);
    });
});

describe('symbol edges in buildGraph', () => {
    const file = (path: string, partition?: string): Extracted => ({
        sourceRef: `src:r::${path}` as NodeRef, sourceHash: H1, keys: [],
        nodes: [{ ref: `src:r::${path}` as NodeRef, kind: 'file', label: path, pack: 'p', importance: 0, origin: 'extracted', ...(partition ? { partition } : {}) }],
        edges: [],
    });
    const pack = { id: 'p', parsers: [], partitions: { partitions: [{ id: 'ui', order: 0, match: ['ui/**'] }, { id: 'db', order: 1, match: ['db/**'] }], allow: { ui: ['db'], db: [] } } };

    it('regression: a built graph keeps a calls edge\'s sites, and the symbol kinds neither re-weigh importance nor earn a layer verdict', () => {
        const a = file('ui/a.ts'), b = file('db/b.ts');
        const sites = [site(3, 4, 0), site(5, 4, 1)];
        // db → ui is upward for an import; the same pair through `calls` is no finding and no verdict.
        const base = [{ ...a }, { ...b, edges: [edge('src:r::db/b.ts' as NodeRef, 'src:r::ui/a.ts' as NodeRef, 'references')] }];
        const withSymbols = [{ ...a }, { ...b, edges: [...base[1]!.edges, edge('src:r::db/b.ts' as NodeRef, 'src:r::ui/a.ts' as NodeRef, 'calls', sites)] }];
        const before = buildGraph({ rootId: 'r', packs: [pack], extracted: base, facts: [], relations: [] });
        const after = buildGraph({ rootId: 'r', packs: [pack], extracted: withSymbols, facts: [], relations: [] });
        const calls = after.edges.find((e) => e.kind === 'calls')!;
        expect(calls.sites).toEqual(sites);
        expect(calls.verdict).toBeUndefined();
        expect(after.nodes.map((n) => [n.ref, n.importance])).toEqual(before.nodes.map((n) => [n.ref, n.importance]));
        expect(after.findings.filter((f) => f.kind === 'partition-violation')).toHaveLength(before.findings.filter((f) => f.kind === 'partition-violation').length);
    });
});

describe('symbol edges in truncateGraph', () => {
    it('regression: a symbol-level edge never earns a seed\'s neighbour a slot (only the import does)', () => {
        const n = (path: string, importance: number) => ({ ref: `src:r::${path}` as NodeRef, kind: 'file', label: path, pack: 'p', importance, origin: 'extracted' as const });
        const nodes = [n('seed.ts', 0.1), n('big.ts', 0.9), n('called.ts', 0.01), n('imported.ts', 0.02)];
        const g = {
            scope: { rootId: 'r', packs: ['p'], digest: 'd' }, nodes, findings: [], stats: { sources: 4, nodes: 4, edges: 2, stale: 0, truncated: false },
            edges: [edge('src:r::seed.ts' as NodeRef, 'src:r::called.ts' as NodeRef, 'calls', [site(1, 1, 0)]), edge('src:r::seed.ts' as NodeRef, 'src:r::imported.ts' as NodeRef, 'references')],
        } as unknown as Parameters<typeof truncateGraph>[0];
        // A node that is not kept folds into an `area` node; the kept ones stay files.
        const kept = truncateGraph(g, { maxNodes: 1, focus: ['src:r::seed.ts' as NodeRef], changed: [] }).nodes.filter((x) => x.kind === 'file').map((x) => x.ref);
        expect(kept).toContain('src:r::imported.ts');
        expect(kept).not.toContain('src:r::called.ts');
    });
});

describe('dedupeSymbolEdges', () => {
    it('keeps one edge per (from, to, kind), merges repeated calls into sites in source order, and caps edges per symbol and sites per edge', () => {
        const a = sym('A.run', 1), b = sym('B.get', 2), c = sym('C', 3);
        const out = dedupeSymbolEdges([
            edge(a, b, 'calls', [site(5, 3, 2)]),
            edge(a, b, 'calls', [site(4, 3, 0)]),
            edge(a, c, 'uses'),
            edge(a, c, 'uses'),
            edge(FILE, b, 'references'),
            edge(FILE, b, 'references'),
        ]);
        expect(out).toEqual([edge(a, b, 'calls', [site(4, 3, 0), site(5, 3, 2)]), edge(a, c, 'uses'), edge(FILE, b, 'references'), edge(FILE, b, 'references')]);
        const many = Array.from({ length: 10 }, (_, i) => edge(a, sym(`T${i}`, i), 'calls', Array.from({ length: 5 }, (_, s) => site(i, s, i * 5 + s))));
        const capped = dedupeSymbolEdges(many, { maxPerSymbol: 3, maxSites: 2 });
        expect(capped).toHaveLength(3);
        expect(capped.every((e) => e.sites!.length === 2)).toBe(true);
    });
});
