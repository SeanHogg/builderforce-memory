/**
 * src/knowledge/outline.test.ts
 * The truncation outline: the whole graph's files and file-to-file references, present only on a cut graph.
 */

import { buildGraph } from './build.js';
import type { BuildGraphInput, KEdge, KNode, KnowledgeGraph, KnowledgePack, NodeRef } from './model.js';
import {
    buildOutline, outlineOf, outlineRef, OUTLINE_FLAG_CYCLE, OUTLINE_MAX_FILES, OUTLINE_MAX_LINKS, OUTLINE_VERDICTS,
} from './outline.js';
import { edge, extractedOf, fileNode, ref, ROOT } from './testkit.test-util.js';

const PACK: KnowledgePack = { id: 'docs', parsers: [] };
const pad = (i: number) => `p/${String(i).padStart(6, '0')}.ts`;

function chain(extra: Partial<KNode> = {}, kind = 'references'): BuildGraphInput {
    const paths = Array.from({ length: 40 }, (_, i) => `d${i % 4}/f${String(i).padStart(2, '0')}.md`);
    const extracted = paths.map((p, i) => extractedOf(fileNode(p, extra), i + 1 < paths.length ? [edge(p, paths[i + 1]!, kind)] : []));
    return { rootId: ROOT, packs: [PACK], extracted, facts: [], relations: [] };
}

function ref2(i: number): NodeRef { return ref(pad(i)); }

function synthetic(files: number, edges: KEdge[], nodeExtra: (i: number) => Partial<KNode> = () => ({})): KnowledgeGraph {
    const nodes: KNode[] = Array.from({ length: files }, (_, i) => fileNode(pad(i), { importance: i / files, ...nodeExtra(i) }));
    return {
        scope: { rootId: ROOT, packs: [], digest: '' },
        nodes, edges, findings: [],
        stats: { sources: files, nodes: files, edges: edges.length, stale: 0, truncated: true },
    };
}

const refEdge = (a: number, b: number, extra: Partial<KEdge> = {}): KEdge => ({ ...edge(pad(a), pad(b), 'references'), ...extra });

describe('buildGraph outline', () => {
    it('a truncated graph carries every file, including the ones the cut dropped', () => {
        const g = buildGraph(chain(), { maxNodes: 6 });
        expect(g.stats.truncated).toBe(true);
        expect(g.outline!.files).toHaveLength(40);
        expect(g.outline!.capped).toBe(false);
        expect(g.nodes.filter((n) => n.kind === 'file').length).toBeLessThan(40);
        const paths = g.outline!.files.map((f) => f.path);
        expect(paths).toEqual([...paths].sort());
    });

    it('an untruncated graph has no attached outline, but outlineOf builds one', () => {
        const g = buildGraph(chain(), { maxNodes: 500 });
        expect(g.stats.truncated).toBe(false);
        expect(g.outline).toBeUndefined();
        const o = outlineOf(g);
        expect(o.files).toHaveLength(40);
        expect(o.links).toHaveLength(39);
    });

    it('outlineOf returns the attached outline of a truncated graph', () => {
        const g = buildGraph(chain(), { maxNodes: 6 });
        expect(outlineOf(g)).toBe(g.outline);
    });

    it('references between cut files are links, with prefix and path round-tripping to the ref', () => {
        const g = buildGraph(chain(), { maxNodes: 6 });
        const o = g.outline!;
        expect(o.prefix).toBe(`src:${ROOT}::`);
        expect(o.links).toHaveLength(39);
        const a = o.files.findIndex((f) => outlineRef(o, f) === ref('d0/f00.md'));
        const b = o.files.findIndex((f) => outlineRef(o, f) === ref('d1/f01.md'));
        expect(o.files[a]!.path).toBe('d0/f00.md');
        expect(o.links).toContainEqual([a, b, 0, 0]);
        const sorted = [...o.links].sort((x, y) => x[0] - y[0] || x[1] - y[1]);
        expect(o.links).toEqual(sorted);
    });

    it('is deterministic', () => {
        const a = buildGraph(chain(), { maxNodes: 6 });
        const b = buildGraph(chain(), { maxNodes: 6 });
        expect(a.outline).toEqual(b.outline);
        expect(JSON.stringify(a.outline)).toBe(JSON.stringify(b.outline));
    });

    // 'mentions' edges are not references, so these files have no community badges (PRD 38 P1) and the cases below are about the whitelist alone.
    it('keeps only the whitelisted badges', () => {
        const g = buildGraph(chain({ badges: { complexity: 7, entry: 'main', secret: 'x', added: 3, status: 'M' } }, 'mentions'), { maxNodes: 6 });
        for (const f of g.outline!.files) expect(f.badges).toEqual({ complexity: 7, entry: 'main', added: 3, status: 'M' });
    });

    it('keeps the externals string badge and still drops unlisted ones', () => {
        const g = buildGraph(chain({ badges: { externals: 'react,zod', notListed: 'x' } }, 'mentions'), { maxNodes: 6 });
        for (const f of g.outline!.files) expect(f.badges).toEqual({ externals: 'react,zod' });
    });

    it('omits badges when none of the whitelisted keys are present', () => {
        const g = buildGraph(chain({ badges: { other: 3 } }, 'mentions'), { maxNodes: 6 });
        for (const f of g.outline!.files) expect(f.badges).toBeUndefined();
        expect(g.outline!.communities).toBeUndefined();
    });

    it('carries the community badges and the communities table of referenced files', () => {
        const g = buildGraph(chain(), { maxNodes: 6 });
        for (const f of g.outline!.files) expect(Object.keys(f.badges ?? {}).sort()).toEqual(['communityConfidence', 'communityId']);
        const table = g.outline!.communities!;
        expect(table.length).toBeGreaterThan(0);
        expect(table.reduce((s, c) => s + c.size, 0)).toBe(g.outline!.files.length);
        expect(table.map((c) => c.id)).toEqual([...table.map((c) => c.id)].sort((a, b) => a - b));
        for (const f of g.outline!.files) expect(table.some((c) => c.id === f.badges!['communityId'])).toBe(true);
    });
});

describe('buildOutline shape', () => {
    it('indexes areas and partitions, and shortens the hash to 12 characters', () => {
        const areaA = ref('areaA') as NodeRef;
        const areaB = ref('areaB') as NodeRef;
        const g = synthetic(3, [], (i) => ({
            area: i === 0 ? areaB : areaA, partition: i === 2 ? undefined : (i === 0 ? 'z' : 'a'), sourceHash: `0123456789abcdef${i}`,
        }));
        const o = buildOutline(g);
        expect(o.areas).toEqual([areaA, areaB]);
        expect(o.partitions).toEqual(['a', 'z']);
        expect(o.files[0]).toMatchObject({ area: 1, partition: 1, hash: '0123456789ab' });
        expect(o.files[1]).toMatchObject({ area: 0, partition: 0 });
        expect(o.files[2]!.partition).toBeUndefined();
    });

    it('uses an empty prefix and full refs when files share none', () => {
        const g = synthetic(2, []);
        g.nodes[1] = { ...g.nodes[1]!, ref: 'src:other::x.ts' };
        const o = buildOutline(g);
        expect(o.prefix).toBe('');
        expect(o.files.map((f) => outlineRef(o, f))).toEqual(['src:other::x.ts', ref(pad(0))]);
    });

    it('rolls section endpoints up to their file and drops the resulting self-links', () => {
        const g = synthetic(3, [
            { ...refEdge(0, 1), from: `${ref2(0)}#s1` as NodeRef, to: `${ref2(1)}#s2` as NodeRef },
            { ...refEdge(0, 0), from: `${ref2(0)}#s1` as NodeRef, to: `${ref2(0)}#s2` as NodeRef },
        ]);
        expect(buildOutline(g).links).toEqual([[0, 1, 0, 0]]);
    });

    it('merges a file pair to its worst verdict and cycle flag', () => {
        const g = synthetic(2, [
            refEdge(0, 1, { verdict: 'allowed' }),
            { ...refEdge(0, 1, { verdict: 'upward', cycle: true }), from: `${ref2(0)}#s` as NodeRef },
        ]);
        expect(buildOutline(g).links).toEqual([[0, 1, OUTLINE_VERDICTS.indexOf('upward'), OUTLINE_FLAG_CYCLE]]);
    });
});

describe('buildOutline caps', () => {
    it('over the file cap keeps the most important files; links re-index into the kept list', () => {
        const n = OUTLINE_MAX_FILES + 5;
        // Files 0..4 are the least important and are dropped; 5 -> 6 survives and 0 -> 6 does not.
        const o = buildOutline(synthetic(n, [refEdge(5, 6, { verdict: 'same' }), refEdge(0, 6)]));
        expect(o.files).toHaveLength(OUTLINE_MAX_FILES);
        expect(o.capped).toBe(true);
        expect(o.files.some((f) => outlineRef(o, f) === ref2(0))).toBe(false);
        expect(o.files.some((f) => outlineRef(o, f) === ref2(n - 1))).toBe(true);
        const a = o.files.findIndex((f) => outlineRef(o, f) === ref2(5));
        const b = o.files.findIndex((f) => outlineRef(o, f) === ref2(6));
        expect(o.links).toEqual([[a, b, OUTLINE_VERDICTS.indexOf('same'), 0]]);
    });

    it('over the link cap keeps the heaviest links by weight and flags capped', () => {
        const n = 400;
        const edges: KEdge[] = [];
        for (let a = 0; a < n && edges.length < OUTLINE_MAX_LINKS + 10; a++) {
            for (let b = 0; b < n && edges.length < OUTLINE_MAX_LINKS + 10; b++) if (a !== b) edges.push(refEdge(a, b));
        }
        // The first pair is the lowest-ranked by every tiebreak except weight: it must survive the cut.
        const first = edges.findIndex((e) => e.from === ref2(0) && e.to === ref2(1));
        edges[first] = refEdge(0, 1, { weight: 50 });
        const o = buildOutline(synthetic(n, edges));
        expect(o.links).toHaveLength(OUTLINE_MAX_LINKS);
        expect(o.capped).toBe(true);
        expect(o.links).toContainEqual([0, 1, 0, 0]);
    });

    it('under the caps is not capped', () => {
        expect(buildOutline(synthetic(10, [refEdge(0, 1)])).capped).toBe(false);
    });
});

describe('buildOutline — systems', () => {
    const files = ['a/x.ts', 'a/packages/web/y.ts', 'b/z.ts'];
    const withSystems = (): KnowledgeGraph => {
        const g = buildGraph({
            rootId: ROOT, packs: [PACK], facts: [], relations: [],
            extracted: files.map((p) => extractedOf(fileNode(p))),
            systems: [
                { path: 'a', name: 'a', files: 0, apps: [{ path: 'a/packages/web', name: 'web' }] },
                { path: 'b', name: 'b', files: 0, apps: [], copyOf: 'a', copySimilarity: 1, copyReason: 'no-git' },
            ],
        });
        return g;
    };

    it('files carry system, app and copyOf as indexes into the outline pools, and the outline carries the systems', () => {
        const o = buildOutline(withSystems());
        expect(o.systems).toEqual(['a', 'b']);
        expect(o.apps).toEqual(['a/packages/web']);
        const byPath = (p: string) => o.files.find((f) => f.path.endsWith(p))!;
        expect(byPath('a/x.ts')).toMatchObject({ system: 0 });
        expect(byPath('a/x.ts').app).toBeUndefined();
        expect(byPath('y.ts')).toMatchObject({ system: 0, app: 0 });
        expect(byPath('b/z.ts')).toMatchObject({ system: 1, copyOf: 0 });
        expect(o.systemInfo?.map((s) => [s.path, s.files, s.copyOf])).toEqual([['a', 2, undefined], ['b', 1, 'a']]);
    });

    it('a root without systems has none of these fields', () => {
        const o = buildOutline(buildGraph({ rootId: ROOT, packs: [PACK], facts: [], relations: [], extracted: files.map((p) => extractedOf(fileNode(p))) }));
        expect(o.systems).toBeUndefined();
        expect(o.apps).toBeUndefined();
        expect(o.systemInfo).toBeUndefined();
        expect(o.files.every((f) => f.system === undefined)).toBe(true);
    });

    it('a one-system root keeps the empty system as index 0', () => {
        const g = buildGraph({
            rootId: ROOT, packs: [PACK], facts: [], relations: [], extracted: files.map((p) => extractedOf(fileNode(p))),
            systems: [{ path: '', name: 'r', files: 0, apps: [] }],
        });
        const o = buildOutline(g);
        expect(o.systems).toEqual(['']);
        expect(o.files.every((f) => f.system === 0)).toBe(true);
    });

    it('survives a JSON round trip (the wire form)', () => {
        const o = buildOutline(withSystems());
        expect(JSON.parse(JSON.stringify(o))).toEqual(o);
    });
});
