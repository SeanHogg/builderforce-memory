/**
 * src/knowledge/findingIdentity.test.ts — identity v2 anchors (PRD 41 §C3): stable when line numbers
 * move and when members are added; never the member set; the documented default.
 */

import { CLONE_GROUP_BADGE, finding, genericFindings } from './findings.js';
import { anchorKindOfRef, anchorWithSnippet, defaultAnchorOf, identityOf, normalizeSnippet } from './findingIdentity.js';
import { godModuleFindings } from './godModule.js';
import { possibleSameFindings } from './resolve.js';
import { normalizeAnalysis } from './comparability.js';
import type { KEdge, KNode, KnowledgeGraph } from './model.js';

const ref = (p: string) => `src:r::${p}`;
const node = (p: string, extra: Partial<KNode> = {}): KNode => ({ ref: ref(p), kind: 'file', label: p, pack: 'code', importance: 0, origin: 'extracted', ...extra });
const edge = (a: string, b: string, extra: Partial<KEdge>): KEdge => ({ from: ref(a), to: ref(b), kind: 'references', weight: 1, origin: 'extracted', ...extra });
const graph = (nodes: KNode[], edges: KEdge[]): KnowledgeGraph => ({
    scope: { rootId: 'r', packs: ['code'], digest: 'd' }, nodes, edges, findings: [],
    stats: { sources: nodes.length, nodes: nodes.length, edges: edges.length, stale: 0, truncated: false },
});
const of = (g: KnowledgeGraph, kind: string) => genericFindings(g, new Map()).filter((f) => f.kind === kind).map((f) => identityOf(f));

describe('identityOf', () => {
    it('composes kind@version:anchorKind:primary and keeps the v1 id', () => {
        const f = finding('god-module', 'code', [ref('a.ts')], 'warn', {}, undefined, { anchor: ref('a.ts'), anchorKind: 'file' });
        expect(identityOf(f)).toEqual({ id: f.id, anchor: `god-module@1:file:${ref('a.ts')}`, anchorKind: 'file', members: [ref('a.ts')] });
        expect(identityOf(f, 3).anchor).toBe(`god-module@3:file:${ref('a.ts')}`);
    });

    it('defaults to the sorted-first node ref (never the set), else the pack', () => {
        const f = finding('orphan', 'docs', [ref('z.md'), ref('a.md#intro')], 'info', {});
        expect(f.anchor).toBeUndefined();
        expect(defaultAnchorOf(f)).toEqual({ anchor: ref('a.md#intro'), anchorKind: 'symbol' });
        expect(identityOf(f).anchor).toBe(`orphan@1:symbol:${ref('a.md#intro')}`);
        expect(defaultAnchorOf({ nodes: [], pack: 'docs' })).toEqual({ anchor: 'pack:docs', anchorKind: 'group' });
    });

    it('reads the anchor kind from a ref', () => {
        expect(anchorKindOfRef(ref('a.ts'))).toBe('file');
        expect(anchorKindOfRef(ref('a.ts#Foo.bar'))).toBe('symbol');
        expect(anchorKindOfRef('ent:code:lease')).toBe('symbol');
        expect(anchorKindOfRef('fact:x')).toBe('group');
    });

    it('a badged clone group keeps its anchor when it gains a member', () => {
        const b = { badges: { [CLONE_GROUP_BADGE]: 'g1' } };
        const two = of(graph([node('a.ts', b), node('b.ts', b)], [edge('a.ts', 'b.ts', { kind: 'duplicate' })]), 'clone-group')[0]!;
        const three = of(graph([node('a.ts', b), node('b.ts', b), node('c.ts', b)],
            [edge('a.ts', 'b.ts', { kind: 'duplicate' }), edge('b.ts', 'c.ts', { kind: 'duplicate' })]), 'clone-group')[0]!;
        expect(three.anchor).toBe(two.anchor);
        expect(two.anchor).toBe('clone-group@1:group:cloneGroup:g1');
        expect(three.id).not.toBe(two.id);
    });

    it('an unbadged clone group / cycle anchors on its sorted-first member, so a member sorting later keeps it', () => {
        const cyc = (extra: string[]) => of(graph(
            ['a.ts', 'b.ts', ...extra].map((p) => node(p)),
            [edge('a.ts', 'b.ts', { cycle: true }), edge('b.ts', 'a.ts', { cycle: true }), ...extra.map((p) => edge('b.ts', p, { cycle: true }))],
        ), 'cycle')[0]!;
        expect(cyc(['z.ts']).anchor).toBe(cyc([]).anchor);
        expect(cyc([]).anchor).toBe(`cycle@1:file:${ref('a.ts')}`);
    });

    it('a partition violation anchors on the layer pair and the from FILE, not the target', () => {
        const nodes = [
            node('ui/a.ts', { partition: 'ui' }), node('ui/a.ts#render', { kind: 'symbol', partition: 'ui' }),
            node('db/x.ts', { partition: 'db' }), node('db/y.ts', { partition: 'db' }),
        ];
        const v1 = of(graph(nodes, [edge('ui/a.ts', 'db/x.ts', { verdict: 'skips' })]), 'partition-violation')[0]!;
        const v2 = of(graph(nodes, [edge('ui/a.ts#render', 'db/y.ts', { verdict: 'skips' })]), 'partition-violation')[0]!;
        expect(v1.anchor).toBe(`partition-violation@1:file:ui->db@${ref('ui/a.ts')}`);
        expect(v2.anchor).toBe(v1.anchor);
    });

    it('god-module and possible-same anchor on their item', () => {
        const [god] = godModuleFindings(graph([node('big.ts', { badges: { patterns: 'god-module' } })], []));
        expect(identityOf(god!).anchor).toBe(`god-module@1:file:${ref('big.ts')}`);
        const [same] = possibleSameFindings([{ a: 'ent:code:zeta', b: 'ent:code:alpha', similarity: 0.9 }]);
        expect(identityOf(same!).anchor).toBe('possible-same@1:symbol:ent:code:alpha');
    });
});

describe('snippet anchors', () => {
    it('re-indenting, CRLF and blank lines keep the normalized-snippet hash', () => {
        expect(normalizeSnippet('  a  =  1;\r\n\r\n\tb = 2;  ')).toBe('a = 1;\nb = 2;');
        expect(anchorWithSnippet('p', 'a = 1;\nb = 2;')).toBe(anchorWithSnippet('p', '    a = 1;\r\n\n    b = 2;'));
        expect(anchorWithSnippet('p', 'a = 1;')).not.toBe(anchorWithSnippet('p', 'a = 2;'));
        expect(anchorWithSnippet('p')).toBe('p');
    });

    it('jscpd / knip anchors hold no line numbers', () => {
        const dups = (start: number) => ({ duplicates: [{ fragment: 'return x;', firstFile: { name: 'src/a.ts', start, end: start + 5 }, secondFile: { name: 'src/b.ts', start: 9, end: 14 } }] });
        const unused = (line: number) => ({ files: ['src/dead.ts'], issues: [{ file: 'src/b.ts', exports: [{ name: 'x', line }], dependencies: [{ name: 'lib' }] }] });
        const a = normalizeAnalysis({}, dups(2), unused(3)).issues, b = normalizeAnalysis({}, dups(80), unused(90)).issues;
        expect(b.map((i) => i.anchor)).toEqual(a.map((i) => i.anchor));
        expect(a.map((i) => i.anchor)).toEqual([
            `duplicate@1:file:${anchorWithSnippet('src/a.ts', 'return x;')}`,
            'unused-file@1:file:src/dead.ts',
            'unused-export@1:symbol:src/b.ts#x',
            'unused-dependency@1:package:src/b.ts#lib',
        ]);
    });
});
