/**
 * src/knowledge/build.test.ts
 * buildGraph: byte-identical output for equal input, the fact join and staleness, the
 * near-duplicate/duplicate rule, live overlay, and the truncation keep-rules.
 */

import { buildGraph, graphDigest } from './build.js';
import { brokenLinkRule } from './packs/docs.js';
import type { BuildGraphInput, KnowledgePack, StoredFact } from './model.js';
import { edge, extractedOf, fileNode, ref, ROOT } from './testkit.test-util.js';

// The broken-link rule makes the truncation fixture's dangling link a finding.
const PACK: KnowledgePack = { id: 'docs', parsers: [], findings: [brokenLinkRule] };

function fixture(): BuildGraphInput {
    return {
        rootId: ROOT,
        packs: [PACK],
        extracted: [
            extractedOf(fileNode('a/one.md'), [edge('a/one.md', 'a/two.md'), edge('a/one.md', 'b/three.md')]),
            extractedOf(fileNode('a/two.md'), [edge('a/two.md', 'b/three.md')]),
            extractedOf(fileNode('b/three.md'), [edge('b/three.md', 'a/one.md', 'duplicate')]),
        ],
        facts: [
            { key: 'note', content: 'A human note', source: 's', importance: 1, origin: 'human', fileAnchors: [{ path: 'a/one.md', contentHash: 'old' }] },
            { key: 'claim:1', content: 'one ends 2027', source: 's', importance: 1, origin: 'model', fileAnchors: [{ path: 'a/one.md', contentHash: 'old' }] },
            { key: 'claim:2', content: 'two is fine', source: 's', importance: 1, origin: 'model', fileAnchors: [{ path: 'a/two.md', contentHash: 'h:a/two.md' }] },
            { key: 'floating', content: 'No anchor', source: 's', importance: 1, origin: 'agent', fileAnchors: [] },
        ],
        relations: [{ from: 'fact:claim:1', to: 'fact:claim:2', kind: 'contradicts', weight: 1, origin: 'model', confidence: 0.8 }],
        nearDuplicates: [
            { a: ref('a/one.md'), b: ref('a/two.md'), similarity: 0.91 },
            { a: ref('a/one.md'), b: ref('b/three.md'), similarity: 0.95 },
        ],
    };
}

function reversed(input: BuildGraphInput): BuildGraphInput {
    return {
        ...input,
        extracted: [...input.extracted].reverse().map((e) => ({ ...e, edges: [...e.edges].reverse() })),
        facts: [...input.facts].reverse(),
        relations: [...input.relations].reverse(),
        nearDuplicates: [...(input.nearDuplicates ?? [])].reverse(),
    };
}

describe('buildGraph — determinism', () => {
    it('equal input (in any order) gives byte-identical JSON and digest', () => {
        const a = JSON.stringify(buildGraph(fixture()));
        const b = JSON.stringify(buildGraph(reversed(fixture())));
        expect(a).toBe(b);
        expect(graphDigest(fixture())).toBe(graphDigest(reversed(fixture())));
        expect(graphDigest(fixture())).toMatch(/^[0-9a-f]{64}$/);
    });

    it('a different input gives a different digest', () => {
        const changed = fixture();
        changed.facts = changed.facts.slice(1);
        expect(graphDigest(changed)).not.toBe(graphDigest(fixture()));
    });
});

describe('buildGraph — the join', () => {
    const g = buildGraph(fixture());
    const node = (r: string) => g.nodes.find((n) => n.ref === r);

    it('joins facts to sources by anchor path', () => {
        expect(g.edges).toEqual(expect.arrayContaining([
            expect.objectContaining({ from: 'fact:note', to: ref('a/one.md'), kind: 'about', origin: 'human' }),
            expect.objectContaining({ from: 'fact:claim:2', to: ref('a/two.md'), kind: 'about', origin: 'model' }),
        ]));
    });

    it('a fact with no anchor is a node with no about edge', () => {
        expect(node('fact:floating')).toBeDefined();
        expect(g.edges.some((e) => e.from === 'fact:floating')).toBe(false);
    });

    it('flags a model fact judged from another hash stale — never a human one', () => {
        expect(node('fact:claim:1')?.stale).toBe(true);
        expect(node('fact:claim:2')?.stale).toBeUndefined();
        expect(node('fact:note')?.stale).toBeUndefined();
        expect(g.stats.stale).toBe(1);
        expect(g.findings.some((f) => f.kind === 'stale' && f.nodes.includes('fact:claim:1'))).toBe(true);
    });

    it('stored relations become edges', () => {
        expect(g.edges).toContainEqual(expect.objectContaining({ from: 'fact:claim:1', to: 'fact:claim:2', kind: 'contradicts', confidence: 0.8 }));
    });

    it('near-duplicates become edges unless the pair is an exact duplicate', () => {
        expect(g.edges).toContainEqual(expect.objectContaining({ from: ref('a/one.md'), to: ref('a/two.md'), kind: 'near-duplicate' }));
        expect(g.edges.some((e) => e.kind === 'near-duplicate' && e.to === ref('b/three.md'))).toBe(false);
        expect(g.findings.some((f) => f.kind === 'clone-group')).toBe(true);
    });

    it('extracted items keep origin extracted, importance is normalized to 0–1', () => {
        expect(node(ref('a/one.md'))?.origin).toBe('extracted');
        expect(Math.max(...g.nodes.map((n) => n.importance))).toBe(1);
        expect(g.nodes.every((n) => n.importance >= 0 && n.importance <= 1)).toBe(true);
    });

    it('live changes badge nodes and run files become co-changed edges', () => {
        const live = buildGraph({
            ...fixture(),
            live: { changes: [{ path: 'a/two.md', status: 'M', added: 3, removed: 1 }], runFiles: ['a/two.md', 'a/one.md'] },
        });
        expect(live.nodes.find((n) => n.ref === ref('a/two.md'))?.badges).toEqual({ added: 3, removed: 1, status: 'M' });
        expect(live.edges).toContainEqual(expect.objectContaining({ from: ref('a/one.md'), to: ref('a/two.md'), kind: 'co-changed', origin: 'agent' }));
    });
});

describe('buildGraph — truncation keep-rules', () => {
    // A chain of 40 files in 4 folders; one broken link (a finding) far from the focus.
    function chain(): BuildGraphInput {
        const paths = Array.from({ length: 40 }, (_, i) => `d${i % 4}/f${String(i).padStart(2, '0')}.md`);
        const extracted = paths.map((p, i) => extractedOf(fileNode(p), i + 1 < paths.length ? [edge(p, paths[i + 1]!)] : []));
        extracted[30]!.edges.push(edge(paths[30]!, 'gone/missing.md'));
        return { rootId: ROOT, packs: [PACK], extracted, facts: [] as StoredFact[], relations: [] };
    }

    it('keeps focus, finding and changed nodes plus their 1-hop neighbours, and folds the rest into areas', () => {
        const input = { ...chain(), live: { changes: [{ path: 'd2/f10.md', status: 'M', added: 1, removed: 0 }] } };
        const g = buildGraph(input, { maxNodes: 8, focus: [ref('d0/f20.md')] });
        const refs = new Set(g.nodes.map((n) => n.ref));
        expect(g.stats.truncated).toBe(true);
        for (const r of ['d0/f20.md', 'd3/f19.md', 'd1/f21.md', 'd2/f10.md', 'd1/f09.md', 'd3/f11.md', 'd2/f30.md', 'gone/missing.md']) {
            expect(refs.has(ref(r))).toBe(true);
        }
        const areas = g.nodes.filter((n) => n.kind === 'area');
        expect(areas.length).toBeGreaterThan(0);
        expect(areas.every((a) => typeof a.badges?.['members'] === 'number')).toBe(true);
        expect(g.edges.every((e) => refs.has(e.from) && refs.has(e.to))).toBe(true);
        expect(g.edges.every((e) => e.from !== e.to)).toBe(true);
        expect(g.stats.nodes).toBe(g.nodes.length);
    });

    it('does not truncate under the budget', () => {
        const g = buildGraph(chain());
        expect(g.stats.truncated).toBe(false);
        expect(g.nodes.some((n) => n.kind === 'area')).toBe(false);
    });

    it('truncation is deterministic too', () => {
        const a = JSON.stringify(buildGraph(chain(), { maxNodes: 6, focus: [ref('d0/f00.md')] }));
        const b = JSON.stringify(buildGraph(reversed(chain()), { maxNodes: 6, focus: [ref('d0/f00.md')] }));
        expect(a).toBe(b);
    });
});
