/**
 * src/knowledge/communities.test.ts
 * PRD 38 P1: Louvain communities over the file reference graph (code `references` and docs `links`), the
 * label-propagation fallback above the cap, the membership confidence, determinism, the placement helper
 * and that an unreferenced file gets no badge. Hand computations are in the comments.
 */

import {
    applyCommunities, communityTable, COMMUNITY_EDGE_KINDS, detectCommunities, placementOf,
} from './communities.js';
import { buildGraph } from './build.js';
import { buildAdjacency, localMoving } from './louvain.js';
import { buildOutline } from './outline.js';
import type { BuildGraphInput, KNode, KnowledgeGraph } from './model.js';
import { traceGraph } from './trace.js';
import { edge, extractedOf, fileNode, ref } from './testkit.test-util.js';

const E = (a: string, b: string, weight = 1) => ({ from: ref(a), to: ref(b), weight });
const same = (a: unknown, b: unknown) => expect(JSON.stringify(a)).toBe(JSON.stringify(b));

/** Two triangles, a1-a2-a3 and b1-b2-b3, joined by the single edge a3-b1. m = 7. */
const CLIQUES = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3'].map((p) => `${p}.md`);
const CLIQUE_EDGES = [
    E('a1.md', 'a2.md'), E('a2.md', 'a3.md'), E('a1.md', 'a3.md'),
    E('b1.md', 'b2.md'), E('b2.md', 'b3.md'), E('b1.md', 'b3.md'),
    E('a3.md', 'b1.md'),
];

describe('detectCommunities — Louvain', () => {
    const r = detectCommunities(CLIQUES.map(ref), CLIQUE_EDGES);
    const id = (p: string) => r.members.get(ref(p))!.communityId;

    it('splits two cliques joined by one edge into two communities with Q > 0.3', () => {
        expect(r.method).toBe('louvain');
        expect(new Set([...r.members.values()].map((m) => m.communityId)).size).toBe(2);
        expect(id('a1.md')).toBe(id('a3.md'));
        expect(id('b1.md')).toBe(id('b3.md'));
        expect(id('a1.md')).not.toBe(id('b1.md'));
        expect(r.modularity).toBeGreaterThan(0.3);
        // Hand computed: 2 × (3/7 − (7/14)²) = 0.357143.
        expect(r.modularity).toBe(0.357143);
    });

    it('community ids are indexes by smallest member path', () => {
        expect(id('a2.md')).toBe(0);
        expect(id('b3.md')).toBe(1);
    });

    it('confidence is the share of a file\'s edge weight inside its community', () => {
        expect(r.members.get(ref('a1.md'))!.confidence).toBe(1);
        // a3: two edges inside (a1, a2), one out (b1) → 2/3.
        expect(r.members.get(ref('a3.md'))!.confidence).toBe(0.666667);
        expect(r.members.get(ref('b1.md'))!.confidence).toBe(0.666667);
    });

    it('sums both directions and weights by reference count', () => {
        const weighted = detectCommunities(CLIQUES.map(ref), [...CLIQUE_EDGES, E('b1.md', 'a3.md', 5)]);
        // Hand computed: the a3-b1 bridge weighs 1 + 5 = 6 (m = 12), so a3 and b1 form one community
        // ({a1,a2}, {a3,b1}, {b2,b3}; Q = 0.166667 beats the two triangles' Q = 0). Each has 6 of its 8 edge weight inside.
        const w = (p: string) => weighted.members.get(ref(p))!;
        expect(w('a3.md').communityId).toBe(w('b1.md').communityId);
        expect([w('a1.md').communityId, w('a3.md').communityId, w('b2.md').communityId]).toEqual([0, 1, 2]);
        expect(w('a3.md').confidence).toBe(0.75);
        expect(w('b1.md').confidence).toBe(0.75);
        expect(weighted.modularity).toBe(0.166667);
    });

    it('a file with no references gets no membership', () => {
        const withIsolated = detectCommunities([...CLIQUES, 'lonely.md'].map(ref), CLIQUE_EDGES);
        expect(withIsolated.members.has(ref('lonely.md'))).toBe(false);
        expect(withIsolated.members.size).toBe(6);
        expect(withIsolated.modularity).toBe(r.modularity);
    });

    it('no edges: nothing is assigned and Q is 0', () => {
        const none = detectCommunities(CLIQUES.map(ref), []);
        expect(none.members.size).toBe(0);
        expect(none.modularity).toBe(0);
    });

    it('ignores self references and non-positive weights', () => {
        const noisy = detectCommunities(CLIQUES.map(ref), [...CLIQUE_EDGES, E('a1.md', 'a1.md', 9), E('a1.md', 'b3.md', 0)]);
        same([...noisy.members], [...r.members]);
    });

    it('shuffled input gives byte-identical output', () => {
        const shuffled = detectCommunities([...CLIQUES].reverse().map(ref), [...CLIQUE_EDGES].reverse().map((e) => ({ ...e })));
        same([...shuffled.members], [...r.members]);
        expect(shuffled.modularity).toBe(r.modularity);
    });

    it('a ring of ten triangles: the second level merges neighbouring cliques', () => {
        // Clique i is ci{a,b,c}; ci-c links to c(i+1)-a. m = 10 × 3 + 10 = 40, 2m = 80, every clique has degree sum 8.
        // Level 1 forms the ten cliques. Level 2 (nodes of degree 8, self-loop 3, ring edges of weight 1): node 0 joins 1 on a
        // 0.2 gain (1 − 8·8/80), 2 joins 3, … so five pairs. Level 3 (degree 16): 1 − 16·16/80 < 0, no further merge.
        // Q = 5 × (14/80 − (16/80)²) = 0.675.
        const nm = (i: number, x: string) => `c${String(i).padStart(2, '0')}${x}.md`;
        const files: string[] = [];
        const links: ReturnType<typeof E>[] = [];
        for (let i = 0; i < 10; i++) {
            files.push(nm(i, 'a'), nm(i, 'b'), nm(i, 'c'));
            links.push(E(nm(i, 'a'), nm(i, 'b')), E(nm(i, 'b'), nm(i, 'c')), E(nm(i, 'a'), nm(i, 'c')), E(nm(i, 'c'), nm((i + 1) % 10, 'a')));
        }
        const ring = detectCommunities(files.map(ref), links);
        const c = (p: string) => ring.members.get(ref(p))!;
        expect(new Set([...ring.members.values()].map((m) => m.communityId)).size).toBe(5);
        expect(c(nm(0, 'a')).communityId).toBe(c(nm(1, 'c')).communityId);
        expect(c(nm(0, 'a')).communityId).not.toBe(c(nm(2, 'a')).communityId);
        expect([0, 2, 4, 6, 8].map((i) => c(nm(i, 'a')).communityId)).toEqual([0, 1, 2, 3, 4]);
        // c00a has 2 of its 3 links inside the pair (the third comes from c09c); c00c has all 3 (its bridge goes to c01a).
        expect(c(nm(0, 'a')).confidence).toBe(0.666667);
        expect(c(nm(0, 'c')).confidence).toBe(1);
        expect(ring.modularity).toBe(0.675);
    });

    it('a tie keeps the lowest community id', () => {
        // a0-a1 and b0-b1 weigh 5; x links a1 and b0 with 1 each. Degrees: a0 5, a1 6, b0 6, b1 5, x 2; 2m = 24.
        // x has the same gain on both sides (1 − 11·2/24 = 0.083), so the lowest id (the a side) wins; confidence 1/2.
        const tie = detectCommunities(
            ['a0', 'a1', 'b0', 'b1', 'x'].map((p) => ref(`${p}.md`)),
            [E('a0.md', 'a1.md', 5), E('b0.md', 'b1.md', 5), E('a1.md', 'x.md'), E('x.md', 'b0.md')],
        );
        const t = (p: string) => tie.members.get(ref(p))!;
        expect([t('a0.md').communityId, t('a1.md').communityId, t('x.md').communityId]).toEqual([0, 0, 0]);
        expect([t('b0.md').communityId, t('b1.md').communityId]).toEqual([1, 1]);
        expect(t('x.md').confidence).toBe(0.5);
    });
});

describe('detectCommunities — label propagation above the cap', () => {
    const TRIANGLES = ['x1', 'x2', 'x3', 'y1', 'y2', 'y3'].map((p) => `${p}.md`);
    const TRI_EDGES = [
        E('x1.md', 'x2.md'), E('x2.md', 'x3.md'), E('x1.md', 'x3.md'),
        E('y1.md', 'y2.md'), E('y2.md', 'y3.md'), E('y1.md', 'y3.md'),
    ];

    it('finds the exact partition of two disjoint triangles', () => {
        // Sorted labels x1..y3 = 0..5. x1 hears labels 1 and 2 (tie → smallest, 1); x2 keeps 1 on a tie; x3 hears 1 twice. Same for y with 4.
        const r = detectCommunities(TRIANGLES.map(ref), TRI_EDGES, 4);
        expect(r.method).toBe('label-propagation');
        expect([...r.members].map(([k, m]) => [k, m.communityId, m.confidence])).toEqual(
            TRIANGLES.map((p, i) => [ref(p), i < 3 ? 0 : 1, 1]),
        );
        // 2 × (6/12 − (6/12)²) = 0.5
        expect(r.modularity).toBe(0.5);
    });

    it('is deterministic under shuffled input', () => {
        const a = detectCommunities(CLIQUES.map(ref), CLIQUE_EDGES, 4);
        const b = detectCommunities([...CLIQUES].reverse().map(ref), [...CLIQUE_EDGES].reverse(), 4);
        expect(a.method).toBe('label-propagation');
        same([...b.members], [...a.members]);
        expect(a.modularity).toBe(b.modularity);
    });

    it('a node keeps its own label when it is the larger of two tied labels', () => {
        // Path p0 - p1 - p3 - p2 (edges p0-p1, p1-p3, p2-p3), labels start as the sorted index 0..3. One sweep:
        //   p0 hears {1} and takes 1; p1 hears {1 (p0), 3 (p3)}, holds 1 and keeps it; p2 hears {3} and takes 3;
        //   p3 hears {1 (p1), 3 (p2)} equally while holding 3, so it KEEPS 3. (A "smallest label wins" rule would give it 1
        //   and merge everything into one community.) The second sweep changes nothing.
        const r = detectCommunities(
            ['p0', 'p1', 'p2', 'p3'].map((p) => ref(`${p}.md`)),
            [E('p0.md', 'p1.md'), E('p1.md', 'p3.md'), E('p2.md', 'p3.md')],
            2,
        );
        expect(r.method).toBe('label-propagation');
        expect([...r.members.values()].map((m) => [m.communityId, m.confidence])).toEqual([[0, 1], [0, 0.5], [1, 1], [1, 0.5]]);
    });
});

describe('Louvain tie rule', () => {
    it('a node stays put when its own community ties a lower-id one', () => {
        // a-b 12, b-d 7, c-d 1 (a..d = indexes 0..3). Degrees a 12, b 19, c 1, d 8; 2m = 40. Sweep 1: a joins b; b stays (6.3 against 3.2);
        // c joins d. d, in {c,d} (id 3), then weighs staying (1 − 1·8/40 = 0.8) against {a,b} (id 1: 7 − 31·8/40 = 0.8): an exact
        // tie, so it stays. A rule that sent ties to the lowest id would pull d into {a,b}.
        const order = ['a', 'b', 'c', 'd'].map((p) => ref(`${p}.md`));
        const adj = buildAdjacency(order, [E('a.md', 'b.md', 12), E('b.md', 'd.md', 7), E('c.md', 'd.md', 1)]);
        const { comm, moved } = localMoving(adj, [0, 0, 0, 0], 40);
        expect(moved).toBe(true);
        expect(comm).toEqual([0, 0, 1, 1]);
    });
});

describe('applyCommunities / buildGraph', () => {
    /** Two triangles in d1/ and d2/ joined by one edge; `lonely.md` has no edges. `kind` is the edge kind of every reference. */
    function input(kind: string): BuildGraphInput {
        const files = ['d1/a.md', 'd1/b.md', 'd1/c.md', 'd2/a.md', 'd2/b.md', 'd2/c.md', 'lonely.md'];
        const pairs: Array<[string, string]> = [
            ['d1/a.md', 'd1/b.md'], ['d1/b.md', 'd1/c.md'], ['d1/a.md', 'd1/c.md'],
            ['d2/a.md', 'd2/b.md'], ['d2/b.md', 'd2/c.md'], ['d2/a.md', 'd2/c.md'], ['d1/c.md', 'd2/a.md'],
        ];
        return {
            rootId: 'r', packs: [],
            extracted: files.map((p) => extractedOf(fileNode(p), pairs.filter(([f]) => f === p).map(([f, t]) => edge(f, t, kind)))),
            facts: [], relations: [],
        };
    }

    it('counts both `references` and `links`', () => {
        expect(COMMUNITY_EDGE_KINDS).toEqual(['references', 'links']);
    });

    it.each(['references', 'links'])('a %s-only graph is clustered, reports Q and the method, and leaves an unreferenced file bare', (kind) => {
        const g = buildGraph(input(kind));
        const node = (p: string) => g.nodes.find((n) => n.ref === ref(p))!;
        expect(node('d1/b.md').badges).toMatchObject({ communityId: 0, communityConfidence: 1 });
        expect(node('d2/b.md').badges).toMatchObject({ communityId: 1 });
        expect(node('d1/c.md').badges?.['communityConfidence']).toBe(0.666667);
        expect(node('lonely.md').badges).toBeUndefined();
        expect(g.stats.modularity).toBe(0.357143);
        expect(g.stats.communityMethod).toBe('louvain');
    });

    it('other edge kinds and stale relations are not references', () => {
        const nodes = CLIQUES.map((p) => fileNode(p));
        const q = applyCommunities(nodes, [
            edge('a1.md', 'a2.md', 'mentions'), edge('a2.md', 'a3.md', 'near-duplicate'), edge('a1.md', 'a3.md', 'links', { stale: true }),
        ]);
        expect(q.modularity).toBe(0);
        expect(nodes.every((n) => n.badges === undefined)).toBe(true);
    });

    it('equal input in a different order builds byte-identical graphs', () => {
        const base = input('links');
        const rev = { ...base, extracted: [...base.extracted].reverse().map((e) => ({ ...e, edges: [...e.edges].reverse() })) };
        same(buildGraph(rev), buildGraph(base));
    });

    it('the outline carries the badges and a labelled communities table', () => {
        const o = buildOutline(buildGraph(input('links')));
        expect(o.files.find((x) => x.path.endsWith('d1/c.md'))!.badges).toEqual({ communityId: 0, communityConfidence: 0.666667 });
        expect(o.files.find((x) => x.path.endsWith('lonely.md'))!.badges).toBeUndefined();
        expect(o.communities).toEqual([{ id: 0, label: 'd1', size: 3 }, { id: 1, label: 'd2', size: 3 }]);
    });

    it('the cut graph keeps modularity in its stats', () => {
        const g = buildGraph(input('links'), { maxNodes: 3 });
        expect(g.stats.truncated).toBe(true);
        expect(g.stats.modularity).toBe(0.357143);
        expect(g.outline!.communities!.length).toBe(2);
    });
});

describe('communityTable', () => {
    it('labels a community with its common directory, else with its most important member', () => {
        const mk = (p: string, importance: number) => fileNode(p, { importance, badges: { communityId: 0 } });
        expect(communityTable([mk('src/a/x.ts', 0.1), mk('src/a/y.ts', 0.2)])).toEqual([{ id: 0, label: 'src/a', size: 2 }]);
        expect(communityTable([mk('src/x.ts', 0.1), mk('lib/y.ts', 0.9)])).toEqual([{ id: 0, label: 'lib/y.ts', size: 2 }]);
    });

    it('builds the label from the outline-relative path (prefix stripped)', () => {
        const n = fileNode('src/a/x.ts', { badges: { communityId: 0 } });
        expect(communityTable([n], 'src:r::')).toEqual([{ id: 0, label: 'src/a', size: 1 }]);
    });

    it('makes labels unique: top member basename first, then a numeric suffix', () => {
        const mk = (p: string, id: number, importance: number) => fileNode(p, { importance, badges: { communityId: id } });
        // Communities 0 and 1 share the directory 'src', so each gets its top member's basename; 2 and 3 are already distinct.
        const table = communityTable([
            mk('src/a.ts', 0, 0.9), mk('src/b.ts', 0, 0.1),
            mk('src/c.ts', 1, 0.9), mk('src/d.ts', 1, 0.1),
            mk('src/lib/x.ts', 2, 0.9), mk('src/lib/y.ts', 2, 0.1),
            mk('doc/z.md', 3, 0.5),
        ]);
        expect(table.map((c) => c.label)).toEqual(['src · a.ts', 'src · c.ts', 'src/lib', 'doc']);
        expect(new Set(table.map((c) => c.label)).size).toBe(table.length);
    });

    it('falls back to a numeric suffix when the top member name also collides', () => {
        const mk = (p: string, id: number) => fileNode(p, { badges: { communityId: id } });
        // Both communities are {src/index.ts + a sibling in src}: same directory, same top basename (ties → smaller ref first).
        const table = communityTable([mk('src/index.ts', 0), mk('src/x.ts', 0), mk('src/index.ts', 1), mk('src/y.ts', 1)]);
        expect(table.map((c) => c.label)).toEqual(['src · index.ts', 'src · index.ts 2']);
    });
});

describe('placementOf', () => {
    it('is the partition when there is one, else the community, else nothing', () => {
        expect(placementOf({ partition: 'domain', badges: { communityId: 3 } })).toEqual({ kind: 'partition', id: 'domain' });
        expect(placementOf({ badges: { communityId: 3, communityConfidence: 0.5 } })).toEqual({ kind: 'community', id: 3 });
        expect(placementOf({ badges: { complexity: 4 } })).toBeUndefined();
        expect(placementOf({})).toBeUndefined();
    });

    it('a trace puts an unpartitioned file in its community lane', () => {
        const g = buildGraph(input2());
        const t = traceGraph(g, ref('d1/a.md'), { edgeKinds: ['links'] });
        expect(t.laneOf[ref('d1/a.md')]).toBe('community 0');
        expect(t.laneOf[ref('d2/a.md')]).toBe('community 1');
        expect(t.lanes).not.toContain('(none)');
    });

    it('with a registry the lanes are partitions first, then communities by numeric id', () => {
        const mk = (p: string, extra: Partial<KNode>) => fileNode(p, extra);
        const g: KnowledgeGraph = {
            scope: { rootId: 'r', packs: [], digest: 'd' },
            nodes: [
                mk('a.ts', { badges: { communityId: 10 } }), mk('b.ts', { badges: { communityId: 2 } }),
                mk('c.ts', { partition: 'domain' }), mk('d.ts', {}),
            ],
            edges: [edge('a.ts', 'b.ts', 'links'), edge('b.ts', 'c.ts', 'links'), edge('c.ts', 'd.ts', 'links')],
            findings: [], stats: { sources: 4, nodes: 4, edges: 3, stale: 0, truncated: false },
        };
        const t = traceGraph(g, ref('a.ts'), { edgeKinds: ['links'], registry: { partitions: [{ id: 'domain', order: 0, match: [] }], allow: {} } });
        // 'community 10' sorts before 'community 2' as text; by number it comes after.
        expect(t.lanes).toEqual(['domain', 'community 2', 'community 10', '(none)']);
    });
});

/** d1/a → d1/b → d1/c → d2/a → d2/b, plus the closing links, so both folders cluster. */
function input2(): BuildGraphInput {
    const pairs: Array<[string, string]> = [
        ['d1/a.md', 'd1/b.md'], ['d1/b.md', 'd1/c.md'], ['d1/a.md', 'd1/c.md'], ['d1/c.md', 'd2/a.md'],
        ['d2/a.md', 'd2/b.md'], ['d2/b.md', 'd2/c.md'], ['d2/a.md', 'd2/c.md'],
    ];
    const files = ['d1/a.md', 'd1/b.md', 'd1/c.md', 'd2/a.md', 'd2/b.md', 'd2/c.md'];
    return {
        rootId: 'r', packs: [],
        extracted: files.map((p) => extractedOf(fileNode(p), pairs.filter(([f]) => f === p).map(([f, t]) => edge(f, t, 'links')))),
        facts: [], relations: [],
    };
}
