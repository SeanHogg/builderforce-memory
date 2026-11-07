/**
 * src/knowledge/systems.test.ts
 * Systems (PRD 41 A9): placement of a path, the system/app/copyOf stamps on file nodes, no layer
 * verdict (so no partition-violation) across two systems, the duplicate-copy finding and graph meta.
 */

import { buildGraph } from './build.js';
import type { BuildGraphInput, GraphSystem, KnowledgePack } from './model.js';
import { placeInSystem } from './systems.js';
import { edge, extractedOf, fileNode, ROOT } from './testkit.test-util.js';

const PACK: KnowledgePack = {
    id: 'docs',
    parsers: [],
    partitions: {
        partitions: [
            { id: 'ui', order: 0, match: ['**/ui/**'] },
            { id: 'data', order: 1, match: ['**/data/**'] },
        ],
        allow: {},
    },
};

const sys = (path: string, extra: Partial<GraphSystem> = {}): GraphSystem => ({ path, name: path || 'root', files: 0, apps: [], ...extra });

function input(systems: GraphSystem[], extraEdges: Array<[string, string]> = []): BuildGraphInput {
    const files = ['a/data/x.ts', 'a/ui/y.ts', 'b/ui/z.ts', 'b/data/w.ts'];
    return {
        rootId: ROOT,
        packs: [PACK],
        extracted: [
            extractedOf(fileNode('a/data/x.ts'), [edge('a/data/x.ts', 'a/ui/y.ts', 'references'), ...extraEdges.map(([f, t]) => edge(f, t, 'references'))]),
            ...files.slice(1).map((f) => extractedOf(fileNode(f))),
        ],
        facts: [],
        relations: [],
        systems,
    };
}

describe('placeInSystem', () => {
    const systems = [sys('a', { apps: [{ path: 'a/packages/web', name: 'web' }] }), sys('a/vendor'), sys('b')];

    it('takes the deepest system, then its deepest app', () => {
        expect(placeInSystem(systems, 'a/packages/web/src/x.ts')).toEqual({ system: 'a', app: 'a/packages/web' });
        expect(placeInSystem(systems, 'a/vendor/lib.ts')).toEqual({ system: 'a/vendor' });
        expect(placeInSystem(systems, 'ab/x.ts')).toBeUndefined();
    });

    it('a system at the root holds every path', () => {
        expect(placeInSystem([sys('')], 'x/y.ts')).toEqual({ system: '' });
    });

    it('marks files of a copy', () => {
        expect(placeInSystem([sys('c', { copyOf: 'a' })], 'c/x.ts')).toEqual({ system: 'c', copyOf: 'a' });
    });
});

describe('buildGraph — systems', () => {
    it('stamps file nodes and reports systems with their file counts', () => {
        const g = buildGraph(input([sys('a'), sys('b', { truncated: true })]));
        const node = (p: string) => g.nodes.find((n) => n.label === p)!;
        expect(node('a/data/x.ts').system).toBe('a');
        expect(node('b/ui/z.ts').system).toBe('b');
        expect(g.meta?.systems?.map((s) => [s.path, s.files, s.truncated])).toEqual([['a', 2, undefined], ['b', 2, true]]);
    });

    it('a root with no systems has no meta and no stamps', () => {
        const g = buildGraph(input([]));
        expect(g.meta).toBeUndefined();
        expect(g.nodes.every((n) => n.system === undefined)).toBe(true);
    });

    it('computes verdicts inside a system', () => {
        const g = buildGraph(input([sys('a'), sys('b')]));
        expect(g.edges.find((e) => e.to.endsWith('a/ui/y.ts'))?.verdict).toBe('upward');
        expect(g.findings.filter((f) => f.kind === 'partition-violation')).toHaveLength(1);
    });

    it('no verdict and no partition-violation across two systems', () => {
        const g = buildGraph(input([sys('a'), sys('b')], [['a/data/x.ts', 'b/ui/z.ts']]));
        const across = g.edges.find((e) => e.to.endsWith('b/ui/z.ts'))!;
        expect(across.verdict).toBeUndefined();
        expect(g.findings.filter((f) => f.kind === 'partition-violation').every((f) => !f.nodes.some((n) => n.endsWith('b/ui/z.ts')))).toBe(true);
    });

    it('a link from a file to a path that is not there, in another system, gets no verdict', () => {
        // b/ui/gone.ts is not a file of the graph: it becomes a `missing` node, which still lives in system b.
        const across = buildGraph(input([sys('a'), sys('b')], [['a/data/x.ts', 'b/ui/gone.ts']]));
        const missing = across.edges.find((e) => e.to.endsWith('b/ui/gone.ts'))!;
        expect(across.nodes.find((n) => n.label === 'gone.ts')?.kind).toBe('missing');
        expect(missing.verdict).toBeUndefined();
        // Inside one system the same missing link is judged as before, and a missing node is not stamped (a file field).
        const within = buildGraph(input([sys('a'), sys('b')], [['a/data/x.ts', 'a/ui/gone.ts']]));
        expect(within.edges.find((e) => e.to.endsWith('a/ui/gone.ts'))?.verdict).toBe('upward');
        expect(within.nodes.find((n) => n.label === 'gone.ts')?.system).toBeUndefined();
    });

    it('the same edge IS a violation when the root is one system', () => {
        const g = buildGraph(input([sys('')], [['a/data/x.ts', 'b/ui/z.ts']]));
        expect(g.edges.find((e) => e.to.endsWith('b/ui/z.ts'))?.verdict).toBe('upward');
    });

    it('a copy gets a duplicate-copy finding and its files carry copyOf', () => {
        const g = buildGraph(input([sys('a'), sys('b', { copyOf: 'a', copySimilarity: 0.97, copyReason: 'no-git' })]));
        const f = g.findings.find((x) => x.kind === 'duplicate-copy')!;
        expect(f.severity).toBe('warn');
        expect(f.params).toEqual({ copy: 'b', original: 'a', similarity: 0.97, reason: 'no-git' });
        expect(f.nodes).toHaveLength(2);
        expect(g.nodes.find((n) => n.label === 'b/ui/z.ts')?.copyOf).toBe('a');
        expect(g.nodes.find((n) => n.label === 'a/ui/y.ts')?.copyOf).toBeUndefined();
    });

    it('systems join the digest, in any order', () => {
        const a = buildGraph(input([sys('a'), sys('b')]));
        expect(buildGraph(input([sys('b'), sys('a')])).scope.digest).toBe(a.scope.digest);
        expect(buildGraph(input([sys('a')])).scope.digest).not.toBe(a.scope.digest);
    });
});
