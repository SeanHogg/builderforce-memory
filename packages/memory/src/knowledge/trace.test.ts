/**
 * src/knowledge/trace.test.ts
 * traceGraph's order (weight, then ref; each node once; back/cycle steps) and the Mermaid
 * sequence diagram drawn from it.
 */

import { toMermaid, traceGraph } from './trace.js';
import type { KnowledgeGraph, NodeRef, PartitionRegistry } from './model.js';
import { edge, fileNode, ref, ROOT } from './testkit.test-util.js';

const REG: PartitionRegistry = {
    partitions: [
        { id: 'presentation', order: 0, match: ['ui/**'] },
        { id: 'application', order: 1, match: ['app/**'] },
        { id: 'domain', order: 2, match: ['dom/**'] },
        { id: 'infrastructure', order: 3, match: ['infra/**'] },
    ],
    allow: {},
};

const graph: KnowledgeGraph = {
    scope: { rootId: ROOT, packs: ['code'], digest: 'd' },
    nodes: [
        fileNode('ui/A.ts', { partition: 'presentation' }),
        fileNode('app/B.ts', { partition: 'application' }),
        fileNode('dom/C.ts', { partition: 'domain' }),
        fileNode('infra/D.ts', { partition: 'infrastructure' }),
    ],
    edges: [
        edge('ui/A.ts', 'dom/C.ts', 'references'),
        edge('ui/A.ts', 'app/B.ts', 'references', { weight: 2 }),
        edge('app/B.ts', 'infra/D.ts', 'references', { verdict: 'skips' }),
        edge('app/B.ts', 'dom/C.ts', 'references'),
        edge('dom/C.ts', 'ui/A.ts', 'references', { cycle: true }),
        edge('ui/A.ts', 'infra/D.ts', 'contains'),
    ],
    findings: [],
    stats: { sources: 4, nodes: 4, edges: 6, stale: 0, truncated: false },
};

const name = (r: NodeRef) => r.slice(r.lastIndexOf('/') + 1, -3);

describe('traceGraph', () => {
    const t = traceGraph(graph, ref('ui/A.ts'), { edgeKinds: ['references'], registry: REG });

    it('walks heaviest edge first, then by ref; each node once; marks back and cycle steps', () => {
        expect(t.steps.map((s) => `${name(s.from)}>${name(s.to)}@${s.depth}${s.back ? ' back' : ''}${s.cycle ? ' cycle' : ''}`)).toEqual([
            'A>B@1',
            'B>C@2',
            'C>A@3 back cycle',
            'B>D@2',
            'A>C@1 back',
        ]);
        expect(t.truncated).toBe(false);
    });

    it('orders lanes by partition order', () => {
        expect(t.lanes).toEqual(['presentation', 'application', 'domain', 'infrastructure']);
    });

    it('is stable across edge order', () => {
        const shuffled = { ...graph, edges: [...graph.edges].reverse() };
        expect(traceGraph(shuffled, ref('ui/A.ts'), { edgeKinds: ['references'], registry: REG })).toEqual(t);
    });

    it('stops at maxSteps and maxDepth', () => {
        expect(traceGraph(graph, ref('ui/A.ts'), { edgeKinds: ['references'], maxSteps: 2 })).toMatchObject({ truncated: true });
        const shallow = traceGraph(graph, ref('ui/A.ts'), { edgeKinds: ['references'], maxDepth: 1 });
        expect(shallow.steps.every((s) => s.depth === 1)).toBe(true);
        expect(shallow.truncated).toBe(true);
    });
});

describe('toMermaid', () => {
    it('draws lanes as participants and steps as messages', () => {
        const t = traceGraph(graph, ref('ui/A.ts'), { edgeKinds: ['references'], registry: REG });
        expect(toMermaid(t, name)).toBe([
            'sequenceDiagram',
            '    participant P0 as presentation',
            '    participant P1 as application',
            '    participant P2 as domain',
            '    participant P3 as infrastructure',
            '    P0->>P1: A references B',
            '    P1->>P2: B references C',
            '    P2-->>P0: C references A [cycle]',
            '    P1->>P3: B references D [skips]',
            '    P0-->>P2: A references C',
            '',
        ].join('\n'));
    });
});
