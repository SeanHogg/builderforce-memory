/**
 * src/knowledge/ioInLoop.test.ts — the `io-in-loop` finding: one per badged file, stable id,
 * deterministic order; unbadged files and non-file nodes are ignored.
 */

import { computeFindings, findingId } from './findings.js';
import { ioInLoopFindings } from './ioInLoop.js';
import type { KNode, KnowledgeGraph } from './model.js';

function fileNode(ref: string, badges: Record<string, number | string>, kind = 'file'): KNode {
    return { ref, kind, label: ref, pack: 'code', importance: 0, origin: 'extracted' as KNode['origin'], badges };
}

function graph(nodes: KNode[]): KnowledgeGraph {
    return {
        scope: { rootId: 'r', packs: ['code'], digest: 'd' }, nodes, edges: [], findings: [],
        stats: { sources: nodes.length, nodes: nodes.length, edges: 0, stale: 0, truncated: false },
    };
}

describe('ioInLoopFindings', () => {
    it('a file with the io-in-loop badge yields one warn finding with a stable id and fix key', () => {
        const found = ioInLoopFindings(graph([fileNode('src:r::sync.ts', { patterns: 'batched-io,io-in-loop' })]));
        expect(found).toHaveLength(1);
        expect(found[0]).toMatchObject({
            kind: 'io-in-loop', pack: 'code', severity: 'warn', nodes: ['src:r::sync.ts'], requestKey: 'knowledgeGraph.fix.ioInLoop', params: {},
        });
        expect(found[0]!.id).toBe(findingId('io-in-loop', ['src:r::sync.ts']));
    });

    it('ignores files without the badge, a lookalike pattern id and non-file nodes', () => {
        const g = graph([
            fileNode('src:r::a.ts', { patterns: 'batched-io' }),
            fileNode('src:r::b.ts', { patterns: 'no-io-in-loop' }),
            fileNode('src:r::c.ts', { patterns: 'io-in-loop' }, 'symbol'),
        ]);
        expect(ioInLoopFindings(g)).toEqual([]);
    });

    it('is part of computeFindings, in a deterministic order', () => {
        const g = graph([
            fileNode('src:r::z.ts', { patterns: 'io-in-loop' }),
            fileNode('src:r::a.ts', { patterns: 'io-in-loop' }),
        ]);
        const a = computeFindings(g, [], new Map(), []).filter((f) => f.kind === 'io-in-loop');
        const b = computeFindings({ ...g, nodes: [...g.nodes].reverse() }, [], new Map(), []).filter((f) => f.kind === 'io-in-loop');
        expect(a).toHaveLength(2);
        expect(a.map((f) => f.id)).toEqual(b.map((f) => f.id));
    });
});
