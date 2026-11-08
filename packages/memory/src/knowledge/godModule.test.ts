/**
 * src/knowledge/godModule.test.ts — the `god-module` finding: measures beside limits, omitted when
 * the badge is absent, stable id, deterministic order.
 */

import { computeFindings, findingId } from './findings.js';
import { GOD_MODULE_LIMITS, godModuleFindings } from './godModule.js';
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

describe('GOD_MODULE_LIMITS', () => {
    it('is frozen at the documented limits', () => {
        expect(GOD_MODULE_LIMITS).toEqual({ lines: 1000, exports: 40, complexity: 60 });
        expect(Object.isFrozen(GOD_MODULE_LIMITS)).toBe(true);
    });
});

describe('godModuleFindings', () => {
    it('regression: a file with the god-module badge yields a heuristic-sized warn finding with measures and limits', () => {
        const g = graph([fileNode('src:r::big.ts', { patterns: 'adapter,god-module', lines: 1500, exports: 12, maxComplexity: 70 })]);
        const found = godModuleFindings(g);
        expect(found).toHaveLength(1);
        expect(found[0]).toMatchObject({
            kind: 'god-module', pack: 'code', severity: 'warn', nodes: ['src:r::big.ts'], requestKey: 'knowledgeGraph.fix.godModule',
            params: { lines: 1500, exports: 12, complexity: 70, linesLimit: 1000, exportsLimit: 40, complexityLimit: 60 },
        });
        expect(found[0]!.id).toBe(findingId('god-module', ['src:r::big.ts']));
    });

    it('omits a measure whose badge is absent, never the limits', () => {
        const [f] = godModuleFindings(graph([fileNode('src:r::old.ts', { patterns: 'god-module', exports: 55 })]));
        expect(f!.params).toEqual({ exports: 55, linesLimit: 1000, exportsLimit: 40, complexityLimit: 60 });
    });

    it('ignores files without the badge, a lookalike pattern id and non-file nodes', () => {
        const g = graph([
            fileNode('src:r::a.ts', { patterns: 'adapter' }),
            fileNode('src:r::b.ts', { patterns: 'not-god-module-ish' }),
            fileNode('src:r::c.ts', { patterns: 'god-module' }, 'symbol'),
        ]);
        expect(godModuleFindings(g)).toEqual([]);
    });

    it('is part of computeFindings, in a deterministic order', () => {
        const g = graph([
            fileNode('src:r::z.ts', { patterns: 'god-module' }),
            fileNode('src:r::a.ts', { patterns: 'god-module' }),
        ]);
        const a = computeFindings(g, [], new Map(), []).filter((f) => f.kind === 'god-module');
        const b = computeFindings({ ...g, nodes: [...g.nodes].reverse() }, [], new Map(), []).filter((f) => f.kind === 'god-module');
        expect(a).toHaveLength(2);
        expect(a.map((f) => f.id)).toEqual(b.map((f) => f.id));
    });
});
