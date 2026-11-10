/**
 * src/contract/regression.lineage.test.ts — a finding's id hashes ALL its refs (`findingId`), so a clone group that
 * gained a member read as one resolved finding plus one introduced finding (PRD 41 §C3). Identity v2 anchors the
 * group on its clone-group badge, and `matchLineage` reads the pair as ONE `changed (grew)`.
 */

import { CLONE_GROUP_BADGE, genericFindings, identityOf, matchLineage, type KEdge, type KNode, type KnowledgeGraph } from '../knowledge/index.js';

const ref = (p: string) => `src:r::${p}`;
const node = (p: string): KNode => ({ ref: ref(p), kind: 'file', label: p, pack: 'code', importance: 0, origin: 'extracted', badges: { [CLONE_GROUP_BADGE]: 'jscpd-1' } });
const dup = (a: string, b: string): KEdge => ({ from: ref(a), to: ref(b), kind: 'duplicate', weight: 1, origin: 'extracted' });
const clones = (files: string[]) => {
    const g: KnowledgeGraph = {
        scope: { rootId: 'r', packs: ['code'], digest: 'd' }, nodes: files.map(node), edges: files.slice(1).map((f) => dup(files[0]!, f)), findings: [],
        stats: { sources: files.length, nodes: files.length, edges: files.length - 1, stale: 0, truncated: false },
    };
    return genericFindings(g, new Map()).map((f) => identityOf(f));
};

describe('regression: lineage', () => {
    it('regression: a clone group gaining a member is changed (grew), never resolved + introduced', () => {
        const before = clones(['a.ts', 'b.ts']), after = clones(['a.ts', 'b.ts', 'c.ts']);
        const lineage = matchLineage(before, after);
        expect(lineage).toEqual([{ lineage: 'changed', change: 'grew', before: before[0]!.id, after: after[0]!.id }]);
        expect(lineage.some((e) => e.lineage === 'resolved' || e.lineage === 'introduced')).toBe(false);
    });
});
