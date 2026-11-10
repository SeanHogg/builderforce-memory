/**
 * src/knowledge/lineage.test.ts — the deterministic lineage matcher (PRD 41 §C3 rules 1–4).
 */

import { CLONE_GROUP_BADGE, genericFindings } from './findings.js';
import { identityOf, type FindingIdentity } from './findingIdentity.js';
import { matchLineage } from './lineage.js';
import type { KEdge, KNode, KnowledgeGraph } from './model.js';

const ref = (p: string) => `src:r::${p}`;
const node = (p: string, badges?: Record<string, string>): KNode => ({ ref: ref(p), kind: 'file', label: p, pack: 'code', importance: 0, origin: 'extracted', ...(badges ? { badges } : {}) });
const dup = (a: string, b: string): KEdge => ({ from: ref(a), to: ref(b), kind: 'duplicate', weight: 1, origin: 'extracted' });
const graph = (nodes: KNode[], edges: KEdge[]): KnowledgeGraph => ({
    scope: { rootId: 'r', packs: ['code'], digest: 'd' }, nodes, edges, findings: [],
    stats: { sources: nodes.length, nodes: nodes.length, edges: edges.length, stale: 0, truncated: false },
});
const cloneIdentities = (g: KnowledgeGraph) => genericFindings(g, new Map()).filter((f) => f.kind === 'clone-group').map((f) => identityOf(f));
const ident = (id: string, anchor: string, members: string[], anchorKind: FindingIdentity['anchorKind'] = 'file'): FindingIdentity => ({ id, anchor, anchorKind, members });

/** Seeded Fisher–Yates, so the shuffle is reproducible. */
function shuffled<T>(list: readonly T[], seed: number): T[] {
    const out = [...list];
    let s = seed;
    for (let i = out.length - 1; i > 0; i--) {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        const j = s % (i + 1);
        [out[i], out[j]] = [out[j]!, out[i]!];
    }
    return out;
}

describe('matchLineage', () => {
    it('a badged clone group gaining a member is changed (grew), not resolved + introduced', () => {
        const b = { [CLONE_GROUP_BADGE]: 'g1' };
        const before = cloneIdentities(graph([node('a.ts', b), node('b.ts', b)], [dup('a.ts', 'b.ts')]));
        const after = cloneIdentities(graph([node('a.ts', b), node('b.ts', b), node('c.ts', b)], [dup('a.ts', 'b.ts'), dup('a.ts', 'c.ts')]));
        expect(before[0]!.id).not.toBe(after[0]!.id);
        expect(matchLineage(before, after)).toEqual([{ lineage: 'changed', change: 'grew', before: before[0]!.id, after: after[0]!.id }]);
        expect(matchLineage(after, before)).toEqual([{ lineage: 'changed', change: 'shrank', before: after[0]!.id, after: before[0]!.id }]);
    });

    it('an unbadged clone group gaining a member that sorts first still matches (rule 3, Jaccard 2/3)', () => {
        const before = cloneIdentities(graph([node('m.ts'), node('n.ts')], [dup('m.ts', 'n.ts')]));
        const after = cloneIdentities(graph([node('a.ts'), node('m.ts'), node('n.ts')], [dup('m.ts', 'n.ts'), dup('a.ts', 'm.ts')]));
        expect(before[0]!.anchor).not.toBe(after[0]!.anchor);
        expect(matchLineage(before, after)).toEqual([{ lineage: 'changed', before: before[0]!.id, after: after[0]!.id }]);
    });

    it('an exact id is persisting', () => {
        const f = ident('k:1', 'k@1:file:x', ['x']);
        expect(matchLineage([f], [f])).toEqual([{ lineage: 'persisting', before: 'k:1', after: 'k:1' }]);
    });

    it('same anchor with equal member counts but different members is changed (moved)', () => {
        expect(matchLineage([ident('k:1', 'k@1:group:g', ['a', 'b'])], [ident('k:2', 'k@1:group:g', ['a', 'c'])]))
            .toEqual([{ lineage: 'changed', change: 'moved', before: 'k:1', after: 'k:2' }]);
    });

    it('two unrelated single-member findings in one file are never changed', () => {
        const before = [ident('god:1', 'god@1:symbol:src:r::a.ts#f', ['src:r::a.ts#f'], 'symbol')];
        const after = [ident('god:2', 'god@1:symbol:src:r::a.ts#g', ['src:r::a.ts#g'], 'symbol')];
        expect(matchLineage(before, after)).toEqual([
            { lineage: 'resolved', before: 'god:1' },
            { lineage: 'introduced', after: 'god:2' },
        ]);
    });

    it('rule 3 needs the same kind and the same anchor kind', () => {
        const before = [ident('a:1', 'a@1:file:x', ['x', 'y'])];
        expect(matchLineage(before, [ident('b:1', 'b@1:file:z', ['x', 'y'])]).map((e) => e.lineage)).toEqual(['resolved', 'introduced']);
        expect(matchLineage(before, [ident('a:2', 'a@1:symbol:z', ['x', 'y'], 'symbol')]).map((e) => e.lineage)).toEqual(['resolved', 'introduced']);
    });

    it('below Jaccard 0.5 is resolved + introduced', () => {
        expect(matchLineage([ident('c:1', 'c@1:file:a', ['a', 'b', 'c'])], [ident('c:2', 'c@1:file:z', ['a', 'x', 'y'])]).map((e) => e.lineage))
            .toEqual(['resolved', 'introduced']);
    });

    it('greedy: the highest Jaccard pairs first, ties by ascending id; any input order gives the same output', () => {
        const before = [
            ident('c:1', 'c@1:file:p', ['a', 'b', 'c', 'd']),
            ident('c:2', 'c@1:file:q', ['a', 'b', 'c', 'e']),
        ];
        const after = [
            ident('c:3', 'c@1:file:r', ['a', 'b', 'c', 'd', 'f']), // J(c:1)=4/5, J(c:2)=3/6
            ident('c:4', 'c@1:file:s', ['a', 'b', 'c', 'g']),      // J(c:1)=3/5, J(c:2)=3/5 — a tie
        ];
        const expected = [
            { lineage: 'changed', before: 'c:1', after: 'c:3' },
            { lineage: 'changed', before: 'c:2', after: 'c:4' },
        ];
        expect(matchLineage(before, after)).toEqual(expected);
        for (let seed = 1; seed <= 20; seed++) {
            expect(matchLineage(shuffled([...before, before[0]!], seed), shuffled(after, seed + 7))).toEqual(expected);
        }
    });
});
