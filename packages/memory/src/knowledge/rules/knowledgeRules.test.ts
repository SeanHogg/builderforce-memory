/**
 * src/knowledge/rules/knowledgeRules.test.ts — the generic findings ported to one FactRule per file (PRD 41 §B5):
 * the registry is well-formed, and the incremental evaluator gives the same findings as the compat facade.
 */

import { GENERIC_FINDING_RULES, KNOWLEDGE_FINDING_RULES, computeFindings, genericFindings } from '../findings.js';
import type { KEdge, KNode, KnowledgeGraph, NodeRef, PartitionRegistry } from '../model.js';
import type { SimilarPair } from '../../retrieval/knowledge-index/types.js';
import { compareStr } from '../refs.js';
import { evaluateRules } from './evaluateRules.js';
import { knowledgeFacts, knowledgeInputHash, KNOWLEDGE_INPUT, type KnowledgeFacts } from './knowledgeInput.js';

const ref = (p: string) => `src:r::${p}` as NodeRef;
const node = (p: string, extra: Partial<KNode> = {}): KNode => ({ ref: ref(p), kind: 'file', label: p, pack: 'code', importance: 0, origin: 'extracted', ...extra });
const edge = (a: string, b: string, extra: Partial<KEdge>): KEdge => ({ from: ref(a), to: ref(b), kind: 'references', weight: 1, origin: 'extracted', ...extra });

const graph = (): KnowledgeGraph => ({
    scope: { rootId: 'r', packs: ['code'], digest: 'd1' },
    nodes: [node('a.ts', { partition: 'ui' }), node('b.ts', { partition: 'db' }), node('c.ts', { stale: true }), node('d.ts', { badges: { patterns: 'god-module', lines: 2000 } })],
    edges: [
        edge('a.ts', 'b.ts', { cycle: true, verdict: 'upward' }), edge('b.ts', 'a.ts', { cycle: true }),
        edge('a.ts', 'c.ts', { kind: 'duplicate' }), edge('b.ts', 'd.ts', { kind: 'near-duplicate', weight: 0.91 }),
    ],
    findings: [],
    stats: { sources: 4, nodes: 4, edges: 4, stale: 1, truncated: false },
});

describe('knowledge finding rules registry', () => {
    it('one global rule per finding kind, each reading the knowledge input', () => {
        const ids = KNOWLEDGE_FINDING_RULES.map((r) => r.id);
        expect(new Set(ids).size).toBe(ids.length);
        expect([...ids].sort(compareStr)).toEqual([
            'clone-group', 'cycle', 'duplicate-copy', 'god-module', 'near-duplicate', 'partition-violation', 'possible-same', 'stale', 'unpartitioned',
        ]);
        for (const r of KNOWLEDGE_FINDING_RULES) {
            // They read the knowledge input alone, so they declare no fact kind (and see no files).
            expect({ id: r.id, scope: r.scope, extra: r.extraInputs, reads: r.reads }).toEqual({ id: r.id, scope: 'global', extra: [KNOWLEDGE_INPUT], reads: [] });
        }
        expect(KNOWLEDGE_FINDING_RULES.slice(0, GENERIC_FINDING_RULES.length)).toEqual(GENERIC_FINDING_RULES);
    });

    it('the facade emits one finding per kind the fixture holds', () => {
        const kinds = genericFindings(graph(), new Map()).map((f) => f.kind).sort(compareStr);
        expect(kinds).toEqual(['clone-group', 'cycle', 'god-module', 'near-duplicate', 'partition-violation', 'stale']);
    });

    const worldOf = (knowledge: KnowledgeFacts) => ({ files: new Map(), extra: { knowledge }, extraHashes: { knowledge: knowledgeInputHash(knowledge) } });

    it('incremental evaluation equals the facade, and reuses every rule while the knowledge input hash holds', () => {
        const g = graph();
        const world = worldOf(knowledgeFacts(g, new Map()));
        const first = evaluateRules(KNOWLEDGE_FINDING_RULES, world);
        const facade = computeFindings(g, [], new Map(), []);
        expect(first.findings).toEqual([...facade].sort((a, b) => compareStr(a.id, b.id)));
        const second = evaluateRules(KNOWLEDGE_FINDING_RULES, worldOf(knowledgeFacts(graph(), new Map())), first.rows);
        expect(second.trace.every((t) => t.outcome === 'reused')).toBe(true);
    });

    it('toggling a registry `allow` (same graph digest) moves the input hash and reruns `unpartitioned`', () => {
        const registry = (allow?: Record<string, string[]>): PartitionRegistry => ({ partitions: [{ id: 'ui', order: 0, match: ['**'] }], ...(allow ? { allow } : {}) });
        const before = knowledgeFacts(graph(), new Map([['code', registry()]]));
        const after = knowledgeFacts(graph(), new Map([['code', registry({ ui: [] })]]));
        expect(knowledgeInputHash(after)).not.toBe(knowledgeInputHash(before));
        const first = evaluateRules(KNOWLEDGE_FINDING_RULES, worldOf(before));
        expect(first.findings.some((f) => f.kind === 'unpartitioned')).toBe(false);
        const second = evaluateRules(KNOWLEDGE_FINDING_RULES, worldOf(after), first.rows);
        expect(second.trace.find((t) => t.rule === 'unpartitioned')?.outcome).toBe('ran');
        expect(second.findings.some((f) => f.kind === 'unpartitioned')).toBe(true);
    });

    it('the input hash covers near-duplicate pairs passed beside the graph, in any order', () => {
        const pairs = [{ a: 'ent:code:alpha', b: 'ent:code:beta', similarity: 0.9 }, { a: 'ent:code:x', b: 'ent:code:y', similarity: 0.8 }] as SimilarPair[];
        const none = knowledgeInputHash(knowledgeFacts(graph(), new Map()));
        const some = knowledgeInputHash(knowledgeFacts(graph(), new Map(), pairs));
        expect(some).not.toBe(none);
        expect(knowledgeInputHash(knowledgeFacts(graph(), new Map(), [...pairs].reverse()))).toBe(some);
    });
});
