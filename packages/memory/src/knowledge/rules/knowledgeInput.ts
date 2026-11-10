/**
 * knowledge/rules/knowledgeInput.ts — the non-fact input of the generic finding rules ported from
 * findings.ts (PRD 41 §B5): the built knowledge graph plus the partition registries and near-duplicate
 * pairs. These rules are `global` with `reads: []` and `extraInputs: ['knowledge']`, so in an incremental
 * evaluation they rerun exactly when {@link knowledgeInputHash} of this input moves.
 */

import type { SimilarPair } from '../../retrieval/knowledge-index/types.js';
import type { Finding } from '../findingModel.js';
import type { KNode, KnowledgeGraph, NodeRef, PartitionRegistry } from '../model.js';
import { canonicalJson, compareStr, FALLBACK_PACK_ID } from '../refs.js';
import { sha256Hex } from '../../retrieval/knowledge-index/sha256.js';
import type { FactRule } from './factRule.js';

export interface KnowledgeFacts {
    graph: KnowledgeGraph;
    byRef: ReadonlyMap<NodeRef, KNode>;
    registries: ReadonlyMap<string, PartitionRegistry>;
    nearDuplicates: readonly SimilarPair[];
}

export interface KnowledgeRuleExtra {
    knowledge: KnowledgeFacts;
}

/** A rule over the knowledge graph. */
export type KnowledgeRule = FactRule<KnowledgeRuleExtra>;

/** The extra input name the knowledge rules declare. */
export const KNOWLEDGE_INPUT = 'knowledge';

export function knowledgeFacts(
    graph: KnowledgeGraph,
    registries: ReadonlyMap<string, PartitionRegistry>,
    nearDuplicates: readonly SimilarPair[] = [],
): KnowledgeFacts {
    return { graph, byRef: new Map(graph.nodes.map((n) => [n.ref, n] as const)), registries, nearDuplicates };
}

/**
 * THE hash a host passes as `extraHashes.knowledge`: sha256 over the canonical JSON of the graph digest, the partition
 * registries (sorted by pack) and the near-duplicate pairs (sorted). The graph digest alone misses a registry edit
 * (`unpartitioned`) or pairs passed beside the graph (`possible-same`).
 */
export function knowledgeInputHash(k: KnowledgeFacts): string {
    return sha256Hex(canonicalJson({
        digest: k.graph.scope.digest,
        registries: [...k.registries].sort((a, b) => compareStr(a[0], b[0])),
        nearDuplicates: k.nearDuplicates.map((p) => canonicalJson(p)).sort(compareStr),
    }));
}

/** The pack of a ref's node, else the fallback pack. */
export const packOfRef = (k: KnowledgeFacts, ref: NodeRef): string => k.byRef.get(ref)?.pack ?? FALLBACK_PACK_ID;

/** Runs knowledge rules once, in registry order, non-incrementally (the `computeFindings` path). */
export function runKnowledgeRules(rules: readonly KnowledgeRule[], knowledge: KnowledgeFacts): Finding[] {
    const files = new Map();
    return rules.flatMap((r) => r.derive({ scope: r.scope, files, extra: { knowledge } }));
}
