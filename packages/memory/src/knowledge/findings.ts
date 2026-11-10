/**
 * knowledge/findings.ts — what a person can act on (PRD 38 §3.5): the REGISTRY of the generic finding
 * rules and the compat facade over them (PRD 41 §B5).
 *
 * Generic findings, for every pack: cycle, clone group, near-duplicate, partition violation,
 * duplicate copy, unpartitioned share, stale knowledge, god module — plus `possible-same` entities.
 * Each is ONE FactRule under knowledge/rules/, registered here as data. Packs add their own through
 * {@link FindingRule}s. Every finding has a stable id (a hash of its kind and refs) and a `requestKey`
 * under `knowledgeGraph.fix.*`: the fixed, prefilled Fix-with-agent request, never a model call.
 */

import { finding, findingId, MAX_FINDING_NODES, requestKeyFor } from './findingBuilder.js';
import type { Finding, KnowledgeGraph, KnowledgePack, PartitionRegistry } from './model.js';
import type { SimilarPair } from '../retrieval/knowledge-index/types.js';
import { compareStr } from './refs.js';
import { knowledgeFacts, runKnowledgeRules, type KnowledgeRule } from './rules/knowledgeInput.js';
import { cycleRule } from './rules/cycle.js';
import { cloneGroupRule, CLONE_GROUP_BADGE } from './rules/cloneGroup.js';
import { nearDuplicateRule } from './rules/nearDuplicate.js';
import { partitionSeverity, partitionViolationRule } from './rules/partitionViolation.js';
import { duplicateCopyRule } from './rules/duplicateCopy.js';
import { unpartitionedRule } from './rules/unpartitioned.js';
import { staleRule } from './rules/stale.js';
import { godModuleRule } from './rules/godModule.js';
import { possibleSameRule } from './rules/possibleSame.js';
import { components } from './rules/components.js';

export { finding, findingId, requestKeyFor, MAX_FINDING_NODES, CLONE_GROUP_BADGE, partitionSeverity, components };

const SEVERITY_RANK = { error: 0, warn: 1, info: 2 } as const;

/** The generic finding rules, in emission order. */
export const GENERIC_FINDING_RULES: readonly KnowledgeRule[] = Object.freeze([
    cycleRule, cloneGroupRule, nearDuplicateRule, partitionViolationRule, duplicateCopyRule, unpartitionedRule, staleRule, godModuleRule,
]);

/** Every knowledge rule `computeFindings` runs: the generic ones plus `possible-same`. */
export const KNOWLEDGE_FINDING_RULES: readonly KnowledgeRule[] = Object.freeze([...GENERIC_FINDING_RULES, possibleSameRule]);

export function genericFindings(g: KnowledgeGraph, registries: ReadonlyMap<string, PartitionRegistry>): Finding[] {
    return runKnowledgeRules(GENERIC_FINDING_RULES, knowledgeFacts(g, registries));
}

/** Generic + possible-same + every pack's rules; deduped by id; severity, then id. */
export function computeFindings(
    g: KnowledgeGraph,
    packs: readonly KnowledgePack[],
    registries: ReadonlyMap<string, PartitionRegistry>,
    nearDuplicates: readonly SimilarPair[],
): Finding[] {
    const all = runKnowledgeRules(KNOWLEDGE_FINDING_RULES, knowledgeFacts(g, registries, nearDuplicates));
    for (const pack of [...packs].sort((a, b) => compareStr(a.id, b.id))) {
        for (const rule of pack.findings ?? []) {
            for (const f of rule.find(g)) {
                all.push({
                    ...f,
                    id: f.id || findingId(f.kind, f.nodes),
                    nodes: [...new Set(f.nodes)].sort(compareStr),
                    requestKey: f.requestKey ?? requestKeyFor(f.kind),
                });
            }
        }
    }
    const byId = new Map<string, Finding>();
    for (const f of all) if (!byId.has(f.id)) byId.set(f.id, f);
    return [...byId.values()].sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity] || compareStr(a.id, b.id));
}
