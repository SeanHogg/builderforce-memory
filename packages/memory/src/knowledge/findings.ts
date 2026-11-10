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
import type { Finding, FindingsProvider, KnowledgeGraph, KnowledgePack, PartitionRegistry } from './model.js';
import type { SimilarPair } from '../retrieval/knowledge-index/types.js';
import { compareStr } from './refs.js';
import { knowledgeFacts, runKnowledgeRules, type KnowledgeFacts, type KnowledgeRule } from './rules/knowledgeInput.js';
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

/** A finding from outside the generic rules (a pack rule, a findings provider) in the one shape: id, sorted nodes, request key. */
const normalizeFinding = (f: Finding): Finding => ({
    ...f,
    id: f.id || findingId(f.kind, f.nodes),
    nodes: [...new Set(f.nodes)].sort(compareStr),
    requestKey: f.requestKey ?? requestKeyFor(f.kind),
});

/** Deduped by id (first wins), sorted by severity then id: the one order of every finding list. */
function finishFindings(all: readonly Finding[]): Finding[] {
    const byId = new Map<string, Finding>();
    for (const f of all) if (!byId.has(f.id)) byId.set(f.id, f);
    return [...byId.values()].sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity] || compareStr(a.id, b.id));
}

/** Generic + possible-same + every pack's rules; deduped by id; severity, then id. */
export function computeFindings(
    g: KnowledgeGraph,
    packs: readonly KnowledgePack[],
    registries: ReadonlyMap<string, PartitionRegistry>,
    nearDuplicates: readonly SimilarPair[],
): Finding[] {
    return finishFindings(internalFindings(g, packs, registries, nearDuplicates, new Set()));
}

/** The built-in and pack findings, without any of the `owned` kinds (generic rules owned by id are not run at all). */
function internalFindings(
    g: KnowledgeGraph,
    packs: readonly KnowledgePack[],
    registries: ReadonlyMap<string, PartitionRegistry>,
    nearDuplicates: readonly SimilarPair[],
    owned: ReadonlySet<string>,
    knowledge: KnowledgeFacts = knowledgeFacts(g, registries, nearDuplicates),
): Finding[] {
    const rules = owned.size === 0 ? KNOWLEDGE_FINDING_RULES : KNOWLEDGE_FINDING_RULES.filter((r) => !owned.has(r.id));
    const all = runKnowledgeRules(rules, knowledge);
    for (const pack of [...packs].sort((a, b) => compareStr(a.id, b.id))) {
        for (const rule of pack.findings ?? []) {
            for (const f of rule.find(g)) all.push(normalizeFinding(f));
        }
    }
    return owned.size === 0 ? all : all.filter((f) => !owned.has(f.kind));
}

/**
 * {@link computeFindings} with a {@link FindingsProvider}: the provider's findings replace every built-in and pack finding
 * of the kinds it owns (`provider.kinds`); the provider sees the WHOLE graph (`buildGraph` calls this before truncation)
 * and its output is kept only for its own kinds, normalized like a pack finding. Same dedupe and order.
 */
export function computeFindingsWith(
    g: KnowledgeGraph,
    packs: readonly KnowledgePack[],
    registries: ReadonlyMap<string, PartitionRegistry>,
    nearDuplicates: readonly SimilarPair[],
    provider: FindingsProvider,
): Finding[] {
    const owned = new Set(provider.kinds);
    // ONE knowledge-rule input, shared by the built-in rules and the provider.
    const knowledge = knowledgeFacts(g, registries, nearDuplicates);
    const internal = internalFindings(g, packs, registries, nearDuplicates, owned, knowledge);
    const supplied = provider.findings({ graph: g, knowledge })
        .filter((f) => owned.has(f.kind))
        .map(normalizeFinding);
    return finishFindings([...internal, ...supplied]);
}
