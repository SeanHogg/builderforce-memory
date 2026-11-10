/**
 * knowledge/rules/nearDuplicate.ts — `near-duplicate`: one finding per near-duplicate edge.
 * Anchor (PRD 41 §C3): the sorted-first ref of the pair.
 */

import { finding, refAnchor } from '../findingBuilder.js';
import { compareStr, round6 } from '../refs.js';
import { defineRule } from './factRule.js';
import { packOfRef, type KnowledgeRuleExtra } from './knowledgeInput.js';

export const nearDuplicateRule = defineRule<KnowledgeRuleExtra>({
    id: 'near-duplicate',
    version: 1,
    reads: [], // the knowledge graph only
    scope: 'global',
    extraInputs: ['knowledge'],
    derive: ({ extra: { knowledge: k } }) =>
        k.graph.edges.filter((e) => e.kind === 'near-duplicate').map((e) => {
            const first = compareStr(e.from, e.to) <= 0 ? e.from : e.to;
            return finding('near-duplicate', packOfRef(k, e.from), [e.from, e.to], 'info', { similarity: round6(e.weight) }, undefined, refAnchor(first));
        }),
});
