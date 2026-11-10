/**
 * knowledge/rules/cycle.ts — `cycle`: one finding per connected group of cycle edges (PRD 38 §3.5).
 * Anchor (PRD 41 §C3): the group's sorted-first member.
 */

import { finding, refAnchor } from '../findingBuilder.js';
import { defineRule } from './factRule.js';
import { components } from './components.js';
import { packOfRef, type KnowledgeRuleExtra } from './knowledgeInput.js';

export const cycleRule = defineRule<KnowledgeRuleExtra>({
    id: 'cycle',
    version: 1,
    reads: [], // the knowledge graph only
    scope: 'global',
    extraInputs: ['knowledge'],
    derive: ({ extra: { knowledge: k } }) =>
        components(k.graph.edges.filter((e) => e.cycle))
            .map((refs) => finding('cycle', packOfRef(k, refs[0]!), refs, 'warn', { nodes: refs.length }, undefined, refAnchor(refs[0]!))),
});
