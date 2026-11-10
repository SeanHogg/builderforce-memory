/**
 * knowledge/rules/stale.ts — `stale`: one finding per source whose stored knowledge went stale (an
 * unanchored stale node counts as its own source). Anchor (PRD 41 §C3): the stale source.
 */

import { finding, refAnchor } from '../findingBuilder.js';
import type { NodeRef } from '../model.js';
import { aboutTargets, FALLBACK_PACK_ID } from '../refs.js';
import { defineRule } from './factRule.js';
import type { KnowledgeRuleExtra } from './knowledgeInput.js';

export const staleRule = defineRule<KnowledgeRuleExtra>({
    id: 'stale',
    version: 1,
    reads: [], // the knowledge graph only (stale flags)
    scope: 'global',
    extraInputs: ['knowledge'],
    derive: ({ extra: { knowledge: k } }) => {
        const about = aboutTargets(k.graph.edges);
        const bySource = new Map<NodeRef, NodeRef[]>();
        for (const n of k.graph.nodes) {
            if (!n.stale) continue;
            const target = about.get(n.ref) ?? n.ref;
            const list = bySource.get(target) ?? [];
            list.push(n.ref);
            bySource.set(target, list);
        }
        return [...bySource].map(([source, refs]) =>
            finding('stale', k.byRef.get(source)?.pack ?? FALLBACK_PACK_ID, [source, ...refs], 'warn', { count: refs.length }, undefined, refAnchor(source)));
    },
});
