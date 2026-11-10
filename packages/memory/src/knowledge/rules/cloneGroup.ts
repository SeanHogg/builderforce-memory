/**
 * knowledge/rules/cloneGroup.ts — `clone-group`: duplicate edges grouped by the clone-group badge
 * (e.g. a jscpd issue id), else by connected component. Anchor (PRD 41 §C3): the badge, else the
 * sorted-first member — so a group that gains a member keeps its identity (lineage `changed (grew)`).
 */

import { finding, refAnchor, type FindingAnchor } from '../findingBuilder.js';
import type { KEdge, NodeRef } from '../model.js';
import { compareStr } from '../refs.js';
import { defineRule } from './factRule.js';
import { components } from './components.js';
import { packOfRef, type KnowledgeRuleExtra } from './knowledgeInput.js';

/** Node badge naming the clone group (e.g. a jscpd issue id) its `duplicate` edges belong to. */
export const CLONE_GROUP_BADGE = 'cloneGroup';

export const cloneGroupRule = defineRule<KnowledgeRuleExtra>({
    id: 'clone-group',
    version: 1,
    reads: [], // the knowledge graph only (its digest covers the duplicate edges)
    scope: 'global',
    extraInputs: ['knowledge'],
    derive: ({ extra: { knowledge: k } }) => {
        const byBadge = new Map<string, KEdge[]>();
        const unbadged: KEdge[] = [];
        for (const e of k.graph.edges) {
            if (e.kind !== 'duplicate') continue;
            const badge = k.byRef.get(e.from)?.badges?.[CLONE_GROUP_BADGE] ?? k.byRef.get(e.to)?.badges?.[CLONE_GROUP_BADGE];
            if (badge === undefined) { unbadged.push(e); continue; }
            const list = byBadge.get(String(badge)) ?? [];
            list.push(e);
            byBadge.set(String(badge), list);
        }
        const groups: Array<{ group: string; refs: NodeRef[] }> = [
            ...[...byBadge].map(([group, es]) => ({ group, refs: es.flatMap((e) => [e.from, e.to]) })),
            ...components(unbadged).map((refs) => ({ group: '', refs })),
        ];
        return groups.map(({ group, refs }) => {
            const unique = [...new Set(refs)];
            const first = unique.sort(compareStr)[0]!;
            const anchor: FindingAnchor = group ? { anchor: `${CLONE_GROUP_BADGE}:${group}`, anchorKind: 'group' } : refAnchor(first);
            return finding('clone-group', packOfRef(k, first), unique, 'warn', { group, members: unique.length }, undefined, anchor);
        });
    },
});
