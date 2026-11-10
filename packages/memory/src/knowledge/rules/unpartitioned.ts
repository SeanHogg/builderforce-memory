/**
 * knowledge/rules/unpartitioned.ts — `unpartitioned`: per pack with an `allow` registry, the share of its
 * files no partition matched. Anchor (PRD 41 §C3): the pack.
 */

import { finding } from '../findingBuilder.js';
import { compareStr } from '../refs.js';
import { defineRule } from './factRule.js';
import type { KnowledgeRuleExtra } from './knowledgeInput.js';

export const unpartitionedRule = defineRule<KnowledgeRuleExtra>({
    id: 'unpartitioned',
    version: 1,
    reads: [], // the knowledge graph and partition registries only
    scope: 'global',
    extraInputs: ['knowledge'],
    derive: ({ extra: { knowledge: k } }) =>
        [...k.registries].sort((a, b) => compareStr(a[0], b[0])).flatMap(([pack, reg]) => {
            if (!reg.allow) return [];
            const files = k.graph.nodes.filter((n) => n.pack === pack && n.kind === 'file');
            const loose = files.filter((n) => n.partition === undefined).map((n) => n.ref);
            if (loose.length === 0) return [];
            return [finding('unpartitioned', pack, loose, 'info', {
                count: loose.length, total: files.length, share: Math.round((loose.length / files.length) * 100),
            }, undefined, { anchor: `pack:${pack}`, anchorKind: 'group' })];
        }),
});
