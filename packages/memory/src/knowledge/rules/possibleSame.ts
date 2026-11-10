/**
 * knowledge/rules/possibleSame.ts — `possible-same` as a rule: two entities the index found near-duplicate
 * whose names do not normalize equal. The finding producer stays in resolve.ts beside entity resolution.
 */

import { possibleSameFindings } from '../resolve.js';
import { defineRule } from './factRule.js';
import type { KnowledgeRuleExtra } from './knowledgeInput.js';

export const possibleSameRule = defineRule<KnowledgeRuleExtra>({
    id: 'possible-same',
    version: 1,
    reads: [], // the near-duplicate pairs of the knowledge input only
    scope: 'global',
    extraInputs: ['knowledge'],
    derive: ({ extra: { knowledge: k } }) => possibleSameFindings(k.nearDuplicates),
});
