/**
 * knowledge/rules/ioInLoop.ts — `io-in-loop` as a rule: the finding stays in knowledge/ioInLoop.ts;
 * this registers it. Reads the `patterns` badge of the file nodes of the knowledge input.
 */

import { ioInLoopFindings, IO_IN_LOOP_KIND } from '../ioInLoop.js';
import { defineRule } from './factRule.js';
import type { KnowledgeRuleExtra } from './knowledgeInput.js';

export const ioInLoopRule = defineRule<KnowledgeRuleExtra>({
    id: IO_IN_LOOP_KIND,
    version: 1,
    reads: [], // the knowledge graph only (the pattern is a file-node badge)
    scope: 'global',
    extraInputs: ['knowledge'],
    derive: ({ extra: { knowledge: k } }) => ioInLoopFindings(k.graph),
});
