/**
 * knowledge/rules/godModule.ts — `god-module` as a rule: the detector and its limits stay in
 * knowledge/godModule.ts (the code pack imports the limits from there); this registers it.
 * Reads the lines, exports and worst-function complexity badges of the file nodes of the knowledge input.
 */

import { godModuleFindings, GOD_MODULE_KIND } from '../godModule.js';
import { defineRule } from './factRule.js';
import type { KnowledgeRuleExtra } from './knowledgeInput.js';

export const godModuleRule = defineRule<KnowledgeRuleExtra>({
    id: GOD_MODULE_KIND,
    version: 1,
    reads: [], // the knowledge graph only (the measures are file-node badges)
    scope: 'global',
    extraInputs: ['knowledge'],
    derive: ({ extra: { knowledge: k } }) => godModuleFindings(k.graph),
});
