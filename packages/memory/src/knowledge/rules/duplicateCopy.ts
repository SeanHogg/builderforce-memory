/**
 * knowledge/rules/duplicateCopy.ts — `duplicate-copy`: a copy of another system in the same root (PRD 41 A9).
 * Its refs name the two systems' folders. Anchor (PRD 41 §C3): the copy's folder.
 */

import { finding } from '../findingBuilder.js';
import { FALLBACK_PACK_ID, srcRef } from '../refs.js';
import { defineRule } from './factRule.js';
import type { KnowledgeRuleExtra } from './knowledgeInput.js';

export const duplicateCopyRule = defineRule<KnowledgeRuleExtra>({
    id: 'duplicate-copy',
    version: 1,
    reads: [], // the knowledge graph only
    scope: 'global',
    extraInputs: ['knowledge'],
    derive: ({ extra: { knowledge: { graph: g } } }) =>
        (g.meta?.systems ?? []).filter((s) => s.copyOf !== undefined).map((s) =>
            finding('duplicate-copy', FALLBACK_PACK_ID, [srcRef(g.scope.rootId, s.path), srcRef(g.scope.rootId, s.copyOf!)], 'warn', {
                copy: s.path, original: s.copyOf!, similarity: s.copySimilarity ?? 1, reason: s.copyReason ?? 'later-name',
            }, [s.path, s.copyOf!], { anchor: srcRef(g.scope.rootId, s.path), anchorKind: 'file' })),
});
