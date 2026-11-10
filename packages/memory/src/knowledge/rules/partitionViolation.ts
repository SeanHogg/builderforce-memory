/**
 * knowledge/rules/partitionViolation.ts — `partition-violation`: an edge whose layer verdict is `upward`
 * (error) or `skips` (warn). Anchor (PRD 41 §C3): the from→to layer pair at the from FILE.
 */

import { finding } from '../findingBuilder.js';
import type { EdgeVerdict, Finding } from '../model.js';
import { fileRefOf } from '../refs.js';
import { defineRule } from './factRule.js';
import { packOfRef, type KnowledgeRuleExtra } from './knowledgeInput.js';

/** A partition violation's severity: an upward edge is an error, a skipped layer a warning. */
export function partitionSeverity(verdict: EdgeVerdict): Finding['severity'] {
    return verdict === 'upward' ? 'error' : 'warn';
}

export const partitionViolationRule = defineRule<KnowledgeRuleExtra>({
    id: 'partition-violation',
    version: 1,
    reads: [], // the knowledge graph only
    scope: 'global',
    extraInputs: ['knowledge'],
    derive: ({ extra: { knowledge: k } }) =>
        k.graph.edges.filter((e) => e.verdict === 'upward' || e.verdict === 'skips').map((e) => {
            const from = k.byRef.get(e.from), to = k.byRef.get(e.to);
            const layers = `${from?.partition ?? ''}->${to?.partition ?? ''}`;
            return finding('partition-violation', packOfRef(k, e.from), [e.from, e.to], partitionSeverity(e.verdict!), {
                verdict: e.verdict!, from: from?.partition ?? '', to: to?.partition ?? '', kind: e.kind,
            }, undefined, { anchor: `${layers}@${fileRefOf(e.from)}`, anchorKind: 'file' });
        }),
});
