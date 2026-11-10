/**
 * knowledge/findingModel.ts — the finding types: what a rule reports over a graph (moved out of model.ts,
 * which re-exports them, so every import of `./model.js` keeps working). Identity v2 lives in findingIdentity.ts.
 */

import type { KnowledgeGraph, NodeRef } from './model.js';

/** A finding a person can act on. `kind` is pack-defined; `requestKey` names an i18n template. */
export interface Finding {
    id: string;
    kind: string;
    pack: string;
    nodes: NodeRef[];
    severity: 'info' | 'warn' | 'error';
    /** Numbers/strings for the label and the Fix-with-agent template. */
    params: Record<string, string | number>;
    /** i18n key of the fixed Fix-with-agent request (e.g. 'knowledgeGraph.fix.cloneGroup'). */
    requestKey?: string;
    /**
     * The PRIMARY anchor the emitting rule names (PRD 41 §C3): an enclosing symbol/file ref, route key,
     * package name or group name, plus `~<normalized-snippet hash>` where a snippet exists. Never a line
     * number, never the member set. Absent ⇒ `identityOf` uses its documented default.
     */
    anchor?: string;
    /** What {@link Finding.anchor} names. */
    anchorKind?: 'symbol' | 'route' | 'package' | 'file' | 'group';
}

export interface FindingRule {
    id: string;
    find(graph: KnowledgeGraph): Finding[];
}
