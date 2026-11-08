/**
 * knowledge/findingBuilder.ts — the ONE finding builder and its id scheme, split out of
 * findings.ts so a rule module (e.g. godModule.ts) and findings.ts can both import it
 * without importing each other.
 */

import { hashText } from '../ingest/hash.js';
import type { Finding, NodeRef } from './model.js';
import { compareStr } from './refs.js';

/** Refs listed on one grouped finding, at most (the count rides in params). */
export const MAX_FINDING_NODES = 50;

export function findingId(kind: string, refs: readonly string[]): string {
    return `${kind}:${hashText([...refs].sort(compareStr).join('\u0000'))}`;
}

/** `clone-group` → `knowledgeGraph.fix.cloneGroup`. */
export function requestKeyFor(kind: string): string {
    return `knowledgeGraph.fix.${kind.replace(/-([a-z])/g, (_, c: string) => c.toUpperCase())}`;
}

/**
 * The ONE finding builder: nodes deduped, sorted and capped at {@link MAX_FINDING_NODES}.
 * The id hashes `idRefs` (default: all the nodes) — pass a narrower set to keep it stable
 * while the listed nodes change.
 */
export function finding(
    kind: string, pack: string, nodes: readonly NodeRef[], severity: Finding['severity'], params: Finding['params'],
    idRefs?: readonly string[],
): Finding {
    const sorted = [...new Set(nodes)].sort(compareStr);
    return {
        id: findingId(kind, idRefs ?? sorted), kind, pack, nodes: sorted.slice(0, MAX_FINDING_NODES), severity, params,
        requestKey: requestKeyFor(kind),
    };
}
