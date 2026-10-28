/**
 * knowledge/resolve.ts — deterministic entity resolution (PRD 38 §4.3).
 *
 *   • Exact: names whose normalized keys ({@link normalizeKey}) are equal are ONE entity.
 *   • Near: pairs the knowledge index calls similar become `possible-same` findings for a
 *     person to judge. They are never merged automatically.
 */

import type { Finding, NodeRef } from './model.js';
import type { SimilarPair } from '../retrieval/knowledge-index/types.js';
import { hashText } from '../ingest/hash.js';
import { compareStr, normalizeKey, round6 } from './refs.js';

export { normalizeKey };

export interface ResolvedEntity {
    key: string;
    /** The canonical ref (the smallest of the group). */
    ref: NodeRef;
    /** Every ref with this key, sorted. */
    refs: NodeRef[];
}

/** Groups refs by exact normalized key. Empty keys are ignored. Sorted by key. */
export function resolveEntities(entries: ReadonlyArray<{ ref: NodeRef; key: string }>): ResolvedEntity[] {
    const groups = new Map<string, Set<NodeRef>>();
    for (const { ref, key } of entries) {
        const k = normalizeKey(key);
        if (!k) continue;
        let g = groups.get(k);
        if (!g) groups.set(k, (g = new Set()));
        g.add(ref);
    }
    return [...groups.keys()].sort(compareStr).map((key) => {
        const refs = [...groups.get(key)!].sort(compareStr);
        return { key, ref: refs[0]!, refs };
    });
}

/** Normalized key → canonical ref, for an extractor's `ctx.knownKeys`. */
export function knownKeysOf(resolved: readonly ResolvedEntity[]): Map<string, NodeRef> {
    return new Map(resolved.map((r) => [r.key, r.ref] as const));
}

/**
 * `possible-same` findings from near-duplicate pairs between two DIFFERENT entities (pairs
 * already merged by key, or naming a non-entity, are not reported).
 */
export function possibleSameFindings(pairs: readonly SimilarPair[], pack = 'memory'): Finding[] {
    const seen = new Set<string>();
    const out: Finding[] = [];
    for (const p of pairs) {
        if (!p.a.startsWith('ent:') || !p.b.startsWith('ent:') || p.a === p.b) continue;
        const [a, b] = compareStr(p.a, p.b) <= 0 ? [p.a, p.b] : [p.b, p.a];
        if (normalizeKey(entityName(a)) === normalizeKey(entityName(b))) continue;
        const id = `possible-same:${hashText(`${a}\u0000${b}`)}`;
        if (seen.has(id)) continue;
        seen.add(id);
        out.push({
            id, kind: 'possible-same', pack, nodes: [a as NodeRef, b as NodeRef], severity: 'info',
            params: { similarity: round6(p.similarity) },
            requestKey: 'knowledgeGraph.fix.possibleSame',
        });
    }
    return out.sort((x, y) => compareStr(x.id, y.id));
}

/** The name part of `ent:<pack>:<name>`. */
export function entityName(ref: string): string {
    const rest = ref.startsWith('ent:') ? ref.slice(4) : ref;
    const colon = rest.indexOf(':');
    return colon < 0 ? rest : rest.slice(colon + 1);
}

/** The stored key of an entity named `name` in `pack`. */
export function entityKey(pack: string, name: string): NodeRef {
    return `ent:${pack}:${normalizeKey(name)}`;
}
