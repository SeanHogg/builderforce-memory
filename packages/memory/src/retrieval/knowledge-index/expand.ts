/**
 * expand.ts — recall expansion over the knowledge graph's edges.
 *
 * A hit alone is an isolated fact; its neighbours make it knowledge (a claim's
 * source section, the claims contradicting it, the entity it is about). Each hop
 * scores `parent score × edge weight × 0.5`; a node reached more than one way keeps
 * its best score; the original hits keep theirs. The result is re-ordered by
 * {@link compareHits}, so it is deterministic whatever order the graph lists
 * neighbours in.
 */

import { compareHits } from '../order.js';
import type { GraphNeighbours, IndexHit } from './types.js';

/** The per-hop decay. */
export const EXPAND_HOP_DECAY = 0.5;

export interface ExpandOptions {
    hops?: 1 | 2;
    edgeKinds?: string[];
}

export function expandHits(hits: readonly IndexHit[], graph: GraphNeighbours, opts: ExpandOptions = {}): IndexHit[] {
    const hops = opts.hops ?? 1;
    const best = new Map<string, IndexHit>();
    for (const hit of [...hits].sort(compareHits)) {
        const prev = best.get(hit.id);
        if (!prev || hit.score > prev.score) best.set(hit.id, { ...hit });
    }
    const original = new Set(best.keys());

    let frontier = [...best.values()];
    for (let hop = 0; hop < hops; hop++) {
        const reached = new Map<string, IndexHit>();
        // Parents in total order, and a strict `>` to replace: equal-score ties keep
        // the first parent in that order, independent of the graph's listing order.
        for (const parent of frontier.sort(compareHits)) {
            for (const edge of graph.neighbours(parent.id, opts.edgeKinds)) {
                if (original.has(edge.id)) continue;
                const score = parent.score * edge.weight * EXPAND_HOP_DECAY;
                const prev = reached.get(edge.id);
                if (!prev || score > prev.score) reached.set(edge.id, { id: edge.id, score, tier: parent.tier });
            }
        }
        frontier = [];
        for (const [id, hit] of reached) {
            const prev = best.get(id);
            if (!prev || hit.score > prev.score) {
                best.set(id, hit);
                frontier.push(hit);
            }
        }
    }
    return [...best.values()].sort(compareHits);
}
