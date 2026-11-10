/**
 * knowledge/rules/components.ts — connected components over edges (union-find), shared by the
 * `cycle` and `clone-group` rules. Moved from findings.ts, which re-exports it.
 */

import type { KEdge, NodeRef } from '../model.js';
import { compareStr } from '../refs.js';

/** Connected components over edges, each sorted; components sorted by first ref. */
export function components(edges: readonly KEdge[]): NodeRef[][] {
    const parent = new Map<NodeRef, NodeRef>();
    const find = (x: NodeRef): NodeRef => {
        let r = x;
        while (parent.get(r) !== r) r = parent.get(r)!;
        parent.set(x, r);
        return r;
    };
    for (const e of edges) {
        for (const r of [e.from, e.to]) if (!parent.has(r)) parent.set(r, r);
        const a = find(e.from), b = find(e.to);
        if (a !== b) parent.set(compareStr(a, b) < 0 ? b : a, compareStr(a, b) < 0 ? a : b);
    }
    const groups = new Map<NodeRef, NodeRef[]>();
    for (const r of [...parent.keys()].sort(compareStr)) {
        const root = find(r);
        let g = groups.get(root);
        if (!g) groups.set(root, (g = []));
        g.push(r);
    }
    return [...groups.values()].sort((a, b) => compareStr(a[0]!, b[0]!));
}
