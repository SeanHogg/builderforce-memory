/**
 * knowledge/truncate.ts — the graph opens bounded (PRD 38 §5.3).
 *
 * The top `maxNodes` by importance (then ref), PLUS focus nodes, every node named by a
 * finding, changed nodes and the 1-hop neighbours of all of those — up to the hard cap.
 * Everything else folds into `area` nodes; edges are re-pointed to the areas, merged, and
 * self-loops dropped.
 */

import type { EdgeVerdict, KEdge, KNode, KnowledgeGraph, NodeRef } from './model.js';
import { areaImportance, byImportance } from './importance.js';
import { compareStr, pathOfRef, srcRef } from './refs.js';

/** Hard cap on nodes, however many the keep rules ask for. */
export const TRUNCATION_HARD_CAP = 2000;
/** The area of folded nodes that are not under any source path (unanchored facts). */
export const UNANCHORED_AREA = '(memory)';

const VERDICT_RANK: Record<EdgeVerdict, number> = { same: 0, allowed: 1, skips: 2, upward: 3 };

export interface TruncateOptions {
    maxNodes: number;
    focus: readonly NodeRef[];
    changed: readonly NodeRef[];
}

export function truncateGraph(g: KnowledgeGraph, opts: TruncateOptions): { nodes: KNode[]; edges: KEdge[]; truncated: boolean } {
    if (g.nodes.length <= opts.maxNodes) return { nodes: g.nodes, edges: g.edges, truncated: false };
    const byRef = new Map(g.nodes.map((n) => [n.ref, n] as const));
    const neighbours = new Map<NodeRef, Set<NodeRef>>();
    const link = (a: NodeRef, b: NodeRef) => {
        let s = neighbours.get(a);
        if (!s) neighbours.set(a, (s = new Set()));
        s.add(b);
    };
    for (const e of g.edges) { link(e.from, e.to); link(e.to, e.from); }

    const kept: NodeRef[] = [];
    const keptSet = new Set<NodeRef>();
    const keep = (ref: NodeRef) => {
        if (keptSet.has(ref) || !byRef.has(ref) || kept.length >= TRUNCATION_HARD_CAP) return;
        keptSet.add(ref);
        kept.push(ref);
    };
    const findingRefs = [...g.findings].sort((a, b) => compareStr(a.id, b.id)).flatMap((f) => f.nodes);
    const seeds = [...[...opts.focus].sort(compareStr), ...findingRefs, ...[...opts.changed].sort(compareStr)];
    for (const s of seeds) keep(s);
    const seeded = [...kept];
    for (const s of seeded) {
        const around = [...(neighbours.get(s) ?? [])].map((r) => byRef.get(r)!).filter(Boolean).sort(byImportance);
        for (const n of around) keep(n.ref);
    }
    for (const n of [...g.nodes].sort(byImportance).slice(0, opts.maxNodes)) keep(n.ref);

    // Fold the rest into areas.
    const aboutTarget = new Map<NodeRef, NodeRef>();
    for (const e of g.edges) if (e.kind === 'about' && !aboutTarget.has(e.from)) aboutTarget.set(e.from, e.to);
    const rootId = g.scope.rootId;
    const areaOf = (n: KNode): NodeRef => {
        if (n.area) return n.area;
        const target = aboutTarget.get(n.ref);
        const area = target && byRef.get(target)?.area;
        return area ?? (pathOfRef(n.ref) !== undefined ? n.ref : srcRef(rootId, UNANCHORED_AREA));
    };
    const fold = new Map<NodeRef, NodeRef>();
    const members = new Map<NodeRef, KNode[]>();
    for (const n of g.nodes) {
        if (keptSet.has(n.ref)) continue;
        const area = areaOf(n);
        fold.set(n.ref, area);
        let m = members.get(area);
        if (!m) members.set(area, (m = []));
        m.push(n);
    }
    const imp = areaImportance(new Map([...members].map(([a, ms]) => [a, ms.map((n) => n.importance)] as const)));
    const areaNodes: KNode[] = [...members.keys()].filter((a) => !keptSet.has(a)).map((area) => {
        const ms = members.get(area)!;
        const partition = mostCommon(ms.map((n) => n.partition));
        return {
            ref: area, kind: 'area', label: pathOfRef(area) ?? area, pack: mostCommon(ms.map((n) => n.pack)) ?? 'memory',
            ...(partition !== undefined ? { partition } : {}),
            importance: imp.get(area) ?? 0, origin: 'extracted' as const,
            badges: { members: ms.length },
        };
    });

    const edges = new Map<string, KEdge>();
    for (const e of g.edges) {
        const from = fold.get(e.from) ?? e.from, to = fold.get(e.to) ?? e.to;
        if (from === to) continue;
        const id = `${from}\u0000${to}\u0000${e.kind}`;
        const prior = edges.get(id);
        if (!prior) { edges.set(id, { ...e, from, to }); continue; }
        prior.weight += e.weight;
        if (e.cycle) prior.cycle = true;
        if (e.stale) prior.stale = true;
        if (e.confidence !== undefined) prior.confidence = Math.max(prior.confidence ?? 0, e.confidence);
        if (e.verdict && (!prior.verdict || VERDICT_RANK[e.verdict] > VERDICT_RANK[prior.verdict])) prior.verdict = e.verdict;
    }
    const nodes = [...g.nodes.filter((n) => keptSet.has(n.ref)), ...areaNodes].sort((a, b) => compareStr(a.ref, b.ref));
    const outEdges = [...edges.values()]
        .map((e) => ({ ...e, weight: Math.round(e.weight * 1e6) / 1e6 }))
        .sort((a, b) => compareStr(a.from, b.from) || compareStr(a.to, b.to) || compareStr(a.kind, b.kind));
    return { nodes, edges: outEdges, truncated: true };
}

/** The most frequent defined value; ties go to the smallest. */
function mostCommon(values: ReadonlyArray<string | undefined>): string | undefined {
    const counts = new Map<string, number>();
    for (const v of values) if (v !== undefined) counts.set(v, (counts.get(v) ?? 0) + 1);
    let best: string | undefined;
    for (const [v, c] of [...counts].sort((a, b) => compareStr(a[0], b[0]))) {
        if (best === undefined || c > counts.get(best)!) best = v;
    }
    return best;
}
