/**
 * knowledge/communities.ts — deterministic community detection over the FILE reference graph
 * (PRD 38 P1: a placement for every file the packs could not place, plus a membership confidence).
 *
 * Louvain modularity maximisation: Q = (1/2m) Σᵢⱼ [Aᵢⱼ − kᵢkⱼ/2m] δ(cᵢ,cⱼ), over the undirected graph
 * whose edge weight is the reference count (both directions summed). Above COMMUNITY_LOUVAIN_MAX_NODES
 * connected files, deterministic label propagation is the fallback. The algorithms live in louvain.ts.
 *
 * Determinism: see louvain.ts; edge weights are merged and summed in a fixed order. Same input (in any
 * order) gives byte-identical output.
 *
 * A file with no references is in no community (absent, never a guess; no folder fallback).
 *
 * Community ids are indexes (0.. in order of each community's smallest member path). They are NOT stable
 * across builds: a changed graph can renumber them. A consumer keeps the id only together with the graph
 * (the outline's `communities` table) that it indexes, never as a long-lived key.
 */

import type { KEdge, KNode, NodeRef, OutlineCommunity } from './model.js';
import {
    buildAdjacency, COMMUNITY_LP_ITERATIONS, COMMUNITY_MAX_ITERATIONS, COMMUNITY_MAX_PASSES, degreeOf, labelPropagation,
    louvain, modularityOf,
} from './louvain.js';
import { basenameOf, compareStr, fileRefOf, pathOfRef, round6 } from './refs.js';

export { COMMUNITY_LP_ITERATIONS, COMMUNITY_MAX_ITERATIONS, COMMUNITY_MAX_PASSES };

/** Above this many connected files label propagation replaces Louvain. Separate from the outline cap. */
export const COMMUNITY_LOUVAIN_MAX_NODES = 20000;
/** The file-to-file edge kinds that count as a reference between two files (code `references`, docs `links`). */
export const COMMUNITY_EDGE_KINDS: readonly string[] = ['references', 'links'];

export type CommunityMethod = 'louvain' | 'label-propagation';

export interface CommunityMembership {
    /** Index of the community: 0.. by smallest member path. Not stable across builds. */
    communityId: number;
    /** Share of this file's edge weight that falls inside its community, 0..1. */
    confidence: number;
}

export interface CommunityResult {
    /** Only files with at least one reference appear. */
    members: Map<NodeRef, CommunityMembership>;
    /** Q of the returned partition, rounded to 6 decimals (0 with no edges). */
    modularity: number;
    method: CommunityMethod;
}

/** Where a node sits on the map: the pack's partition when it has one, else its community. */
export type NodePlacement = { kind: 'partition'; id: string } | { kind: 'community'; id: number };

/**
 * Communities over `edges` (any direction; both summed). `maxNodes` is the number of connected files above
 * which label propagation replaces Louvain.
 */
export function detectCommunities(
    refs: readonly NodeRef[],
    edges: ReadonlyArray<{ from: NodeRef; to: NodeRef; weight: number }>,
    maxNodes: number = COMMUNITY_LOUVAIN_MAX_NODES,
): CommunityResult {
    const order = [...new Set(refs)].sort(compareStr);
    const adj = buildAdjacency(order, edges);
    const connected = order.map((_, i) => i).filter((i) => adj[i]!.size > 0);
    const method: CommunityMethod = connected.length > maxNodes ? 'label-propagation' : 'louvain';
    const comm = connected.length === 0 ? order.map((_, i) => i) : method === 'louvain' ? louvain(adj) : labelPropagation(adj);

    // Ids by first appearance over the sorted connected files: community 0 holds the smallest path.
    const idOf = new Map<number, number>();
    const members = new Map<NodeRef, CommunityMembership>();
    for (const i of connected) {
        if (!idOf.has(comm[i]!)) idOf.set(comm[i]!, idOf.size);
        let inside = 0;
        for (const [j, w] of adj[i]!) if (comm[j] === comm[i]) inside += w;
        members.set(order[i]!, { communityId: idOf.get(comm[i]!)!, confidence: round6(inside / degreeOf(adj[i]!)) });
    }
    return { members, modularity: modularityOf(adj, comm), method };
}

/** THE test of "this edge is a reference between files" (a live edge of COMMUNITY_EDGE_KINDS); coChange.ts reads it too. */
export function isReferenceEdge(e: Pick<KEdge, 'kind' | 'stale'>): boolean {
    return COMMUNITY_EDGE_KINDS.includes(e.kind) && !e.stale;
}

/**
 * File-to-file links of the chosen kinds, merged per ordered pair. The weights of a pair are summed in ascending
 * order, so the total never depends on the order the edges arrive in (the sort across pairs is `buildAdjacency`'s).
 */
function fileLinks(files: ReadonlySet<NodeRef>, edges: Iterable<KEdge>): Array<{ from: NodeRef; to: NodeRef; weight: number }> {
    const pairs = new Map<string, { from: NodeRef; to: NodeRef; weights: number[] }>();
    for (const e of edges) {
        if (!isReferenceEdge(e)) continue;
        const from = fileRefOf(e.from), to = fileRefOf(e.to);
        if (!files.has(from) || !files.has(to)) continue;
        const key = `${from}\u0000${to}`;
        const hit = pairs.get(key);
        if (hit) hit.weights.push(e.weight);
        else pairs.set(key, { from, to, weights: [e.weight] });
    }
    return [...pairs.values()].map((p) => ({
        from: p.from, to: p.to, weight: p.weights.sort((a, b) => a - b).reduce((s, w) => s + w, 0),
    }));
}

/**
 * The one call `buildGraph` makes: stamps `communityId` / `communityConfidence` badges on file nodes that
 * have references (file-to-file edges of COMMUNITY_EDGE_KINDS, stale relations skipped) and returns Q and
 * the method for the graph stats.
 */
export function applyCommunities(nodes: Iterable<KNode>, edges: Iterable<KEdge>): { modularity: number; method: CommunityMethod } {
    const files = new Map<NodeRef, KNode>();
    for (const n of nodes) if (n.kind === 'file') files.set(n.ref, n);
    const { members, modularity, method } = detectCommunities([...files.keys()], fileLinks(new Set(files.keys()), edges));
    for (const [ref, m] of members) {
        const n = files.get(ref)!;
        n.badges = { ...n.badges, communityId: m.communityId, communityConfidence: m.confidence };
    }
    return { modularity, method };
}

/** The ONE placement of a node: its partition, else its community, else none (never a guess). */
export function placementOf(node: Pick<KNode, 'partition' | 'badges'>): NodePlacement | undefined {
    if (node.partition !== undefined) return { kind: 'partition', id: node.partition };
    const id = node.badges?.['communityId'];
    return typeof id === 'number' ? { kind: 'community', id } : undefined;
}

/** Longest directory shared by every path ('' when none). */
function commonDirectory(paths: readonly string[]): string {
    const dirs = paths.map((p) => p.split('/').slice(0, -1));
    const first = dirs[0] ?? [];
    let n = first.length;
    for (const d of dirs) {
        let i = 0;
        while (i < n && d[i] === first[i]) i++;
        n = i;
    }
    return first.slice(0, n).join('/');
}

/**
 * The outline's `communities` table for the given file nodes: id, a label and the member count, sorted by id.
 * The label is the members' longest common directory (else the path of the most important member), built from
 * the outline-relative path (`prefix` stripped, like `OutlineFile.path`). Labels are unique: two communities with
 * the same label get ` · <top member basename>` appended, then a numeric suffix if they still collide.
 */
export function communityTable(files: readonly KNode[], prefix = ''): OutlineCommunity[] {
    const groups = new Map<number, KNode[]>();
    for (const n of files) {
        const id = n.badges?.['communityId'];
        if (typeof id !== 'number') continue;
        const list = groups.get(id) ?? [];
        list.push(n);
        groups.set(id, list);
    }
    const relative = (n: KNode): string => (prefix !== '' && n.ref.startsWith(prefix) ? n.ref.slice(prefix.length) : (pathOfRef(n.ref) ?? n.ref));
    const rows = [...groups.keys()].sort((a, b) => a - b).map((id) => {
        const members = groups.get(id)!;
        const top = [...members].sort((a, b) => b.importance - a.importance || compareStr(a.ref, b.ref))[0]!;
        const topPath = relative(top);
        return { id, size: members.length, base: commonDirectory(members.map(relative)) || topPath, topName: basenameOf(topPath) };
    });
    const count = new Map<string, number>();
    for (const r of rows) count.set(r.base, (count.get(r.base) ?? 0) + 1);
    const withTop = rows.map((r) => ({ ...r, label: count.get(r.base)! > 1 ? `${r.base} · ${r.topName}` : r.base }));
    const seen = new Map<string, number>();
    return withTop.map((r) => {
        const n = (seen.get(r.label) ?? 0) + 1;
        seen.set(r.label, n);
        return { id: r.id, label: n === 1 ? r.label : `${r.label} ${n}`, size: r.size };
    });
}
