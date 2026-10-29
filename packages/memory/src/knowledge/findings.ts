/**
 * knowledge/findings.ts — what a person can act on (PRD 38 §3.5).
 *
 * Generic findings, for every pack: cycle, clone group, near-duplicate, partition
 * violation, unpartitioned share, stale knowledge — plus `possible-same` entities. Packs
 * add their own through {@link FindingRule}s. Every finding has a stable id (a hash of its
 * kind and refs) and a `requestKey` under `knowledgeGraph.fix.*`: the fixed, prefilled
 * Fix-with-agent request, never a model call.
 */

import { hashText } from '../ingest/hash.js';
import type { EdgeVerdict, Finding, KEdge, KNode, KnowledgeGraph, KnowledgePack, NodeRef, PartitionRegistry } from './model.js';
import type { SimilarPair } from '../retrieval/knowledge-index/types.js';
import { possibleSameFindings } from './resolve.js';
import { aboutTargets, compareStr, round6 } from './refs.js';

/** Node badge naming the clone group (e.g. a jscpd issue id) its `duplicate` edges belong to. */
export const CLONE_GROUP_BADGE = 'cloneGroup';
/** Refs listed on one grouped finding, at most (the count rides in params). */
export const MAX_FINDING_NODES = 50;

const SEVERITY_RANK = { error: 0, warn: 1, info: 2 } as const;

/** A partition violation's severity: an upward edge is an error, a skipped layer a warning. */
export function partitionSeverity(verdict: EdgeVerdict): Finding['severity'] {
    return verdict === 'upward' ? 'error' : 'warn';
}

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

export function genericFindings(g: KnowledgeGraph, registries: ReadonlyMap<string, PartitionRegistry>): Finding[] {
    const byRef = new Map(g.nodes.map((n) => [n.ref, n] as const));
    const packOf = (ref: NodeRef) => byRef.get(ref)?.pack ?? 'memory';
    const out: Finding[] = [];

    for (const refs of components(g.edges.filter((e) => e.cycle))) {
        out.push(finding('cycle', packOf(refs[0]!), refs, 'warn', { nodes: refs.length }));
    }

    // Clone groups: duplicate edges grouped by the clone-group badge, else by component.
    const dup = g.edges.filter((e) => e.kind === 'duplicate');
    const byBadge = new Map<string, KEdge[]>();
    const unbadged: KEdge[] = [];
    for (const e of dup) {
        const badge = byRef.get(e.from)?.badges?.[CLONE_GROUP_BADGE] ?? byRef.get(e.to)?.badges?.[CLONE_GROUP_BADGE];
        if (badge === undefined) { unbadged.push(e); continue; }
        const list = byBadge.get(String(badge)) ?? [];
        list.push(e);
        byBadge.set(String(badge), list);
    }
    const groups: Array<{ group: string; refs: NodeRef[] }> = [
        ...[...byBadge].map(([group, es]) => ({ group, refs: es.flatMap((e) => [e.from, e.to]) })),
        ...components(unbadged).map((refs) => ({ group: '', refs })),
    ];
    for (const { group, refs } of groups) {
        const unique = [...new Set(refs)];
        out.push(finding('clone-group', packOf(unique.sort(compareStr)[0]!), unique, 'warn', { group, members: unique.length }));
    }

    for (const e of g.edges) {
        if (e.kind === 'near-duplicate') {
            out.push(finding('near-duplicate', packOf(e.from), [e.from, e.to], 'info', { similarity: round6(e.weight) }));
        }
        if (e.verdict === 'upward' || e.verdict === 'skips') {
            const from = byRef.get(e.from), to = byRef.get(e.to);
            out.push(finding('partition-violation', packOf(e.from), [e.from, e.to], partitionSeverity(e.verdict), {
                verdict: e.verdict, from: from?.partition ?? '', to: to?.partition ?? '', kind: e.kind,
            }));
        }
    }

    for (const [pack, reg] of [...registries].sort((a, b) => compareStr(a[0], b[0]))) {
        if (!reg.allow) continue;
        const files = g.nodes.filter((n) => n.pack === pack && n.kind === 'file');
        const loose = files.filter((n) => n.partition === undefined).map((n) => n.ref);
        if (loose.length === 0) continue;
        out.push(finding('unpartitioned', pack, loose, 'info', {
            count: loose.length, total: files.length, share: Math.round((loose.length / files.length) * 100),
        }));
    }

    out.push(...staleFindings(g, byRef));
    return out;
}

/** One finding per source whose stored knowledge went stale (unanchored stale nodes: one each). */
function staleFindings(g: KnowledgeGraph, byRef: ReadonlyMap<NodeRef, KNode>): Finding[] {
    const about = aboutTargets(g.edges);
    const bySource = new Map<NodeRef, NodeRef[]>();
    for (const n of g.nodes) {
        if (!n.stale) continue;
        const target = about.get(n.ref) ?? n.ref;
        const list = bySource.get(target) ?? [];
        list.push(n.ref);
        bySource.set(target, list);
    }
    return [...bySource].map(([source, refs]) =>
        finding('stale', byRef.get(source)?.pack ?? 'memory', [source, ...refs], 'warn', { count: refs.length }));
}

/** Generic + possible-same + every pack's rules; deduped by id; severity, then id. */
export function computeFindings(
    g: KnowledgeGraph,
    packs: readonly KnowledgePack[],
    registries: ReadonlyMap<string, PartitionRegistry>,
    nearDuplicates: readonly SimilarPair[],
): Finding[] {
    const all = [...genericFindings(g, registries), ...possibleSameFindings(nearDuplicates)];
    for (const pack of [...packs].sort((a, b) => compareStr(a.id, b.id))) {
        for (const rule of pack.findings ?? []) {
            for (const f of rule.find(g)) {
                all.push({
                    ...f,
                    id: f.id || findingId(f.kind, f.nodes),
                    nodes: [...new Set(f.nodes)].sort(compareStr),
                    requestKey: f.requestKey ?? requestKeyFor(f.kind),
                });
            }
        }
    }
    const byId = new Map<string, Finding>();
    for (const f of all) if (!byId.has(f.id)) byId.set(f.id, f);
    return [...byId.values()].sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity] || compareStr(a.id, b.id));
}
