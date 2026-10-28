/**
 * knowledge/build.ts — `buildGraph`: join perception (extracted, computed) with memory
 * (stored facts and relations) into one bounded {@link KnowledgeGraph} (PRD 38 §4.1).
 *
 *   1. extracted nodes/edges (analyzer output merges its badges into the same refs);
 *   2. facts become nodes, joined to sources by `fileAnchors.path`; a model fact whose
 *      anchor hash differs from the source's current hash is `stale`;
 *   3. stored relations become edges; near-duplicate pairs become `near-duplicate` edges
 *      (an exact `duplicate` between the same pair wins); live changes badge nodes and
 *      `runFiles` become `co-changed` edges;
 *   4. partitions, areas, verdicts; PageRank importance; findings; then truncation.
 *
 * Deterministic: equal input ⇒ byte-identical JSON. Every list is sorted, every sort ends
 * on ref, objects are built with a fixed key order.
 */

import { areaOfPath } from '../working/codemapAreas.js';
import { sha256Hex } from '../retrieval/knowledge-index/sha256.js';
import type {
    BuildGraphInput, BuildGraphOptions, Extracted, KEdge, KNode, KnowledgeGraph, NodeRef,
    PartitionRegistry, StoredFact,
} from './model.js';
import { computeFindings } from './findings.js';
import { pageRank } from './importance.js';
import { loadPartitions, partitionOf, verdictOf } from './partitions.js';
import { entityName } from './resolve.js';
import {
    basenameOf, canonicalJson, compareStr, factRef, pathOfRef, round6, sortBy, srcRef,
} from './refs.js';
import { truncateGraph } from './truncate.js';

export const DEFAULT_MAX_NODES = 400;
export const MAX_NODES_CAP = 2000;
const LABEL_CHARS = 120;

const edgeId = (from: string, to: string, kind: string) => `${from}\u0000${to}\u0000${kind}`;
export const compareEdges = (a: KEdge, b: KEdge): number =>
    compareStr(a.from, b.from) || compareStr(a.to, b.to) || compareStr(a.kind, b.kind);

/** sha-256 over the canonical input: equal inputs (in any order) ⇒ equal digests. */
export function graphDigest(input: BuildGraphInput): string {
    return sha256Hex(canonicalJson({
        rootId: input.rootId,
        packs: input.packs.map((p) => p.id).sort(compareStr),
        extracted: sortBy(input.extracted, (e) => e.sourceRef, (e) => e.sourceHash).map(canonicalExtracted),
        facts: sortBy(input.facts, (f) => f.key).map((f) => ({ ...f, fileAnchors: sortBy(f.fileAnchors, (a) => a.path) })),
        relations: sortBy(input.relations, (r) => r.from, (r) => r.to, (r) => r.kind),
        live: input.live && {
            changes: sortBy(input.live.changes, (c) => c.path),
            runFiles: nonEmpty(input.live.runFiles) && [...input.live.runFiles!].sort(compareStr),
        },
        nearDuplicates: nonEmpty(input.nearDuplicates) && sortBy(input.nearDuplicates!, (p) => p.a, (p) => p.b),
    }));
}

/** An absent list and an empty list mean the same input, so they hash the same. */
const nonEmpty = (xs: readonly unknown[] | undefined): boolean | undefined => (xs && xs.length > 0) || undefined;

function canonicalExtracted(e: Extracted) {
    return {
        sourceRef: e.sourceRef, sourceHash: e.sourceHash,
        nodes: sortBy(e.nodes, (n) => n.ref),
        edges: [...e.edges].sort(compareEdges),
        keys: sortBy(e.keys, (k) => k.ref, (k) => k.key),
    };
}

function mergeBadges(a?: Record<string, number | string>, b?: Record<string, number | string>) {
    if (!a) return b && { ...b };
    if (!b) return { ...a };
    const out = { ...a };
    // Commutative: on a conflict the larger value (as a string) wins, whatever the order.
    for (const [k, v] of Object.entries(b)) out[k] = k in out && String(out[k]) >= String(v) ? out[k]! : v;
    return out;
}

function oneLine(text: string): string {
    const flat = text.replace(/\s+/g, ' ').trim();
    return flat.length > LABEL_CHARS ? `${flat.slice(0, LABEL_CHARS - 1)}…` : flat;
}

class GraphBuilder {
    readonly nodes = new Map<NodeRef, KNode>();
    readonly edges = new Map<string, KEdge>();

    node(n: KNode): void {
        const prior = this.nodes.get(n.ref);
        if (!prior) { this.nodes.set(n.ref, { ...n, ...(n.badges ? { badges: { ...n.badges } } : {}) }); return; }
        const badges = mergeBadges(prior.badges, n.badges);
        if (badges) prior.badges = badges;
    }

    edge(e: KEdge): void {
        const id = edgeId(e.from, e.to, e.kind);
        const prior = this.edges.get(id);
        if (!prior) { this.edges.set(id, { ...e }); return; }
        prior.weight = Math.max(prior.weight, e.weight);
        if (e.cycle) prior.cycle = true;
        if (e.confidence !== undefined) prior.confidence = Math.max(prior.confidence ?? 0, e.confidence);
    }

    has(from: NodeRef, to: NodeRef, kind: string): boolean {
        return this.edges.has(edgeId(from, to, kind));
    }
}

function factNode(f: StoredFact, rootId: string, g: GraphBuilder): KNode {
    const ref = factRef(f.key);
    const kind = f.key.startsWith('ent:') ? 'entity' : f.key.startsWith('claim:') ? 'claim' : 'fact';
    const anchors = sortBy(f.fileAnchors, (a) => a.path);
    const anchoredPack = anchors.map((a) => g.nodes.get(srcRef(rootId, a.path))?.pack).find(Boolean);
    const pack = kind === 'entity' ? f.key.split(':')[1] || 'memory' : anchoredPack ?? 'memory';
    const stale = f.origin === 'model' && anchors.some((a) => {
        const current = g.nodes.get(srcRef(rootId, a.path))?.sourceHash;
        return !!a.contentHash && !!current && current !== a.contentHash;
    });
    const first = anchors[0];
    return {
        ref, kind, label: oneLine(kind === 'entity' ? f.content || entityName(ref) : f.content), pack,
        importance: 0, origin: f.origin,
        ...(first?.contentHash ? { sourceHash: first.contentHash } : {}),
        ...(stale ? { stale: true } : {}),
        ...(first?.line !== undefined ? { badges: { line: first.line } } : {}),
    };
}

export function buildGraph(input: BuildGraphInput, opts: BuildGraphOptions = {}): KnowledgeGraph {
    const { rootId } = input;
    const g = new GraphBuilder();
    const sources = new Set<string>();

    for (const ex of sortBy(input.extracted, (e) => e.sourceRef, (e) => e.sourceHash)) {
        if (ex.sourceRef.startsWith('src:')) sources.add(ex.sourceRef);
        for (const n of sortBy(ex.nodes, (x) => x.ref)) g.node({ ...n, origin: 'extracted' });
        for (const e of [...ex.edges].sort(compareEdges)) g.edge({ ...e, origin: 'extracted' });
    }

    const facts = sortBy(input.facts, (f) => f.key);
    for (const f of facts) g.node(factNode(f, rootId, g));
    for (const f of facts) {
        for (const a of sortBy(f.fileAnchors, (x) => x.path)) {
            const to = srcRef(rootId, a.path);
            if (g.nodes.has(to)) g.edge({ from: factRef(f.key), to, kind: 'about', weight: 1, origin: f.origin });
        }
    }

    for (const r of sortBy(input.relations, (x) => x.from, (x) => x.to, (x) => x.kind)) {
        g.edge({
            from: r.from, to: r.to, kind: r.kind, weight: r.weight, origin: r.origin,
            ...(r.confidence !== undefined ? { confidence: r.confidence } : {}),
        });
    }

    for (const p of sortBy(input.nearDuplicates ?? [], (x) => x.a, (x) => x.b)) {
        if (p.a === p.b || p.a.startsWith('ent:') || p.b.startsWith('ent:')) continue;
        const [a, b] = (compareStr(p.a, p.b) < 0 ? [p.a, p.b] : [p.b, p.a]) as [NodeRef, NodeRef];
        if (!g.nodes.has(a) || !g.nodes.has(b) || g.has(a, b, 'duplicate') || g.has(b, a, 'duplicate')) continue;
        g.edge({ from: a, to: b, kind: 'near-duplicate', weight: round6(p.similarity), origin: 'extracted' });
    }

    const changed = applyLive(input, g);
    closeEndpoints(g);
    const registries = new Map(input.packs.map((p) => [p.id, loadPartitions(p)] as const));
    placeNodes(rootId, g, registries);

    const rank = pageRank([...g.nodes.keys()], [...g.edges.values()]);
    for (const n of g.nodes.values()) n.importance = rank.get(n.ref) ?? 0;

    const allNodes = [...g.nodes.values()].map(finalNode).sort((a, b) => compareStr(a.ref, b.ref));
    const allEdges = [...g.edges.values()].map(finalEdge).sort(compareEdges);
    const packs = input.packs.map((p) => p.id).sort(compareStr);
    const full: KnowledgeGraph = {
        scope: { rootId, packs, digest: graphDigest(input) },
        nodes: allNodes, edges: allEdges, findings: [],
        stats: { sources: sources.size, nodes: allNodes.length, edges: allEdges.length, stale: 0, truncated: false },
    };
    full.findings = computeFindings(full, input.packs, registries, input.nearDuplicates ?? []);
    full.stats.stale = allNodes.filter((n) => n.stale).length;

    const maxNodes = Math.min(MAX_NODES_CAP, Math.max(1, Math.floor(opts.maxNodes ?? DEFAULT_MAX_NODES)));
    const cut = truncateGraph(full, { maxNodes, focus: opts.focus ?? [], changed });
    return {
        ...full,
        nodes: cut.nodes,
        edges: cut.edges,
        stats: { ...full.stats, nodes: cut.nodes.length, edges: cut.edges.length, truncated: cut.truncated },
    };
}

function applyLive(input: BuildGraphInput, g: GraphBuilder): NodeRef[] {
    const changed: NodeRef[] = [];
    if (!input.live) return changed;
    for (const c of sortBy(input.live.changes, (x) => x.path)) {
        const ref = srcRef(input.rootId, c.path);
        if (!g.nodes.has(ref)) continue;
        g.node({ ...g.nodes.get(ref)!, badges: { added: c.added, removed: c.removed, status: c.status } });
        changed.push(ref);
    }
    const run = [...new Set(input.live.runFiles ?? [])].sort(compareStr)
        .map((p) => srcRef(input.rootId, p)).filter((r) => g.nodes.has(r));
    for (let i = 0; i < run.length; i++) {
        for (let j = i + 1; j < run.length; j++) {
            g.edge({ from: run[i]!, to: run[j]!, kind: 'co-changed', weight: 1, origin: 'agent' });
        }
    }
    return changed;
}

/** A missing `src:` endpoint becomes a `missing` placeholder (a broken link); a missing fact drops the edge. */
function closeEndpoints(g: GraphBuilder): void {
    for (const [id, e] of [...g.edges.entries()].sort((a, b) => compareStr(a[0], b[0]))) {
        const ends = [e.from, e.to].filter((end) => !g.nodes.has(end));
        if (ends.some((end) => pathOfRef(end) === undefined)) { g.edges.delete(id); continue; }
        for (const end of ends) {
            const path = pathOfRef(end)!;
            const other = g.nodes.get(end === e.from ? e.to : e.from);
            g.node({
                ref: end, kind: 'missing', label: basenameOf(path) || path, pack: other?.pack ?? 'memory',
                importance: 0, origin: 'extracted', badges: { missing: 1 },
            });
        }
    }
}

function placeNodes(rootId: string, g: GraphBuilder, registries: Map<string, PartitionRegistry>): void {
    for (const n of g.nodes.values()) {
        const path = pathOfRef(n.ref);
        if (path === undefined) continue;
        const reg = registries.get(n.pack);
        const partition = reg ? partitionOf(path, reg) : undefined;
        if (partition !== undefined) n.partition = partition;
        const area = areaOfPath(path);
        if (area) n.area = srcRef(rootId, area);
    }
    for (const e of g.edges.values()) {
        const from = g.nodes.get(e.from), to = g.nodes.get(e.to);
        const reg = from && registries.get(from.pack);
        if (!from || !to || !reg || from.pack !== to.pack) continue;
        const verdict = verdictOf(from.partition, to.partition, reg);
        if (verdict) e.verdict = verdict;
    }
}

function finalNode(n: KNode): KNode {
    const badges = n.badges && Object.keys(n.badges).length
        ? Object.fromEntries(Object.keys(n.badges).sort(compareStr).map((k) => [k, n.badges![k]!]))
        : undefined;
    return {
        ref: n.ref, kind: n.kind, label: n.label, pack: n.pack,
        ...(n.partition !== undefined ? { partition: n.partition } : {}),
        ...(n.area !== undefined ? { area: n.area } : {}),
        importance: n.importance, origin: n.origin,
        ...(n.sourceHash !== undefined ? { sourceHash: n.sourceHash } : {}),
        ...(n.stale ? { stale: true } : {}),
        ...(badges ? { badges } : {}),
    };
}

export function finalEdge(e: KEdge): KEdge {
    return {
        from: e.from, to: e.to, kind: e.kind, weight: round6(e.weight), origin: e.origin,
        ...(e.confidence !== undefined ? { confidence: round6(e.confidence) } : {}),
        ...(e.verdict ? { verdict: e.verdict } : {}),
        ...(e.cycle ? { cycle: true } : {}),
    };
}
