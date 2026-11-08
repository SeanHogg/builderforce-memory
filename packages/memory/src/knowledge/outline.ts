/**
 * knowledge/outline.ts — the whole graph's files and file-to-file references, kept when the
 * graph is truncated so an overview (stations, layers, lines) never depends on which nodes
 * the cut kept. Built from the FULL graph; everything is sorted, so equal input is byte-identical.
 */

import type { EdgeVerdict, GraphOutline, KnowledgeGraph, NodeRef, OutlineFile } from './model.js';
import { communityTable } from './communities.js';
import { byImportanceAssetsLast } from './importance.js';
import { compareStr, fileRefOf } from './refs.js';

export const OUTLINE_MAX_FILES = 20000;
export const OUTLINE_MAX_LINKS = 60000;
/**
 * The only badges an outline file carries. Mirrors agent-tools' CODE_BADGES plus the live-overlay
 * badges (added/removed/status); memory cannot import agent-tools, so the list is kept here.
 */
export const OUTLINE_BADGE_KEYS: readonly string[] = [
    'complexity', 'maxComplexity', 'coverage', 'churn', 'owner', 'patterns', 'cloneGroup', 'unused', 'unusedExports',
    'entry', 'layerRule', 'unusedDependencies', 'externals', 'added', 'removed', 'changed', 'status',
    'communityId', 'communityConfidence',
    // An asset file (assetFile.ts): a binary file the review stores as a node, never read.
    'asset', 'mediaType', 'size',
];
/** Verdict codes of a link, least to worst; code 0 is "no verdict". */
export const OUTLINE_VERDICTS: readonly ('none' | EdgeVerdict)[] = ['none', 'same', 'allowed', 'skips', 'upward'];
/** Bit 1 of a link's flags. */
export const OUTLINE_FLAG_CYCLE = 1;
const HASH_CHARS = 12;

/** The ref of an outline file. */
export function outlineRef(o: Pick<GraphOutline, 'prefix'>, f: Pick<OutlineFile, 'path'>): NodeRef {
    return (o.prefix + f.path) as NodeRef;
}

/** The ONE entry point for consumers: the attached outline of a truncated graph, else one built from the graph itself. */
export function outlineOf(g: KnowledgeGraph): GraphOutline {
    return g.outline ?? buildOutline(g);
}

function commonPrefix(refs: readonly string[]): string {
    const first = refs[0];
    if (first === undefined) return '';
    const sep = first.indexOf('::');
    if (sep < 0) return '';
    const prefix = first.slice(0, sep + 2);
    return refs.every((r) => r.startsWith(prefix)) ? prefix : '';
}

interface Merged { from: number; to: number; verdict: number; cycle: boolean; weight: number }

export function buildOutline(full: KnowledgeGraph): GraphOutline {
    let capped = false;
    let fileNodes = full.nodes.filter((n) => n.kind === 'file');
    if (fileNodes.length > OUTLINE_MAX_FILES) {
        fileNodes = [...fileNodes].sort(byImportanceAssetsLast).slice(0, OUTLINE_MAX_FILES);
        capped = true;
    }
    fileNodes = [...fileNodes].sort((a, b) => compareStr(a.ref, b.ref));

    const prefix = commonPrefix(fileNodes.map((n) => n.ref));
    const areas = [...new Set(fileNodes.flatMap((n) => (n.area !== undefined ? [n.area] : [])))].sort(compareStr);
    const partitions = [...new Set(fileNodes.flatMap((n) => (n.partition !== undefined ? [n.partition] : [])))].sort(compareStr);
    const areaIndex = new Map<NodeRef, number>(areas.map((a, i) => [a, i] as const));
    const partitionIndex = new Map<string, number>(partitions.map((p, i) => [p, i] as const));

    const systems = [...new Set(fileNodes.flatMap((n) => [...(n.system !== undefined ? [n.system] : []), ...(n.copyOf !== undefined ? [n.copyOf] : [])]))].sort(compareStr);
    const apps = [...new Set(fileNodes.flatMap((n) => (n.app !== undefined ? [n.app] : [])))].sort(compareStr);
    const systemIndex = new Map<string, number>(systems.map((s, i) => [s, i] as const));
    const appIndex = new Map<string, number>(apps.map((a, i) => [a, i] as const));

    const files: OutlineFile[] = fileNodes.map((n) => {
        const badges: Record<string, number | string> = {};
        for (const k of OUTLINE_BADGE_KEYS) {
            const v = n.badges?.[k];
            if (v !== undefined) badges[k] = v;
        }
        return {
            path: n.ref.slice(prefix.length),
            ...(n.area !== undefined ? { area: areaIndex.get(n.area)! } : {}),
            ...(n.partition !== undefined ? { partition: partitionIndex.get(n.partition)! } : {}),
            ...(n.system !== undefined ? { system: systemIndex.get(n.system)! } : {}),
            ...(n.app !== undefined ? { app: appIndex.get(n.app)! } : {}),
            ...(n.copyOf !== undefined ? { copyOf: systemIndex.get(n.copyOf)! } : {}),
            importance: n.importance,
            ...(n.sourceHash !== undefined ? { hash: n.sourceHash.slice(0, HASH_CHARS) } : {}),
            ...(Object.keys(badges).length > 0 ? { badges } : {}),
        };
    });

    const index = new Map<NodeRef, number>(fileNodes.map((n, i) => [n.ref, i] as const));
    const merged = new Map<string, Merged>();
    for (const e of full.edges) {
        if (e.kind !== 'references') continue;
        const from = index.get(fileRefOf(e.from));
        const to = index.get(fileRefOf(e.to));
        if (from === undefined || to === undefined || from === to) continue;
        const key = `${from}>${to}`;
        const verdict = e.verdict === undefined ? 0 : OUTLINE_VERDICTS.indexOf(e.verdict);
        const hit = merged.get(key);
        if (hit) {
            hit.verdict = Math.max(hit.verdict, verdict);
            hit.cycle ||= e.cycle === true;
            hit.weight += e.weight ?? 0;
        } else {
            merged.set(key, { from, to, verdict, cycle: e.cycle === true, weight: e.weight ?? 0 });
        }
    }
    let list = [...merged.values()];
    if (list.length > OUTLINE_MAX_LINKS) {
        const imp = (m: Merged) => fileNodes[m.from]!.importance + fileNodes[m.to]!.importance;
        list = list
            .sort((a, b) => b.weight - a.weight || imp(b) - imp(a) || a.from - b.from || a.to - b.to)
            .slice(0, OUTLINE_MAX_LINKS);
        capped = true;
    }
    list.sort((a, b) => a.from - b.from || a.to - b.to);
    const links = list.map((m) => [m.from, m.to, m.verdict, m.cycle ? OUTLINE_FLAG_CYCLE : 0] as [number, number, number, number]);
    const communities = communityTable(fileNodes, prefix);
    return {
        prefix, areas, partitions,
        ...(communities.length > 0 ? { communities } : {}),
        ...(systems.length > 0 ? { systems } : {}),
        ...(apps.length > 0 ? { apps } : {}),
        ...(full.meta?.systems ? { systemInfo: full.meta.systems } : {}),
        files, links, capped,
    };
}
