/**
 * knowledge/cards.ts — `toCards`: one knowledge-index card per graph node
 * (design: Builderforce.ai/docs/design/deterministic-knowledge-index.md, "Cards").
 *
 * | kind            | text                                                              | keys                          |
 * |-----------------|-------------------------------------------------------------------|-------------------------------|
 * | file            | path · area · partition · symbols · doc line · top-5 neighbours   | path, basename, symbols, area |
 * | section         | file · breadcrumb · first 400 chars · outgoing link targets       | file path, slug, front matter |
 * | entity (ent:)   | name · type · top-5 mentioning sources                            | normalized name, aliases      |
 * | claim           | subject predicate object · quote · source section                 | subject/object entity keys    |
 * | area            | path · partition · top-10 members                                 | path                          |
 *
 * `hash` is the index's own text hash (`hashDocText`), so an unchanged card is never re-embedded.
 */

import { hashDocText } from '../retrieval/knowledge-index/digest.js';
import type { IndexDoc, KEdge, KNode, KnowledgeGraph, NodeRef } from './model.js';
import { byImportance } from './importance.js';
import { entityName } from './resolve.js';
import { basenameOf, compareStr, fileRefOf, normalizeKey, pathOfRef, slugOf } from './refs.js';

const SEP = ' · ';
const EXCERPT_CHARS = 400;

export function toCards(g: KnowledgeGraph): IndexDoc[] {
    const byRef = new Map(g.nodes.map((n) => [n.ref, n] as const));
    const outEdges = new Map<NodeRef, KEdge[]>();
    const inEdges = new Map<NodeRef, KEdge[]>();
    const push = (m: Map<NodeRef, KEdge[]>, k: NodeRef, e: KEdge) => {
        const l = m.get(k) ?? [];
        l.push(e);
        m.set(k, l);
    };
    for (const e of g.edges) { push(outEdges, e.from, e); push(inEdges, e.to, e); }
    const label = (r: NodeRef) => byRef.get(r)?.label ?? pathOfRef(r) ?? r;
    const ctx: CardCtx = { byRef, outEdges, inEdges, label, members: membersByArea(g.nodes) };

    return g.nodes.map((n) => {
        const { text, keys } = cardOf(n, ctx);
        const uniqueKeys = [...new Set(keys.filter(Boolean))].sort(compareStr);
        return { id: n.ref, text, ...(uniqueKeys.length ? { keys: uniqueKeys } : {}), hash: hashDocText(text) };
    }).sort((a, b) => compareStr(a.id, b.id));
}

interface CardCtx {
    byRef: Map<NodeRef, KNode>;
    outEdges: Map<NodeRef, KEdge[]>;
    inEdges: Map<NodeRef, KEdge[]>;
    label(r: NodeRef): string;
    members: Map<NodeRef, KNode[]>;
}

function membersByArea(nodes: readonly KNode[]): Map<NodeRef, KNode[]> {
    const out = new Map<NodeRef, KNode[]>();
    for (const n of nodes) {
        if (!n.area || n.kind !== 'file') continue;
        const l = out.get(n.area) ?? [];
        l.push(n);
        out.set(n.area, l);
    }
    return out;
}

/** Neighbour refs (either direction), heaviest first, then by ref. */
function topNeighbours(ref: NodeRef, ctx: CardCtx, n: number): NodeRef[] {
    const scored = new Map<NodeRef, number>();
    for (const e of ctx.outEdges.get(ref) ?? []) scored.set(e.to, Math.max(scored.get(e.to) ?? 0, e.weight));
    for (const e of ctx.inEdges.get(ref) ?? []) scored.set(e.from, Math.max(scored.get(e.from) ?? 0, e.weight));
    return [...scored].sort((a, b) => b[1] - a[1] || compareStr(a[0], b[0])).slice(0, n).map(([r]) => r);
}

const badge = (n: KNode | undefined, k: string): string | undefined => {
    const v = n?.badges?.[k];
    return v === undefined || v === '' ? undefined : String(v);
};

const frontMatter = (n: KNode | undefined): string[] =>
    Object.keys(n?.badges ?? {}).filter((k) => k.startsWith('fm.')).sort(compareStr)
        .flatMap((k) => String(n!.badges![k]).split(',').map((v) => normalizeKey(v)));

function cardOf(n: KNode, ctx: CardCtx): { text: string; keys: string[] } {
    const path = pathOfRef(n.ref) ?? '';
    const join = (parts: Array<string | undefined>) => parts.filter((p) => p !== undefined && p !== '').join(SEP);
    switch (n.kind) {
        case 'file': {
            const contained = (ctx.outEdges.get(n.ref) ?? []).filter((e) => e.kind === 'contains')
                .map((e) => ctx.byRef.get(e.to)).filter((c): c is KNode => c?.kind === 'symbol');
            const symbols = contained.map((s) => s.label).sort(compareStr);
            return {
                text: join([path, n.area && pathOfRef(n.area), n.partition, symbols.join(', '), badge(n, 'doc'),
                    topNeighbours(n.ref, ctx, 5).map(ctx.label).join(', ')]),
                keys: [path, basenameOf(path), ...symbols, n.area ? pathOfRef(n.area) ?? '' : ''],
            };
        }
        case 'section': {
            const file = ctx.byRef.get(fileRefOf(n.ref));
            const links = (ctx.outEdges.get(n.ref) ?? []).filter((e) => e.kind === 'links')
                .map((e) => pathOfRef(e.to) ?? e.to).sort(compareStr);
            return {
                text: join([path, badge(n, 'breadcrumb') ?? n.label, badge(n, 'excerpt')?.slice(0, EXCERPT_CHARS), links.join(', ')]),
                keys: [path, slugOf(n.label), ...frontMatter(file)],
            };
        }
        case 'entity': {
            const sources = (ctx.inEdges.get(n.ref) ?? []).filter((e) => e.kind === 'mentions' || e.kind === 'about')
                .map((e) => ctx.byRef.get(e.from)).filter((s): s is KNode => !!s);
            const about = (ctx.outEdges.get(n.ref) ?? []).filter((e) => e.kind === 'about')
                .map((e) => ctx.byRef.get(e.to)).filter((s): s is KNode => !!s);
            const top = [...new Map([...sources, ...about].map((s) => [s.ref, s])).values()].sort(byImportance).slice(0, 5);
            const name = entityName(n.ref);
            return {
                text: join([n.label, badge(n, 'type'), top.map((s) => s.label).join(', ')]),
                keys: [normalizeKey(name), normalizeKey(n.label.replace(/\s*\([^)]*\)\s*$/, '')),
                    ...String(badge(n, 'aliases') ?? '').split(',').map(normalizeKey)],
            };
        }
        case 'claim': {
            const out = ctx.outEdges.get(n.ref) ?? [];
            const sections = out.filter((e) => e.kind === 'about').map((e) => ctx.label(e.to)).sort(compareStr);
            const entities = out.filter((e) => e.kind === 'mentions' && e.to.startsWith('ent:')).map((e) => normalizeKey(entityName(e.to)));
            return { text: join([n.label, sections.join(', ')]), keys: entities };
        }
        case 'area': {
            const members = [...(ctx.members.get(n.ref) ?? [])].sort(byImportance).slice(0, 10).map((m) => m.label);
            return { text: join([path, n.partition, members.join(', ')]), keys: [path] };
        }
        default:
            return { text: join([n.label, n.kind, path]), keys: [normalizeKey(n.label), path] };
    }
}
