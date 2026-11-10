/**
 * knowledge/symbolEdges.ts — the symbol-level relationship vocabulary (PRD 43 P0b, B1a rules 9–10): what a code
 * pack's extractor says between two SYMBOLS (a class, an interface, a function, a method), beside the file-level
 * kinds (`references`, `contains` …). Generic: memory names the kinds and their call-site shape; which language
 * construct counts as which kind is the code pack's business.
 *
 *   extends     class/interface → the base it extends
 *   implements  class → an interface it implements
 *   composes    class → a class it holds as a field / property / constructor parameter property (composition, aggregation)
 *   uses        symbol → a type it names in a signature or body (none of the above)
 *   calls       function/method → a function/method it calls; carries its call {@link CallSite}s in source order
 *
 * A target outside the root (a package, a file not extracted) is still a `src:` ref, so `buildGraph` closes it as a
 * `missing` placeholder, exactly as an unresolved import is.
 *
 * Bounded: {@link dedupeSymbolEdges} keeps ONE edge per (from, to, kind) — repeated calls become extra sites on it —
 * and at most {@link MAX_SYMBOL_EDGES_PER_SYMBOL} edges and {@link MAX_CALL_SITES_PER_EDGE} sites per edge.
 */

import type { CallSite, KEdge } from './model.js';

export const SYMBOL_EDGE_KINDS = ['extends', 'implements', 'composes', 'uses', 'calls'] as const;
export type SymbolEdgeKind = (typeof SYMBOL_EDGE_KINDS)[number];

/** Edges one symbol may start (all kinds together): a generated or god function must not flood the graph. */
export const MAX_SYMBOL_EDGES_PER_SYMBOL = 64;
/** Call sites kept on one `calls` edge (the lowest `order`s): a call in a hot loop body is a handful of sites, not hundreds. */
export const MAX_CALL_SITES_PER_EDGE = 16;

export const isSymbolEdgeKind = (kind: string): kind is SymbolEdgeKind => (SYMBOL_EDGE_KINDS as readonly string[]).includes(kind);

/** Sites in source order: by `order`, then position (a well-formed list already is). */
export const compareCallSites = (a: CallSite, b: CallSite): number =>
    a.order - b.order || (a.path < b.path ? -1 : a.path > b.path ? 1 : 0) || a.line - b.line || a.column - b.column;

/**
 * ONE edge per (from, to, kind): weights take the max, sites merge (sorted by {@link compareCallSites}, capped at
 * `maxSites`). Edges of a `from` past `maxPerSymbol` are dropped — the first ones kept, in input order, so an
 * extractor that emits a body's relations in source order keeps the earliest. Non-symbol kinds pass through unchanged
 * and uncounted. Order of the result: first appearance. Pure.
 */
export function dedupeSymbolEdges(
    edges: readonly KEdge[], opts: { maxPerSymbol?: number; maxSites?: number } = {},
): KEdge[] {
    const maxPerSymbol = opts.maxPerSymbol ?? MAX_SYMBOL_EDGES_PER_SYMBOL;
    const maxSites = opts.maxSites ?? MAX_CALL_SITES_PER_EDGE;
    const out: KEdge[] = [];
    const byId = new Map<string, KEdge>();
    const perFrom = new Map<string, number>();
    for (const e of edges) {
        if (!isSymbolEdgeKind(e.kind)) { out.push(e); continue; }
        const id = `${e.from}\u0000${e.to}\u0000${e.kind}`;
        const prior = byId.get(id);
        if (prior) {
            prior.weight = Math.max(prior.weight, e.weight);
            if (e.sites?.length) prior.sites = [...(prior.sites ?? []), ...e.sites];
            continue;
        }
        const n = perFrom.get(e.from) ?? 0;
        if (n >= maxPerSymbol) continue;
        perFrom.set(e.from, n + 1);
        const copy: KEdge = { ...e, ...(e.sites ? { sites: [...e.sites] } : {}) };
        byId.set(id, copy);
        out.push(copy);
    }
    for (const e of out) {
        if (!e.sites) continue;
        if (e.sites.length === 0) { delete e.sites; continue; }
        e.sites = e.sites.sort(compareCallSites).slice(0, maxSites);
    }
    return out;
}
