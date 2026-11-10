/**
 * knowledge/refs.ts — the NodeRef grammar and the deterministic primitives every
 * knowledge module shares (ordering, canonical JSON, entity keys).
 *
 * Pure and import-free of `ingest` runtime code, so ingest's extractors can use it
 * without an import cycle.
 *
 *   src:${rootId}::${path}            a source file (or an area: a directory path)
 *   src:${rootId}::${path}#${slug}    a section of a source
 *   ent:${pack}:${normalizedName}     an entity (also its project_facts key)
 *   fact:${key}                       any other fact (claims: fact:claim:${hash})
 */

import type { KEdge, NodeRef } from './model.js';

/** The generic pack id: what a node or finding belongs to when no real pack owns it (memory facts, `missing` placeholders, asset files). */
export const FALLBACK_PACK_ID = 'memory';

/** Code-unit ordering — never `localeCompare`, whose result depends on the host's ICU. */
export function compareStr(a: string, b: string): number {
    return a < b ? -1 : a > b ? 1 : 0;
}

export function srcRef(rootId: string, path: string): NodeRef {
    return `src:${rootId}::${path}`;
}

/**
 * Splits `src:<rootId>::<path>` into its root and its FULL path (never section-stripped: `src/C#/a.cs` stays whole). A root id
 * may end in ':' (the linked repo's `cloud:<id>:`), so the colons after the first `::` that directly follow it belong to the
 * root: `src:cloud:5:::docs/a.md` is root `cloud:5:`, path `docs/a.md`. (A path that itself starts with ':' is read as part of
 * the root — root ids never hold '::' nor end in ':' when built by `rootIdFor`, so only a foreign id can make it ambiguous.)
 * Pass `knownRootIds` where the root is known: a ref of one of them is split by its exact prefix, so even a path that starts
 * with ':' is exact. Undefined when `ref` is not a source ref. THE grammar: every reader of a `src:` ref goes through this.
 */
export function splitSourceRef(ref: string, knownRootIds: readonly string[] = []): { rootId: string; path: string } | undefined {
    if (!ref.startsWith('src:')) return undefined;
    for (const rootId of knownRootIds) if (ref.startsWith(`src:${rootId}::`)) return { rootId, path: ref.slice(`src:${rootId}::`.length) };
    const at = ref.indexOf('::', 4);
    if (at <= 4) return undefined;
    let end = at;
    while (ref[end + 2] === ':') end += 1;
    return { rootId: ref.slice(4, end), path: ref.slice(end + 2) };
}

/** The root-relative path of a `src:` ref (the section suffix dropped), else undefined. Splits by {@link splitSourceRef}. */
export function pathOfRef(ref: string): string | undefined {
    const rest = splitSourceRef(ref)?.path;
    if (rest === undefined) return undefined;
    const hash = rest.indexOf('#');
    return hash < 0 ? rest : rest.slice(0, hash);
}

/** The ref of the file a `src:` ref belongs to (itself for a file): the ref cut at the first '#' of its path ({@link splitSourceRef}). */
export function fileRefOf(ref: NodeRef): NodeRef {
    const split = splitSourceRef(ref);
    if (!split) return ref;
    const hash = split.path.indexOf('#');
    return hash < 0 ? ref : (ref.slice(0, ref.length - split.path.length + hash) as NodeRef);
}

/** The node ref of a stored fact key: `ent:` keys are refs already; the rest are `fact:`. */
export function factRef(key: string): NodeRef {
    return key.startsWith('ent:') ? (key as NodeRef) : `fact:${key}`;
}

/** The stored fact key of a fact/entity ref (inverse of {@link factRef}). */
export function factKeyOf(ref: NodeRef): string | undefined {
    if (ref.startsWith('ent:')) return ref;
    if (ref.startsWith('fact:')) return ref.slice(5);
    return undefined;
}

/**
 * The normalized key of a name: NFKC, case-folded, punctuation stripped (dashes,
 * underscores and slashes read as spaces so `run-store` and `run store` agree),
 * whitespace collapsed.
 */
export function normalizeKey(name: string): string {
    return name
        .normalize('NFKC')
        .toLowerCase()
        .replace(/[\p{Pd}\p{Pc}/\\]+/gu, ' ')
        .replace(/[\p{P}\p{S}]+/gu, '')
        .replace(/\s+/g, ' ')
        .trim();
}

/** A heading slug: the normalized key with spaces as dashes. */
export function slugOf(text: string): string {
    return normalizeKey(text).replace(/ /g, '-') || 'section';
}

export function basenameOf(path: string): string {
    const i = path.lastIndexOf('/');
    return i < 0 ? path : path.slice(i + 1);
}

/** Rounds to 6 decimals so floating noise never reaches the JSON. */
export function round6(n: number): number {
    return Math.round(n * 1e6) / 1e6;
}

/** JSON with object keys sorted and `undefined` members dropped — the canonical form for digests. */
export function canonicalJson(value: unknown): string {
    return JSON.stringify(sortKeys(value));
}

function sortKeys(value: unknown): unknown {
    if (Array.isArray(value)) return value.map(sortKeys);
    if (value === null || typeof value !== 'object') return value;
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value as Record<string, unknown>).sort(compareStr)) {
        const v = (value as Record<string, unknown>)[key];
        if (v !== undefined) out[key] = sortKeys(v);
    }
    return out;
}

/** Sorts a copy by the given string keys, in order. */
export function sortBy<T>(items: readonly T[], ...keys: Array<(t: T) => string>): T[] {
    return [...items].sort((a, b) => {
        for (const key of keys) {
            const c = compareStr(key(a), key(b));
            if (c !== 0) return c;
        }
        return 0;
    });
}

/**
 * A relative target walked from the linking file's folder (a leading `/` is the root):
 * `segments` is where it lands under the root, `ups` how many levels it climbed ABOVE the
 * root. Pure path arithmetic — the one place that decides "inside", "root itself" and "escapes".
 */
export function walkRelative(fromPath: string, target: string): { ups: number; segments: string[] } {
    const t = target.replace(/\\/g, '/');
    const segments = t.startsWith('/') ? [] : fromPath.split('/').slice(0, -1);
    let ups = 0;
    for (const seg of t.split('/')) {
        if (seg === '' || seg === '.') continue;
        if (seg === '..') {
            if (segments.length) segments.pop();
            else ups += 1;
        } else segments.push(seg);
    }
    return { ups, segments };
}

/**
 * `[[Name]]` → the one listed file named `Name` (or `Name.md`), case-insensitively, anywhere
 * in the root. Ambiguous names resolve to none, so the link keeps its beside-the-file fallback.
 */
export function nameResolver(paths: readonly string[]): (name: string) => string | undefined {
    const byName = new Map<string, string | null>();
    for (const p of [...paths].sort(compareStr)) {
        const base = basenameOf(p).toLowerCase();
        for (const key of new Set([base, base.replace(/\.md$/, '')])) {
            byName.set(key, byName.has(key) && byName.get(key) !== p ? null : p);
        }
    }
    return (name) => byName.get(name.trim().toLowerCase()) ?? undefined;
}

/**
 * The `about` target of each node, from edges in graph order. By default the first; with
 * `preferSection` a section wins over the file it belongs to (the most specific source).
 */
export function aboutTargets(edges: readonly KEdge[], opts: { preferSection?: boolean } = {}): Map<NodeRef, NodeRef> {
    const out = new Map<NodeRef, NodeRef>();
    for (const e of edges) {
        if (e.kind !== 'about') continue;
        const prior = out.get(e.from);
        if (!prior || (opts.preferSection && prior === fileRefOf(prior) && e.to !== fileRefOf(e.to))) out.set(e.from, e.to);
    }
    return out;
}
