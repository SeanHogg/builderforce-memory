/**
 * knowledge/pack/extrDiff.ts — within-file diff (PRD 41 A11 rule 2): a re-extracted file sends the
 * nodes, edges and keys that changed against the acknowledged prior extraction of the same source,
 * plus the identities that vanished, instead of its whole extraction.
 *
 * Lossless by construction: `applyEntryDiff(prior, diffEntry(prior, next))` deep-equals `next`
 * (order included). `diffEntry` returns undefined when it cannot promise that — an identity that
 * repeats within a list, or surviving items that changed relative order — and the caller sends the whole entry.
 */

import type { Extracted, KEdge, KNode } from './types.js';
import type { ExtrEntryDiff, ListDiff } from './types.js';

export function deepEqual(a: unknown, b: unknown): boolean {
    if (a === b) return true;
    if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return Object.is(a, b);
    if (Array.isArray(a) !== Array.isArray(b)) return false;
    if (Array.isArray(a)) return a.length === (b as unknown[]).length && a.every((x, i) => deepEqual(x, (b as unknown[])[i]));
    const ka = Object.keys(a as object).filter((k) => (a as Record<string, unknown>)[k] !== undefined);
    const kb = Object.keys(b as object).filter((k) => (b as Record<string, unknown>)[k] !== undefined);
    return ka.length === kb.length && ka.every((k) => deepEqual((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

function diffList<T>(prior: readonly T[], next: readonly T[], id: (x: T) => string): ListDiff<T> | undefined {
    const before = new Map<string, T>();
    for (const p of prior) { const k = id(p); if (before.has(k)) return undefined; before.set(k, p); }
    const seen = new Set<string>();
    for (const n of next) { const k = id(n); if (seen.has(k)) return undefined; seen.add(k); }
    const survives = (n: T): boolean => before.has(id(n)) && deepEqual(before.get(id(n)), n);
    const survivorsNext = next.filter(survives).map(id);
    const after = new Map(next.map((n) => [id(n), n] as const));
    const survivorsPrior = prior.filter((p) => after.has(id(p)) && deepEqual(p, after.get(id(p)))).map(id);
    if (survivorsNext.length !== survivorsPrior.length || survivorsNext.some((k, i) => k !== survivorsPrior[i])) return undefined;
    const keep = new Set(survivorsNext);
    return {
        remove: prior.map(id).filter((k) => !keep.has(k)),
        add: next.flatMap((item, at) => (keep.has(id(item)) ? [] : [{ at, item }])),
    };
}

function applyList<T>(prior: readonly T[], diff: ListDiff<T>, id: (x: T) => string): T[] {
    const gone = new Set(diff.remove);
    const kept = prior.filter((p) => !gone.has(id(p)));
    const total = kept.length + diff.add.length;
    const out: T[] = new Array(total);
    for (const a of diff.add) out[a.at] = a.item;
    let k = 0;
    for (let i = 0; i < total; i++) if (!(i in out)) out[i] = kept[k++]!;
    return out;
}

const nodeId = (n: KNode): string => n.ref;
const edgeId = (e: KEdge): string => `${e.from}\u0000${e.to}\u0000${e.kind}`;
const keyId = (k: { ref: string; key: string }): string => `${k.ref}\u0000${k.key}`;

export function diffEntry(prior: Extracted, next: Extracted): ExtrEntryDiff | undefined {
    if (prior.sourceRef !== next.sourceRef) return undefined;
    const nodes = diffList(prior.nodes, next.nodes, nodeId);
    const edges = diffList(prior.edges, next.edges, edgeId);
    const keys = diffList(prior.keys, next.keys, keyId);
    if (!nodes || !edges || !keys) return undefined;
    return {
        sourceRef: next.sourceRef, sourceHash: next.sourceHash,
        ...(next.extractorVersion !== undefined ? { extractorVersion: next.extractorVersion } : {}),
        nodes, edges, keys,
    };
}

export function applyEntryDiff(prior: Extracted, diff: ExtrEntryDiff): Extracted {
    return {
        sourceRef: diff.sourceRef as Extracted['sourceRef'], sourceHash: diff.sourceHash,
        ...(diff.extractorVersion !== undefined ? { extractorVersion: diff.extractorVersion } : {}),
        nodes: applyList(prior.nodes, diff.nodes, nodeId),
        edges: applyList(prior.edges, diff.edges, edgeId),
        keys: applyList(prior.keys, diff.keys, keyId) as Extracted['keys'],
    };
}

/**
 * The entries of `next` as diffs against the entries of `prior` with the same `sourceRef`, or undefined
 * when any entry has no usable prior (then the hash is sent whole). A diff only pays when it is smaller:
 * the caller compares sizes if it cares; this decides only what is possible.
 */
export function diffEntries(prior: readonly Extracted[], next: readonly Extracted[]): ExtrEntryDiff[] | undefined {
    const out: ExtrEntryDiff[] = [];
    for (const n of next) {
        const p = prior.find((x) => x.sourceRef === n.sourceRef);
        const d = p && diffEntry(p, n);
        if (!d) return undefined;
        out.push(d);
    }
    return out;
}

/** Rebuilds a hash's entries from its diffs and the prior hash's entries; throws when a prior entry is missing. */
export function applyEntryDiffs(prior: readonly Extracted[], diffs: readonly ExtrEntryDiff[]): Extracted[] {
    return diffs.map((d) => {
        const p = prior.find((x) => x.sourceRef === d.sourceRef);
        if (!p) throw new Error(`no prior extraction for ${d.sourceRef}`);
        return applyEntryDiff(p, d);
    });
}
