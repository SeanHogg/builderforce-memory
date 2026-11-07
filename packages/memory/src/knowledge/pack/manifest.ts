/**
 * knowledge/pack/manifest.ts — `manifestTree`, the ONE function both sides use to find a delta
 * (PRD 41 A11 rule 2, the reconcile path).
 *
 * Leaves are `(key, contentDigest)` for every file, fact and relation of a root, grouped into 4,096
 * buckets by a stable hash of the key. A bucket's digest is the hash of its sorted leaves; the root
 * digest is the hash of the bucket digests. Two sides with equal root digests hold equal state; with
 * unequal ones, only the buckets whose digests differ need their leaves compared.
 *
 * What counts as "a leaf of the root" is decided HERE ({@link rootLeaves}), so the client's local rows
 * and the server's stored rows are cut the same way and a digest means the same on both.
 */

import { sha256Hex } from '../../retrieval/knowledge-index/sha256.js';
import type { ManifestLeaf, StoredFact, StoredRelation } from './types.js';

export const MANIFEST_BUCKETS = 4096;

/** FNV-1a (32-bit) over the key's UTF-16 code units: stable across runtimes and releases. */
export function keyHash(key: string): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < key.length; i++) { h ^= key.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
    return h >>> 0;
}

export const bucketOf = (key: string): number => keyHash(key) % MANIFEST_BUCKETS;

const EMPTY_BUCKET = sha256Hex('');

export interface ManifestTree {
    root: string;
    /** `MANIFEST_BUCKETS` bucket digests, hex. */
    buckets: string[];
    /** The leaves of each non-empty bucket, sorted by key. */
    leaves: Map<number, ManifestLeaf[]>;
}

const cmp = (a: ManifestLeaf, b: ManifestLeaf): number => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0);

export function manifestTree(leaves: readonly ManifestLeaf[]): ManifestTree {
    const grouped = new Map<number, ManifestLeaf[]>();
    for (const l of leaves) {
        const b = bucketOf(l.key);
        const list = grouped.get(b);
        if (list) list.push(l); else grouped.set(b, [l]);
    }
    const buckets: string[] = new Array(MANIFEST_BUCKETS);
    for (let b = 0; b < MANIFEST_BUCKETS; b++) {
        const list = grouped.get(b);
        if (!list) { buckets[b] = EMPTY_BUCKET; continue; }
        list.sort(cmp);
        buckets[b] = sha256Hex(list.map((l) => `${l.key}\u0000${l.digest}\n`).join(''));
    }
    return { root: sha256Hex(buckets.join('')), buckets, leaves: grouped };
}

/** The bucket ids whose digests differ (or are missing on one side). */
export function differingBuckets(a: readonly string[], b: readonly string[]): number[] {
    const out: number[] = [];
    for (let i = 0; i < MANIFEST_BUCKETS; i++) if (a[i] !== b[i]) out.push(i);
    return out;
}

/** A source's leaf key: its path, or `~tag|path` for a root-level output (the blob store's own source identity). */
export function fileLeafKey(path: string, hash: string): string {
    if (!hash.startsWith('~')) return `f:${path}`;
    const colon = hash.indexOf(':');
    return `f:${colon > 0 ? hash.slice(0, colon) : hash}|${path}`;
}

export const factLeafKey = (key: string): string => `a:${key}`;
export const relationLeafKey = (r: { from: string; to: string; kind: string }): string => `r:${r.from}\u0000${r.to}\u0000${r.kind}`;

/** The digest of what a fact says: its key, text and provenance kind. Importance and timestamps are the server's to score. */
export const factDigest = (f: Pick<StoredFact, 'key' | 'content' | 'origin'>): string => sha256Hex(JSON.stringify([f.key, f.content, f.origin]));

export const relationDigest = (r: Pick<StoredRelation, 'weight' | 'origin' | 'confidence' | 'sourceHash'>): string =>
    sha256Hex(`${r.weight.toFixed(3)}|${r.origin}|${r.confidence === undefined ? '' : r.confidence.toFixed(3)}|${r.sourceHash ?? ''}`);

export interface RootRows {
    files: ReadonlyArray<{ path: string; hash: string }>;
    facts: readonly StoredFact[];
    relations: readonly StoredRelation[];
}

/**
 * The leaves of a root. A fact belongs to the root when it is anchored to one of its files; a relation
 * when either end is one of its sources (`src:<root>::…`) or one of those facts.
 */
export function rootLeaves(rootId: string, rows: RootRows): ManifestLeaf[] {
    const paths = new Set(rows.files.map((f) => f.path));
    const facts = rows.facts.filter((f) => f.fileAnchors.some((a) => paths.has(a.path)));
    const factRefs = new Set(facts.map((f) => `fact:${f.key}`));
    const prefix = `src:${rootId}::`;
    const touches = (ref: string): boolean => ref.startsWith(prefix) || factRefs.has(ref);
    return [
        ...rows.files.map((f) => ({ key: fileLeafKey(f.path, f.hash), digest: f.hash })),
        ...facts.map((f) => ({ key: factLeafKey(f.key), digest: factDigest(f) })),
        ...rows.relations.filter((r) => touches(r.from) || touches(r.to)).map((r) => ({ key: relationLeafKey(r), digest: relationDigest(r) })),
    ];
}

/** The root digest of a root's rows: what a pack asserts as its `targetDigest`. */
export const rootDigestOf = (rootId: string, rows: RootRows): string => manifestTree(rootLeaves(rootId, rows)).root;

export interface LeafDelta {
    /** Keys the remote lacks or holds with another digest. */
    upsert: string[];
    /** Keys the remote holds that the local side does not. */
    remove: string[];
}

/** Compares local leaves with the remote's leaves of the differing buckets; only those buckets are considered. */
export function leafDelta(local: ManifestTree, remote: readonly ManifestLeaf[], buckets: readonly number[]): LeafDelta {
    const remoteByKey = new Map(remote.map((l) => [l.key, l.digest] as const));
    const upsert: string[] = [], seen = new Set<string>();
    for (const b of buckets) {
        for (const l of local.leaves.get(b) ?? []) {
            seen.add(l.key);
            if (remoteByKey.get(l.key) !== l.digest) upsert.push(l.key);
        }
    }
    return { upsert, remove: remote.map((l) => l.key).filter((k) => !seen.has(k)) };
}
