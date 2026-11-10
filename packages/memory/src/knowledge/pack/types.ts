/**
 * knowledge/pack/types.ts — the BFKP v1 data model (PRD 41 A11): what `encodePack` takes and
 * `decodePack` returns. Rows reuse the knowledge model's own types (`StoredFact`, `StoredRelation`,
 * `Extracted`, `ScopeRecord`), so a pack carries exactly what the single writer stores.
 */

import type { CallSite, Extracted, KEdge, KNode, NodeRef, ScopeRecord, StoredFact, StoredOrigin, StoredRelation } from '../model.js';
import type { Budget } from './varint.js';

/** A reviewed file: its root-relative path, the hash its extracted output is keyed under, and its size. */
export interface PackFileRow { path: string; hash: string; bytes: number }

/** A source's current hash (what the old `markStale` call recorded). '' = the file was removed. */
export interface PackStaleRow { sourceRef: string; hash: string }

/** A relation's upsert identity. */
export interface RelationId { from: string; to: string; kind: string }

/** Tombstones: keys the sender no longer has. */
export interface PackDeletes { files: string[]; facts: string[]; relations: RelationId[] }

/** One list of a re-extracted entry, as a diff against the prior extraction: items to drop and items to insert at a position. */
export interface ListDiff<T> { remove: string[]; add: Array<{ at: number; item: T }> }

/** What changed in one source's extraction since the acknowledged prior one (same `sourceRef`). */
export interface ExtrEntryDiff {
    sourceRef: string;
    sourceHash: string;
    extractorVersion?: number;
    nodes: ListDiff<KNode>;
    edges: ListDiff<KEdge>;
    keys: ListDiff<{ ref: string; key: string }>;
}

/** The Extracted output of one content hash: whole, or as a diff against the prior hash's output. */
export type PackExtr =
    | { kind: 'full'; hash: string; entries: Extracted[] }
    | { kind: 'diff'; hash: string; prior: string; entries: ExtrEntryDiff[] };

/** A leaf of the manifest tree: a stable key and the digest of its content. */
export interface ManifestLeaf { key: string; digest: string }

/** Everything a pack can carry besides its frame fields. Every part is optional (an empty section is not written). */
export interface PackBody {
    files?: PackFileRow[];
    extracted?: PackExtr[];
    facts?: StoredFact[];
    relations?: StoredRelation[];
    stale?: PackStaleRow[];
    deletes?: PackDeletes;
    scope?: ScopeRecord;
    /** The analysis snapshot (JSON-compatible). */
    analysis?: unknown;
    /** Hashes named for their own sake (a `have` request). Hashes a section already references are not repeated here. */
    hashes?: string[];
    /** Manifest leaves (a reconcile answer). */
    leaves?: ManifestLeaf[];
    /** The 4,096 bucket digests of the manifest tree, hex (a manifest GET answer). */
    buckets?: string[];
}

/** The frame fields plus the body. */
export interface Pack extends PackBody {
    rootId: string;
    /** The server generation this pack applies to; 0 = a full pack. */
    baseGeneration: number;
    /** The digest the server must reach after applying (64 hex); `ZERO_DIGEST` when the pack asserts none. */
    targetDigest: string;
}

export const ZERO_DIGEST = '0'.repeat(64);

/** A pack's decoded counts, for guards before any work. */
export interface PackLimits {
    /** Bytes the body may inflate to (default 24 MB). */
    maxDecodedBytes: number;
    /** Strings in the table (default 400,000). */
    maxStrings: number;
    /** Rows in the whole pack (default 200,000). */
    maxRows: number;
}

/**
 * The limits a decode reads under. `maxRows` is CUMULATIVE: one `budget` is drawn down by every list of every
 * section, so a pack of many small lists is bounded as a whole. `maxDecodedBytes` bounds the inflated body AND the
 * total bytes of the string table.
 */
export interface ReadLimits extends PackLimits { budget: Budget }

/** Sized for a 128 MB Worker: the decoded pack, its string table and its rows all live in memory at once. A client cuts a pack below these. */
export const DEFAULT_PACK_LIMITS: PackLimits = { maxDecodedBytes: 24 * 1024 * 1024, maxStrings: 400_000, maxRows: 200_000 };

/** The decoded body inflated past `maxDecodedBytes`: the server answers 413 `pack_too_large`. */
export class PackTooLargeError extends Error {
    constructor(readonly limit: number) { super(`pack inflates past ${limit} bytes`); this.name = 'PackTooLargeError'; }
}

export type { CallSite, Extracted, KEdge, KNode, NodeRef, ScopeRecord, StoredFact, StoredOrigin, StoredRelation };
