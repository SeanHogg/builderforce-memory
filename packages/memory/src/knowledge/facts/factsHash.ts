/**
 * knowledge/facts/factsHash.ts — the identity of a `FileFacts` object (PRD 41 §B1, §B4, §B8): its WIRE form, its canonical
 * JSON, its content address (`factsHash`) and the per-kind hashes early cutoff compares.
 *
 * Hash stability across additive fields: `mounts` and `uses` (§C5.1) were added after v3 shipped. A canonical object always
 * carries them (`[]` when none), but the wire form OMITS them when empty — as it omits every absent optional field — so a
 * facts object written before they existed keeps its `factsHash` and kind hashes when read back. A golden test pins one
 * address computed by the code before the change (factsHash.test.ts).
 */

import { canonicalJson } from '../refs.js';
import { sha256Hex } from '../../retrieval/knowledge-index/sha256.js';
import { canonicalFileFacts, FACT_KIND_FIELDS, type FactKind, type FileFacts } from './fileFacts.js';

/** The optional list fields added after v3 shipped; omitted from the wire form when empty. */
const ADDED_LIST_FIELDS = ['mounts', 'uses'] as const;

const wireCache = new WeakMap<FileFacts, FileFacts>();

/**
 * The WIRE form of `f` (canonicalized): the canonical object with each empty added list field (`mounts`, `uses`) omitted.
 * What every hash, the stored object JSON and the dump are built from. Frozen; cached per canonical object.
 */
export function factsWireForm(f: FileFacts): FileFacts {
    const c = canonicalFileFacts(f);
    let hit = wireCache.get(c);
    if (!hit) {
        const out: FileFacts = { ...c };
        for (const field of ADDED_LIST_FIELDS) if ((out[field]?.length ?? 0) === 0) delete out[field];
        hit = Object.freeze(out);
        wireCache.set(c, hit);
    }
    return hit;
}

/** The canonical JSON of a facts object (its wire form): the bytes a fact store keeps under {@link factsHash}. */
export function factsJson(f: FileFacts): string {
    return canonicalJson(factsWireForm(f));
}

/** The content address of a facts object: sha256 of its canonical JSON ({@link factsJson}). */
export function factsHash(f: FileFacts): string {
    return sha256Hex(factsJson(f));
}

/** The header every kind hash includes. */
const headerOf = (f: FileFacts) => ({ factsVersion: f.factsVersion, extractor: f.extractor, language: f.language });

const kindCache = new WeakMap<FileFacts, Readonly<Record<FactKind, string>>>();

/**
 * The hash of each fact kind of `f` (canonicalized first). `content` is the content hash itself; `structure` and `body`
 * are sha256 over the canonical JSON of the header plus that kind's fields (from the wire form). Cached per CANONICAL
 * (deep-frozen) object; a non-canonical input is canonicalized and hashed again on every call, so mutating it can never
 * serve a stale hash.
 */
export function factKindHashes(f: FileFacts): Readonly<Record<FactKind, string>> {
    const c = canonicalFileFacts(f);
    const hit = kindCache.get(c);
    if (hit) return hit;
    const w = factsWireForm(c);
    const of = (kind: FactKind) => {
        const body: Record<string, unknown> = { header: headerOf(c) };
        // An empty added list is absent from the wire form, so it never enters the hash (canonicalJson drops undefined).
        for (const field of FACT_KIND_FIELDS[kind]) body[field] = w[field];
        return sha256Hex(canonicalJson(body));
    };
    const hashes = Object.freeze({ body: of('body'), content: c.contentHash, structure: of('structure') });
    kindCache.set(c, hashes);
    return hashes;
}
