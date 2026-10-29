/**
 * knowledge/extractFile.ts — the ONE single-file extraction (PRD 38 §4.3): compose the
 * packs, resolve the parser, parse, extract with the context (known entity keys,
 * `resolvePath`), stamp {@link EXTRACTOR_VERSION} and read/write the extracted cache by
 * content hash. `reviewDirectory` runs it per changed file; a host re-extracting one edited
 * file calls {@link extractFile}.
 */

import type { ParsedSection, SourceDocument } from '../ingest/types.js';
import type { ExtractContext, Extracted, KnowledgePack, KnowledgeParser, KnowledgeStore, NodeRef } from './model.js';
import { basenameOf, srcRef } from './refs.js';
import { entityName, knownKeysOf, resolveEntities } from './resolve.js';
import { asPathSource, composePacks, EXTRACTOR_VERSION, resolveRelativePath, type ComposedPacks } from './sources.js';

/** The normalized entity keys of `ent:` fact keys, for `mentions`. */
export function knownKeysOfEntities(keys: Iterable<string>): ReadonlyMap<string, NodeRef> {
    return knownKeysOf(resolveEntities([...keys].filter((k) => k.startsWith('ent:')).map((k) => ({ ref: k as NodeRef, key: entityName(k) }))));
}

export interface ExtractFileArgs {
    rootId: string;
    /** Root-relative path. */
    path: string;
    /** Content hash of the file (the extracted cache's key). */
    hash: string;
    doc: SourceDocument;
    packs: readonly KnowledgePack[];
    store: KnowledgeStore;
    /** Already composed from `packs` (a review composes once). */
    composed?: ComposedPacks;
    /** Already known entity keys (a review reads them once); default: the store's `ent:` facts. */
    knownKeys?: ReadonlyMap<string, NodeRef>;
}

export interface FileExtraction {
    extracted: Extracted;
    parser: KnowledgeParser;
    pack: KnowledgePack;
    /** The parser's sections of the file (the review judges these). */
    sections: ParsedSection[];
}

/** Extraction plus what produced it; undefined when no pack's parser accepts the file. */
export async function extractFileDetailed(a: ExtractFileArgs): Promise<FileExtraction | undefined> {
    const src = asPathSource(a.doc, a.path, a.hash);
    const hit = (a.composed ?? composePacks(a.packs)).resolve(src);
    if (!hit) return undefined;
    const sections = hit.parser.parse(src);
    const ref = srcRef(a.rootId, a.path);
    const cached = await a.store.getExtracted?.(a.hash);
    // A cached entry of another extractor version is ignored: its output may differ from today's.
    const found = cached?.find((e) => e.sourceRef === ref && e.extractorVersion === EXTRACTOR_VERSION);
    if (found) return { extracted: found, parser: hit.parser, pack: hit.pack, sections };

    const ctx: ExtractContext = {
        rootId: a.rootId,
        knownKeys: a.knownKeys ?? knownKeysOfEntities((await a.store.facts({ rootId: a.rootId })).map((f) => f.key)),
        resolvePath: resolveRelativePath,
    };
    const extracted: Extracted = hit.parser.extract
        ? { ...hit.parser.extract(src, sections, ctx), sourceHash: a.hash, extractorVersion: EXTRACTOR_VERSION }
        : {
            sourceRef: ref, sourceHash: a.hash, extractorVersion: EXTRACTOR_VERSION, edges: [], keys: [{ ref, key: a.path }],
            nodes: [{ ref, kind: 'file', label: basenameOf(a.path), pack: hit.pack.id, importance: 0, origin: 'extracted', sourceHash: a.hash }],
        };
    await a.store.putExtracted?.(a.hash, [...(cached ?? []).filter((e) => e.sourceRef !== ref), extracted]);
    return { extracted, parser: hit.parser, pack: hit.pack, sections };
}

/**
 * Re-extracts ONE file — the stable entry point for a host that learns a single file changed
 * (an agent's file edit) without running a review.
 *
 * Input ({@link ExtractFileArgs}): `rootId`, the root-relative `path`, the content `hash` of
 * the text in `doc`, the `packs` and a `store`. Output: the file's {@link Extracted}, stamped
 * with {@link EXTRACTOR_VERSION}; or undefined when no pack's parser accepts the file (a
 * binary document without a host parser, say).
 *
 * Cached by content hash through `store.getExtracted` / `putExtracted` (an entry of another
 * extractor version is ignored and replaced), so calling it again for the same hash is free.
 * It writes only that cache — never knowledge — and reads the store's `ent:` facts for
 * `mentions` unless `knownKeys` is given. The signature is part of the subpath's public contract.
 */
export async function extractFile(a: ExtractFileArgs): Promise<Extracted | undefined> {
    return (await extractFileDetailed(a))?.extracted;
}
