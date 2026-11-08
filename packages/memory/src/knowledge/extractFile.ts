/**
 * knowledge/extractFile.ts — the ONE single-file extraction (PRD 38 §4.3): compose the
 * packs, resolve the parser, parse, extract with the context (known entity keys,
 * `resolvePath`), stamp {@link EXTRACTOR_VERSION} and read/write the extracted cache by
 * content hash. `reviewDirectory` runs it per changed file; a host re-extracting one edited
 * file calls {@link extractFile}.
 */

import type { ParsedSection, SourceDocument } from '../ingest/types.js';
import type { Extracted, KnowledgePack, KnowledgeParser, KnowledgeStore, NodeRef } from './model.js';
import { BinaryContent, CHANGED_SINCE_PLAN, ExtractFailed, jobOf, runExtractor, type ExtractPool } from './extractJob.js';
import { srcRef } from './refs.js';
import { entityName, knownKeysOf, resolveEntities } from './resolve.js';
import { asPathSource, composePacks, EXTRACTOR_VERSION, isAssetExtract, resolveRelativePath, stubSource, type ComposedPacks } from './sources.js';

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
    /** The document as read; absent with `readByPool` (the pool reads the file itself). */
    doc?: SourceDocument;
    packs: readonly KnowledgePack[];
    store: KnowledgeStore;
    /** Already composed from `packs` (a review composes once). */
    composed?: ComposedPacks;
    /** Already known entity keys (a review reads them once); default: the store's `ent:` facts. */
    knownKeys?: ReadonlyMap<string, NodeRef>;
    /**
     * A host's extraction pool (PRD 41 A8): a cache miss is extracted by `pool(job, signal)` instead of in
     * this thread, which then never parses. The extract cache is still read and written here.
     */
    pool?: ExtractPool;
    signal?: AbortSignal;
    /** With a `pool`: whether the file's pack judges it, i.e. the worker must return its sections (default: never). */
    wantsSections?: (pack: KnowledgePack) => boolean;
    /** The pool reads the file itself: no `doc` (see `ExtractJob.readByPool`). */
    readByPool?: boolean;
    /** With `readByPool`: the read deadline for this path (`DirectorySource.readTimeoutMs`), handed to the worker. */
    readMs?: number;
    /** Told where the time went: the cache lookup and the pool job (PRD 41 A10, so a stall names itself). */
    onTiming?: (t: { lookupMs: number; poolMs: number }) => void;
    /** Leave the extract cache write to the caller (the review writes it in plan order); default: write it. */
    writeCache?: boolean;
}

export interface FileExtraction {
    extracted: Extracted;
    parser: KnowledgeParser;
    pack: KnowledgePack;
    /** The parser's sections of the file (the review judges these). */
    sections: ParsedSection[];
    /** The extract came from the content-hash cache — nothing was extracted. */
    cached: boolean;
}

/** How the file reaches the pool: its document, or `readByPool` when the pool reads it. */
function jobSource(a: ExtractFileArgs): { doc: SourceDocument; readByPool?: false } | { doc?: undefined; readByPool: true } {
    return a.readByPool || !a.doc ? { readByPool: true } : { doc: a.doc };
}

/** Adds `extracted` to the content-hash cache, keeping the entries of the other files that share the hash. */
export async function putExtract(store: KnowledgeStore, hash: string, extracted: Extracted, prior?: readonly Extracted[]): Promise<void> {
    const cached = prior ?? (await store.getExtracted?.(hash)) ?? [];
    await store.putExtracted?.(hash, [...cached.filter((e) => e.sourceRef !== extracted.sourceRef), extracted]);
}

/**
 * Records, once, that nothing is extracted for `hash` (a binary file): the cache holds an explicit `[]`, which a store keeps
 * distinct from "absent" (a text file not extracted yet). A hash that already has entries is left alone.
 */
export async function putNoExtract(store: KnowledgeStore, hash: string): Promise<void> {
    if (!store.putExtracted || !store.getExtracted || (await store.getExtracted(hash)) !== undefined) return;
    await store.putExtracted(hash, []);
}

/** Extraction plus what produced it; undefined when no pack's parser accepts the file. */
export async function extractFileDetailed(a: ExtractFileArgs): Promise<FileExtraction | undefined> {
    // Pack resolution needs only the path and media type, so a file the pool reads is resolved from a content-free key.
    const src = asPathSource(a.doc ?? stubSource(a.path), a.path, a.hash);
    const hit = (a.composed ?? composePacks(a.packs)).resolve(src);
    if (!hit) return undefined;
    if (!a.doc && !a.pool) throw new Error(`extractFile: ${a.path} has no document and no pool to read it`);
    const ref = srcRef(a.rootId, a.path);
    // With a pool the main thread never parses: the worker returns the sections when the file will be judged.
    const wants = !!a.pool && !!a.wantsSections?.(hit.pack);
    // The parser and the extractor run in this thread only without a pool; their errors are the PROVEN-final failures (ExtractFailed).
    const proven = <T>(work: () => T): T => { try { return work(); } catch (error) { throw new ExtractFailed(error instanceof Error ? error.message : String(error), { cause: error }); } };
    let sections: ParsedSection[] = a.pool ? [] : proven(() => hit.parser.parse(src));
    const tLookup = Date.now();
    const cached = await a.store.getExtracted?.(a.hash);
    const lookupMs = Date.now() - tLookup;
    // A cached entry of another extractor version is ignored: its output may differ from today's. So is an asset's entry
    // (a document stored as an asset before a host parser claimed it): it holds no extraction.
    const found = cached?.find((e) => e.sourceRef === ref && e.extractorVersion === EXTRACTOR_VERSION && !isAssetExtract(e));
    if (found) {
        const tSections = Date.now();
        if (a.pool && wants) {
            const out = await a.pool(jobOf({ rootId: a.rootId, path: a.path, hash: a.hash, knownKeys: new Map(), sectionsOnly: true, ...(a.readMs ? { readMs: a.readMs } : {}), ...jobSource(a) }), a.signal);
            if (out.binary) throw new BinaryContent();
            if (!out.sections) throw new Error(`extract pool returned no sections for ${a.path}`);
            sections = out.sections;
        }
        a.onTiming?.({ lookupMs, poolMs: Date.now() - tSections });
        return { extracted: found, parser: hit.parser, pack: hit.pack, sections, cached: true };
    }

    const knownKeys = a.knownKeys ?? knownKeysOfEntities((await a.store.facts({ rootId: a.rootId })).map((f) => f.key));
    let extracted: Extracted;
    if (a.pool) {
        const tPool = Date.now();
        const out = await a.pool(jobOf({ rootId: a.rootId, path: a.path, hash: a.hash, knownKeys, needSections: wants, ...(a.readMs ? { readMs: a.readMs } : {}), ...jobSource(a) }), a.signal);
        a.onTiming?.({ lookupMs, poolMs: Date.now() - tPool });
        if (out.binary) throw new BinaryContent();
        // Exact match only: a result for another file, or none although the packs accept this one, is a failure.
        const mine = out.extracted.find((e) => e.sourceRef === ref);
        if (!mine) throw new Error(`extract pool returned no extract for ${a.path}`);
        // The pool read other bytes than the plan hashed: nothing is stored under the planned hash.
        if (mine.sourceHash !== a.hash) throw new Error(CHANGED_SINCE_PLAN); // transient: the bytes may settle
        if (wants) {
            if (!out.sections) throw new Error(`extract pool returned no sections for ${a.path}`);
            sections = out.sections;
        }
        extracted = mine;
    } else {
        extracted = proven(() => runExtractor(hit, src, sections, { rootId: a.rootId, knownKeys, resolvePath: resolveRelativePath }, a));
    }
    if (a.writeCache !== false) await putExtract(a.store, a.hash, extracted, cached ?? []);
    return { extracted, parser: hit.parser, pack: hit.pack, sections, cached: false };
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
