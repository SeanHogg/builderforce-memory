/**
 * knowledge/extractJob.ts — the pure, serializable unit of extraction (PRD 41 A8).
 *
 * An {@link ExtractJob} is plain data (no functions, no store), so it can cross a
 * `worker_threads` boundary: a host's pool posts the job to a worker, the worker calls
 * {@link extractJob} with the packs it loaded itself, and the {@link Extracted} comes back.
 * The in-thread default of `reviewDirectory` runs the very same {@link runExtractor}, so a
 * review with and without `ReviewArgs.extractPool` extracts the same bytes. The extract
 * CACHE (store reads/writes) never crosses the boundary: the review consults it around the pool.
 */

import type { ParsedSection, SourceDocument } from '../ingest/types.js';
import type { ExtractContext, Extracted, KnowledgePack, KnowledgeParser, NodeRef } from './model.js';
import { basenameOf, srcRef } from './refs.js';
import { asPathSource, composePacks, EXTRACTOR_VERSION, resolveRelativePath, type ComposedPacks } from './sources.js';

/** Everything the extractor needs for one file, as structured-cloneable data. */
export interface ExtractJob {
    rootId: string;
    /** Root-relative path. */
    path: string;
    /** Content hash of the file. */
    hash: string;
    /** The document as read (plain data). */
    doc: SourceDocument;
    /** The known entity keys (`mentions`), as entries of the review's map. */
    knownKeys: ReadonlyArray<readonly [string, NodeRef]>;
    /** The file will be judged: return its parsed sections too (the main thread never parses when a pool is supplied). */
    needSections?: boolean;
    /**
     * The pool reads the file itself (PRD 41 A10 rule 8): `doc` is a content-free stand-in that only carries the path
     * and media type; the worker reads `path`, runs the binary check and extracts. The main thread neither reads nor copies the text.
     */
    readByPool?: true;
    /** The extract is already cached and only the sections are wanted: skip extracting. Implies `needSections`. */
    sectionsOnly?: boolean;
}

/** Thrown by the extract stage when the pool found the file binary; the review records a `skipped` step. */
export class BinaryContent extends Error {
    constructor() { super('binary'); this.name = 'BinaryContent'; }
}

/** What a worker returns: the file's extract(s) — `[]` when no pack accepts it — and, if asked, its sections. */
export interface ExtractJobResult {
    /** The pool read the file itself (see `ExtractJob.readByPool`) and found it binary: nothing was extracted. */
    binary?: true;
    extracted: Extracted[];
    sections?: ParsedSection[];
}

/**
 * A host's extraction pool: called for a changed file whose extract is not cached (or, with
 * `sectionsOnly`, whose sections are needed). `signal` is the review's: a stopped review should
 * terminate in-flight worker jobs.
 */
export type ExtractPool = (job: ExtractJob, signal?: AbortSignal) => Promise<ExtractJobResult>;

/** The ONE extraction of an already-resolved source: stamps {@link EXTRACTOR_VERSION}. */
export function runExtractor(
    hit: { parser: KnowledgeParser; pack: KnowledgePack },
    src: SourceDocument,
    sections: ParsedSection[],
    ctx: ExtractContext,
    file: { path: string; hash: string },
): Extracted {
    const ref = srcRef(ctx.rootId, file.path);
    return hit.parser.extract
        ? { ...hit.parser.extract(src, sections, ctx), sourceHash: file.hash, extractorVersion: EXTRACTOR_VERSION }
        : {
            sourceRef: ref, sourceHash: file.hash, extractorVersion: EXTRACTOR_VERSION, edges: [], keys: [{ ref, key: file.path }],
            nodes: [{ ref, kind: 'file', label: basenameOf(file.path), pack: hit.pack.id, importance: 0, origin: 'extracted', sourceHash: file.hash }],
        };
}

/** The job as a serializable value for `doc` (what a pool receives). */
export function jobOf(
    f: {
        rootId: string; path: string; hash: string; doc: SourceDocument; knownKeys: ReadonlyMap<string, NodeRef>;
        needSections?: boolean; sectionsOnly?: boolean; readByPool?: boolean;
    },
): ExtractJob {
    return {
        rootId: f.rootId, path: f.path, hash: f.hash, doc: f.doc, knownKeys: [...f.knownKeys],
        ...(f.needSections ? { needSections: true } : {}), ...(f.sectionsOnly ? { sectionsOnly: true } : {}), ...(f.readByPool ? { readByPool: true as const } : {}),
    };
}

/**
 * Extracts one file from a job — pure: no store, no clock. `extracted` is `[]` when no pack's parser
 * accepts the file. `packs` is the worker's own pack list (or its composition, to compose once).
 */
export function extractJob(job: ExtractJob, packs: readonly KnowledgePack[] | ComposedPacks): ExtractJobResult {
    const composed = Array.isArray(packs) ? composePacks(packs as readonly KnowledgePack[]) : (packs as ComposedPacks);
    const src = asPathSource(job.doc, job.path, job.hash);
    const hit = composed.resolve(src);
    if (!hit) return { extracted: [] };
    const sections = hit.parser.parse(src);
    const wantSections = job.needSections || job.sectionsOnly;
    if (job.sectionsOnly) return { extracted: [], sections };
    const ctx: ExtractContext = { rootId: job.rootId, knownKeys: new Map(job.knownKeys), resolvePath: resolveRelativePath };
    return { extracted: [runExtractor(hit, src, sections, ctx, job)], ...(wantSections ? { sections } : {}) };
}
