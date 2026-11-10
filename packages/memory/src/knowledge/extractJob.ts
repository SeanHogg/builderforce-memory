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
/** What every job carries, whichever way the file reaches the worker. */
export interface ExtractJobBase {
    rootId: string;
    /** Root-relative path. */
    path: string;
    /** Content hash of the file. */
    hash: string;
    /** The known entity keys (`mentions`), as entries of the review's map. */
    knownKeys: ReadonlyArray<readonly [string, NodeRef]>;
    /** The file will be judged: return its parsed sections too (the main thread never parses when a pool is supplied). */
    needSections?: boolean;
    /** The extract is already cached and only the sections are wanted: skip extracting. Implies `needSections`. */
    sectionsOnly?: boolean;
    /**
     * The extractor key the review extracts under (`ExtractFileArgs.extractorKey`). A worker does NOT adopt it: it extracts with
     * the inputs it holds and echoes ITS key in {@link ExtractJobResult.extractorKey}; a mismatch is not cached (the file stays owed).
     */
    extractorKey?: string;
}

/**
 * How the file reaches the worker (PRD 41 A10 rule 8): `doc` is the document the review read (plain data); `readByPool` says the
 * worker reads and binary-checks `readByPool.path` itself at the planned `hash` — no stand-in document, no text copied across the thread.
 * A pool host must open `readByPool.path`, the source's listed spelling (`PlannedFile.listedPath`), never `job.path`: `job.path` is the
 * file's identity (normalized: POSIX, NFC) and may differ from the spelling the source can open (an NFD-named file, for one).
 */
export type ExtractJobSource =
    | { doc: SourceDocument; readByPool?: never }
    | { doc?: never; readByPool: { path: string; hash: string; /** The read deadline for this path (`DirectorySource.readTimeoutMs`), when the source sets one; a worker may use it in place of its constant. */ readMs?: number } };

export type ExtractJob = ExtractJobBase & ExtractJobSource;

/** The planned bytes are not the bytes read: nothing is stored under the planned hash. Always TRANSIENT (the bytes may settle). */
export const CHANGED_SINCE_PLAN = 'changed since plan';

/**
 * A failure PROVEN final: the parser or extractor itself threw on these bytes (in-thread, or a pool's worker reporting it). Every other
 * failure (a read, a timeout, a crashed or disposed worker, a failing store, a changed-since-plan) is transient: the file may extract next time.
 */
export class ExtractFailed extends Error {
    constructor(message: string, options?: { cause?: unknown }) { super(message); this.name = 'ExtractFailed'; if (options?.cause !== undefined) (this as { cause?: unknown }).cause = options.cause; }
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
    /**
     * The extractor key the worker ACTUALLY extracted under (e.g. the digest of the path aliases it resolved with), absent when it
     * has none. The review stamps the extract with this key, and when a key was asked for and this one differs (a long-lived worker
     * still holding old aliases), the result is neither cached nor used: the file stays owed. A host pool must rebuild its
     * key-dependent inputs (and so its key) when the requested key changes.
     */
    extractorKey?: string;
}

/** The pool extracted under another extractor key than the review asked for: nothing is cached, the file stays owed. Transient. */
export const EXTRACTOR_KEY_MISMATCH = 'extractor key mismatch';

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
        rootId: string; path: string; hash: string; knownKeys: ReadonlyMap<string, NodeRef>;
        needSections?: boolean; sectionsOnly?: boolean; readMs?: number; extractorKey?: string;
    } & ({ doc: SourceDocument; readByPool?: false } | { doc?: undefined; readByPool: true; /** The path the worker opens (the source's spelling); default `path`. */ readPath?: string }),
): ExtractJob {
    return {
        rootId: f.rootId, path: f.path, hash: f.hash, knownKeys: [...f.knownKeys],
        ...(f.needSections ? { needSections: true } : {}), ...(f.sectionsOnly ? { sectionsOnly: true } : {}),
        ...(f.extractorKey ? { extractorKey: f.extractorKey } : {}),
        ...(f.readByPool ? { readByPool: { path: f.readPath ?? f.path, hash: f.hash, ...(f.readMs ? { readMs: f.readMs } : {}) } } : { doc: f.doc! }),
    } as ExtractJob;
}

/**
 * Extracts one file from a job — pure: no store, no clock. `extracted` is `[]` when no pack's parser
 * accepts the file. `packs` is the worker's own pack list (or its composition, to compose once).
 * `workerKey` is the extractor key THESE packs were built under (absent: none); it is echoed as the result's `extractorKey`,
 * never copied from the job, so a worker holding stale inputs is caught by the review rather than stamped with the new key.
 */
export function extractJob(job: ExtractJob, packs: readonly KnowledgePack[] | ComposedPacks, workerKey?: string): ExtractJobResult {
    const composed = Array.isArray(packs) ? composePacks(packs as readonly KnowledgePack[]) : (packs as ComposedPacks);
    // A pure job has no filesystem: a `readByPool` job is for a worker that reads (the host's handler), not for this function.
    if (!job.doc) throw new Error(`extract job for ${job.path} carries no document (readByPool needs a reading worker)`);
    const echo = workerKey ? { extractorKey: workerKey } : {};
    const src = asPathSource(job.doc, job.path, job.hash);
    const hit = composed.resolve(src);
    if (!hit) return { extracted: [], ...echo };
    const sections = hit.parser.parse(src);
    const wantSections = job.needSections || job.sectionsOnly;
    if (job.sectionsOnly) return { extracted: [], sections, ...echo };
    const ctx: ExtractContext = { rootId: job.rootId, knownKeys: new Map(job.knownKeys), resolvePath: resolveRelativePath };
    return { extracted: [runExtractor(hit, src, sections, ctx, job)], ...(wantSections ? { sections } : {}), ...echo };
}
