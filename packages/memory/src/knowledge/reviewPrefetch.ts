/**
 * knowledge/reviewPrefetch.ts — the read + extract stage of a review (PRD 38 §4.3, PRD 41 A8).
 *
 * `createPrefetch` returns the per-file job `prefetchOrdered` runs ahead of the consumer: read,
 * binary check, extract (the content-hash cache is READ here; through `args.extractPool` on a miss when
 * the host supplies one). A job never throws — it settles to a {@link Prefetch}, and a failure carries
 * its reason so the review can keep the file's prior knowledge and retry it next time. It never WRITES
 * the extract cache: the review's ordered writer does, so the cache matches a sequential run. Each job
 * holds one lane (0..width-1) and emits its `extract` step the moment it settles.
 */

import type { SourceDocument } from '../ingest/types.js';
import type { KnowledgePack, KnowledgeStore, NodeRef, ReviewArgs } from './model.js';
import { extractFileDetailed, type FileExtraction } from './extractFile.js';
import { BinaryContent, CHANGED_SINCE_PLAN, ExtractFailed } from './extractJob.js';
import { judgesPack } from './reviewJudge.js';
import { createDeadlines } from './jobDeadline.js';
import { createModelLimiter } from './modelLimiter.js';
import { Lanes, reasonOf, type ReviewStep } from './reviewRunModel.js';
import { isBinaryText, type ComposedPacks } from './sources.js';

/** What the prefetch stage learned about one file. `failed` and `read-error` carry why. */
export type Prefetch = (
    | { kind: 'read-error'; reason: string }
    | { kind: 'binary' }
    /** Planned as an asset (binary by extension): decided without a read, the review writes its file node. */
    | { kind: 'asset' }
    | { kind: 'none' }
    | { kind: 'extracted'; doc?: SourceDocument; extraction: FileExtraction }
    | { kind: 'failed'; reason: string; transient?: true }
) & { ms: number; lane: number; parts?: NonNullable<ReviewStep['parts']> };

export interface PrefetchContext {
    args: ReviewArgs;
    /** The run's signal (the caller's, plus the run's own stop on a thrown error). */
    signal: AbortSignal;
    rootId: string;
    packs: readonly KnowledgePack[];
    store: KnowledgeStore;
    composed: ComposedPacks;
    knownKeys: ReadonlyMap<string, NodeRef>;
    /** Lanes = the read-ahead width: files read and extracted ahead of the writer. */
    width: number;
    /** Files in the extract stage at once (the pool's width); default `width`. */
    extractWidth?: number;
    /** The run's extractor key (the plan's `extractor.extractorKeyNow`), passed to every extraction. */
    extractorKey?: string;
}

/** A {@link Prefetch} before its timing is stamped. */
type Outcome = Prefetch extends infer P ? (P extends unknown ? Omit<P, 'ms' | 'lane'> : never) : never;

/** The extract step event of a settled job. */
export function extractStepOf(path: string, job: Prefetch): ReviewStep {
    const base = { path, phase: 'extract' as const, ms: job.ms, lane: job.lane, ...(job.parts && Object.keys(job.parts).length > 0 ? { parts: job.parts } : {}) };
    switch (job.kind) {
        case 'read-error': return { ...base, outcome: 'failed', reason: `read: ${job.reason}` };
        case 'failed': return { ...base, outcome: 'failed', reason: job.reason };
        case 'binary': return { ...base, outcome: 'skipped', reason: 'binary' };
        case 'asset': return { ...base, outcome: 'ok', reason: 'asset' }; // its node is written: the file is stored, just not read
        case 'none': return { ...base, outcome: 'skipped', reason: 'no-pack' };
        case 'extracted': return { ...base, outcome: job.extraction.cached ? 'cached' : 'ok' };
    }
}

export { CHANGED_SINCE_PLAN };

/** Per-job ceilings of the extract path (PRD 41 A10 rule 2). */
export const DEFAULT_READ_TIMEOUT_MS = 5_000;
export const DEFAULT_EXTRACT_TIMEOUT_MS = 10_000;

/** A changed file as the prefetch sees it: a `PlannedFile`'s path, hash, size, asset flag and the source's own spelling (reads use it). */
export type PrefetchFile = { path: string; hash: string; size: number; asset?: true; listedPath?: string };

export function createPrefetch(c: PrefetchContext): (f: PrefetchFile) => Promise<Prefetch> {
    // A lane is a worker: taken while the file holds an extract slot. The extract stage holds at most `extractWidth` at once, so the
    // extract deadline counts from the moment a job holds a slot, never from its wait in line (the limiter is used as a plain semaphore).
    const lanes = new Lanes(c.extractWidth ?? c.width);
    const extractSlots = createModelLimiter(c.extractWidth ?? c.width);
    const withDeadline = createDeadlines(c.signal);
    const defaultReadMs = c.args.timeouts?.readMs ?? DEFAULT_READ_TIMEOUT_MS;
    /** The source's own read deadline for `path`, when it sets a usable one (finite, above 0). */
    const readOverrideOf = (path: string): number | undefined => {
        const ms = c.args.source.readTimeoutMs?.(path);
        return typeof ms === 'number' && Number.isFinite(ms) && ms > 0 ? ms : undefined;
    };
    const readMsOf = (path: string): number => readOverrideOf(path) ?? defaultReadMs;
    const extractMs = c.args.timeouts?.extractMs ?? DEFAULT_EXTRACT_TIMEOUT_MS;
    // Judging is a later pass in a mechanical review: no sections are wanted from the pool.
    const wantsSections = (pack: KnowledgePack) => !c.args.mechanical && judgesPack(c.args.judgement, pack);
    const byPool = !!c.args.extractPool && !!c.args.deferRead;

    const run = async (f: PrefetchFile, parts: NonNullable<ReviewStep['parts']>, held: { lane: number }): Promise<Outcome> => {
        // A stopped run starts nothing: a job that has not begun settles at once. Once the read has begun the job finishes (its slot wait below is signal-free for a file already read).
        if (c.signal.aborted) return { kind: 'failed', reason: reasonOf(c.signal.reason), transient: true };
        // The plan decided it (isAssetPath): no read, no extract slot.
        if (f.asset) return { kind: 'asset' };
        let doc: SourceDocument | undefined;
        const t0 = Date.now();
        // The source's own spelling opens the file; `f.path` (normalized) is its identity.
        const readPath = f.listedPath ?? f.path;
        if (!byPool) {
            try { doc = await withDeadline(readMsOf(readPath), (signal) => c.args.source.read(readPath, signal)); } catch (error) {
                return { kind: 'read-error', reason: reasonOf(error) };
            }
            parts.readMs = Date.now() - t0;
        }
        try {
            if (doc && doc.content.kind === 'text' && isBinaryText(doc.content.text)) return { kind: 'binary' };
            const extraction = await extractSlots.run(async () => {
                held.lane = lanes.take();
                try { return await withDeadline(byPool ? extractMs + (readOverrideOf(readPath) ?? 0) : extractMs,(signal) => extractFileDetailed({
                rootId: c.rootId, path: f.path, hash: f.hash, ...(doc ? { doc } : {}), packs: c.packs, store: c.store, composed: c.composed,
                knownKeys: c.knownKeys, writeCache: false,
                ...(c.extractorKey ? { extractorKey: c.extractorKey } : {}),
                onTiming: (t) => { parts.lookupMs = t.lookupMs; parts.poolMs = t.poolMs; },
                ...(c.args.extractPool ? { pool: c.args.extractPool, signal, wantsSections, ...(byPool ? {
                    readByPool: true, ...(f.listedPath !== undefined ? { readPath } : {}), ...(readOverrideOf(readPath) ? { readMs: readOverrideOf(readPath)! } : {}),
                } : {}) } : {}),
            })); } finally { lanes.free(held.lane); }
            }, byPool ? c.signal : undefined);
            if (!extraction) return { kind: 'none' };
            // Binary and no-pack files are decided above and never hash-checked (as in the host worker). For a file that WILL be
            // stored, the bytes read must be the bytes planned: hashed once, from the text already in hand. The review writes the
            // extract cache only for an 'extracted' job, so nothing is stored under the stale hash.
            const hashOf = c.args.source.hashOf;
            if (hashOf && doc && doc.content.kind === 'text' && hashOf.call(c.args.source, doc.content.text, readPath) !== f.hash) return { kind: 'failed', reason: CHANGED_SINCE_PLAN, transient: true };
            return { kind: 'extracted', ...(doc ? { doc } : {}), extraction };
        } catch (error) {
            if (error instanceof BinaryContent) return { kind: 'binary' };
            // Transient UNLESS proven final: only the parser or extractor throwing on these bytes (ExtractFailed) is. A read, a timeout, a crashed or
            // disposed worker, a failing store, a stop: all say nothing final about the file.
            return { kind: 'failed', reason: reasonOf(error), ...((error instanceof ExtractFailed || (error as { name?: unknown } | null)?.name === 'ExtractFailed') && !c.signal.aborted ? {} : { transient: true as const }) };
        }
    };

    return async (f) => {
        const t0 = Date.now();
        const parts: NonNullable<ReviewStep['parts']> = {};
        const held = { lane: 0 };
        const out = await run(f, parts, held);
        const job = { ...out, ms: Date.now() - t0, lane: held.lane, parts } as Prefetch;
        // A job that ended only because the run was stopped is not a failed file: one Stop must not emit hundreds of failed steps. A job that finished normally still reports.
        if (!(c.signal.aborted && job.kind === 'failed')) c.args.onStep?.(extractStepOf(f.path, job));
        return job;
    };
}
