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
import { BinaryContent } from './extractJob.js';
import { judgesPack } from './reviewJudge.js';
import { JobTimeout, withDeadline } from './jobDeadline.js';
import { createModelLimiter } from './modelLimiter.js';
import { Lanes, reasonOf, type ReviewStep } from './reviewRunModel.js';
import { isBinaryText, stubSource, type ComposedPacks } from './sources.js';

/** What the prefetch stage learned about one file. `failed` and `read-error` carry why. */
export type Prefetch = (
    | { kind: 'read-error'; reason: string }
    | { kind: 'binary' }
    | { kind: 'none' }
    | { kind: 'extracted'; doc: SourceDocument; extraction: FileExtraction }
    | { kind: 'failed'; reason: string }
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
        case 'none': return { ...base, outcome: 'skipped', reason: 'no-pack' };
        case 'extracted': return { ...base, outcome: job.extraction.cached ? 'cached' : 'ok' };
    }
}

/** Per-job ceilings of the extract path (PRD 41 A10 rule 2). */
export const DEFAULT_READ_TIMEOUT_MS = 5_000;
export const DEFAULT_EXTRACT_TIMEOUT_MS = 10_000;

export function createPrefetch(c: PrefetchContext): (f: { path: string; hash: string }) => Promise<Prefetch> {
    // A lane is a worker: taken while the file holds an extract slot. The extract stage holds at most `extractWidth` at once, so the
    // extract deadline counts from the moment a job holds a slot, never from its wait in line (the limiter is used as a plain semaphore).
    const lanes = new Lanes(c.extractWidth ?? c.width);
    const extractSlots = createModelLimiter(c.extractWidth ?? c.width);
    const readMs = c.args.timeouts?.readMs ?? DEFAULT_READ_TIMEOUT_MS;
    const extractMs = c.args.timeouts?.extractMs ?? DEFAULT_EXTRACT_TIMEOUT_MS;
    // Judging is a later pass in a mechanical review: no sections are wanted from the pool.
    const wantsSections = (pack: KnowledgePack) => !c.args.mechanical && judgesPack(c.args.judgement, pack);
    const byPool = !!c.args.extractPool && !!c.args.deferRead;

    const run = async (f: { path: string; hash: string }, parts: NonNullable<ReviewStep['parts']>, held: { lane: number }): Promise<Outcome> => {
        let doc: SourceDocument;
        const t0 = Date.now();
        if (byPool) doc = stubSource(f.path);
        else {
            try { doc = await withDeadline(readMs, () => c.args.source.read(f.path), c.signal); } catch (error) {
                return { kind: 'read-error', reason: reasonOf(error) };
            }
            parts.readMs = Date.now() - t0;
        }
        try {
            if (doc.content.kind === 'text' && !byPool && isBinaryText(doc.content.text)) return { kind: 'binary' };
            const extraction = await extractSlots.run(async () => {
                held.lane = lanes.take();
                try { return await withDeadline(extractMs, (signal) => extractFileDetailed({
                rootId: c.rootId, path: f.path, hash: f.hash, doc, packs: c.packs, store: c.store, composed: c.composed,
                knownKeys: c.knownKeys, writeCache: false,
                onTiming: (t) => { parts.lookupMs = t.lookupMs; parts.poolMs = t.poolMs; },
                ...(c.args.extractPool ? { pool: c.args.extractPool, signal, wantsSections, ...(byPool ? { readByPool: true } : {}) } : {}),
            }), c.signal); } finally { lanes.free(held.lane); }
            }, c.signal);
            return extraction ? { kind: 'extracted', doc, extraction } : { kind: 'none' };
        } catch (error) {
            if (error instanceof BinaryContent) return { kind: 'binary' };
            return { kind: 'failed', reason: reasonOf(error) };
        }
    };

    return async (f) => {
        const t0 = Date.now();
        const parts: NonNullable<ReviewStep['parts']> = {};
        const held = { lane: 0 };
        const out = await run(f, parts, held);
        const job = { ...out, ms: Date.now() - t0, lane: held.lane, parts } as Prefetch;
        c.args.onStep?.(extractStepOf(f.path, job));
        return job;
    };
}
