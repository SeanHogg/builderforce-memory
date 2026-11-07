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
import { judgesPack } from './reviewJudge.js';
import { Lanes, reasonOf, type ReviewStep } from './reviewRunModel.js';
import { isBinaryText, type ComposedPacks } from './sources.js';

/** What the prefetch stage learned about one file. `failed` and `read-error` carry why. */
export type Prefetch = (
    | { kind: 'read-error'; reason: string }
    | { kind: 'binary' }
    | { kind: 'none' }
    | { kind: 'extracted'; doc: SourceDocument; extraction: FileExtraction }
    | { kind: 'failed'; reason: string }
) & { ms: number; lane: number };

export interface PrefetchContext {
    args: ReviewArgs;
    /** The run's signal (the caller's, plus the run's own stop on a thrown error). */
    signal: AbortSignal;
    rootId: string;
    packs: readonly KnowledgePack[];
    store: KnowledgeStore;
    composed: ComposedPacks;
    knownKeys: ReadonlyMap<string, NodeRef>;
    /** Lanes = the read-ahead width. */
    width: number;
}

/** A {@link Prefetch} before its timing is stamped. */
type Outcome = Prefetch extends infer P ? (P extends unknown ? Omit<P, 'ms' | 'lane'> : never) : never;

/** The extract step event of a settled job. */
export function extractStepOf(path: string, job: Prefetch): ReviewStep {
    const base = { path, phase: 'extract' as const, ms: job.ms, lane: job.lane };
    switch (job.kind) {
        case 'read-error': return { ...base, outcome: 'failed', reason: `read: ${job.reason}` };
        case 'failed': return { ...base, outcome: 'failed', reason: job.reason };
        case 'binary': return { ...base, outcome: 'skipped', reason: 'binary' };
        case 'none': return { ...base, outcome: 'skipped', reason: 'no-pack' };
        case 'extracted': return { ...base, outcome: job.extraction.cached ? 'cached' : 'ok' };
    }
}

export function createPrefetch(c: PrefetchContext): (f: { path: string; hash: string }) => Promise<Prefetch> {
    const lanes = new Lanes(c.width);
    const wantsSections = (pack: KnowledgePack) => judgesPack(c.args.judgement, pack);

    const run = async (f: { path: string; hash: string }): Promise<Outcome> => {
        let doc: SourceDocument;
        try { doc = await c.args.source.read(f.path); } catch (error) { return { kind: 'read-error', reason: reasonOf(error) }; }
        try {
            if (doc.content.kind === 'text' && isBinaryText(doc.content.text)) return { kind: 'binary' };
            const extraction = await extractFileDetailed({
                rootId: c.rootId, path: f.path, hash: f.hash, doc, packs: c.packs, store: c.store, composed: c.composed,
                knownKeys: c.knownKeys, writeCache: false,
                ...(c.args.extractPool ? { pool: c.args.extractPool, signal: c.signal, wantsSections } : {}),
            });
            return extraction ? { kind: 'extracted', doc, extraction } : { kind: 'none' };
        } catch (error) { return { kind: 'failed', reason: reasonOf(error) }; }
    };

    return async (f) => {
        const lane = lanes.take();
        const t0 = Date.now();
        try {
            const out = await run(f);
            const job = { ...out, ms: Date.now() - t0, lane } as Prefetch;
            c.args.onStep?.(extractStepOf(f.path, job));
            return job;
        } finally {
            lanes.free(lane);
        }
    };
}
