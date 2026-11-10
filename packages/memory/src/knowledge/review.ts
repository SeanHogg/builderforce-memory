/**
 * knowledge/review.ts — `reviewDirectory`: point it at a folder and it learns what the
 * folder says (PRD 38 §4.3).
 *
 *   plan (walk + manifest diff) → per changed file: read · parse (the ONE ParserRegistry,
 *   composed from the packs) · extract (cached by content hash) · judge (changed sections
 *   of judged kinds, within budget) · write · markStale → retire · analyse (analyzers, own `analyse`
 *   progress phase) → compare near-duplicate
 *   claims → manifest + scope digest. Claims whose compare pairs were skipped wait in the
 *   compare manifest for *Continue*.
 *   Read + extract are prefetched with bounded concurrency (`args.concurrency`; through `args.extractPool`
 *   when a host supplies one) and up to `args.judgeConcurrency` files are judged at once, but ONE writer
 *   applies judge · write · markStale strictly in file order, and each file's model calls are reserved from
 *   the shared budget in file order — so the stored bytes and the budget spend do not depend on the widths
 *   (PRD 41 A8). A file whose read, extraction or judging fails is recorded in `report.failures` and skipped;
 *   only an abort or a local store write error ends the run.
 *
 * Store judgement, compute perception: only `model` facts and relations are written —
 * `extracted` output goes to the extracted cache, never to memory ({@link assertStoredBatch}).
 * A file whose read throws is skipped, never retired: its knowledge and manifest hash stay
 * as they were (`readErrors`); only a file the fresh plan no longer lists is removed.
 * Abort is honoured between files and inside judgement; a file is written whole or not at
 * all. A file whose sections were cut by the budget stays "changed" in the manifest, so
 * *Continue* (the confirmed plan passed back as `args.plan`, no re-walk) judges only the
 * sections not judged yet. A file owed under an extractor upgrade that this run left
 * unfinished (a failed read, an abort, the budget) stays owed — see owedFiles.ts — so the
 * next review or *Continue* picks up exactly those files, whatever their content does.
 *
 * The stages are one module each: reviewContext.ts (planning, the shared state, write and retire), reviewPrefetch.ts (read +
 * extract), reviewJudge.ts (judging), reviewApply.ts (the ordered writer), reviewAnalyse.ts (analyzers), reviewFinish.ts
 * (compare, durability, manifests). This file wires the read-ahead loop and the run's stop between them.
 */

import type { ReviewArgs, ReviewReport } from './model.js';
import { prefetchOrdered } from './orderedPool.js';
import { DEFAULT_REVIEW_CONCURRENCY, DEFAULT_REVIEW_JUDGE_CONCURRENCY, JUDGE_CONCURRENCY } from './judge.js';
import { createModelLimiter } from './modelLimiter.js';
import { startJudge } from './reviewJudge.js';
import { createPrefetch } from './reviewPrefetch.js';
import { Lanes } from './reviewRunModel.js';
import { openReview, retire, type ReviewContext } from './reviewContext.js';
import { applyFile, type ApplyContext, type Slot, type WriterGate } from './reviewApply.js';
import { runAnalyzers } from './reviewAnalyse.js';
import { finishReview } from './reviewFinish.js';

export { assertStoredBatch } from './reviewContext.js';

/** Read-ahead window for a MECHANICAL review with a pool that reads files itself: wide enough to buffer finished results behind one slow head file. Real extraction width is bounded separately (the extract-slot limiter in reviewPrefetch). */
const POOL_READ_AHEAD_DEFAULT = 256;
const POOL_READ_AHEAD_MAX = 512;

/** Files read and extracted ahead of the ordered writer. */
function readAheadOf(args: ReviewArgs, concurrency: number): number {
    if (!args.extractPool) return concurrency;
    // Wide only for a mechanical pool-reads run: each settled prefetch then holds just the small extracted output.
    // A judging review keeps each settled file's parsed sections in memory, so it keeps the narrow window.
    if (args.deferRead && args.mechanical) {
        const want = Math.floor(args.readAhead ?? Math.max(POOL_READ_AHEAD_DEFAULT, concurrency * 32)) || concurrency;
        return Math.min(POOL_READ_AHEAD_MAX, Math.max(concurrency, want));
    }
    return Math.min(64, Math.max(concurrency, Math.floor(args.readAhead ?? Math.max(32, concurrency * 4)) || concurrency));
}

/**
 * A store that can hold persistence (`batched`) gets the whole review as one batch, so a snapshot
 * store writes a bounded number of times instead of once per file; any other store is unaffected.
 */
export async function reviewDirectory(args: ReviewArgs): Promise<ReviewReport> {
    return args.store.batched ? args.store.batched((flush) => reviewRun(args, flush)) : reviewRun(args, async () => {});
}

async function reviewRun(args: ReviewArgs, flush: () => Promise<void>): Promise<ReviewReport> {
    const c = await openReview(args);
    const { report } = c;
    const runSignal = await applyChangedFiles(c);
    // A Stop ends the run: the file in hand was finished and applied, but nothing after it (retire, analyzers, the digest stamp) runs.
    report.aborted ||= runSignal.aborted;

    // Pass 2 (`only`) is judging alone: the map pass already retired the removed files and ran the analyzers.
    if (!report.aborted && !args.only) {
        // The last file is written: leave "write" now, the analyzers below can take minutes.
        // Opened as 0/1 (not 0/0, which would close the row at once) so the row stays active until the analyzers close it.
        c.progress({ phase: 'analyse', done: 0, total: 1 });
        await retire(c, c.detail.plan.removed);
        await runAnalyzers(c);
        // A Stop during the analyzers ends the run like one during the write loop (`runSignal` no longer follows `signal` here).
        report.aborted ||= runSignal.aborted || !!args.signal?.aborted;
        if (!report.aborted) c.progress({ phase: 'resolve', done: 0, total: c.newClaims.size });
    }
    return finishReview(c, flush);
}

/**
 * Read + binary check + extract run ahead with bounded concurrency (extraction only reads `knownKeys`, fixed at open, and the
 * content-hash extract cache); with `args.extractPool` the width is the pool's. Judging runs `judgeWidth` files at once; ONE
 * writer ({@link applyFile}) consumes them strictly in plan order. Returns the run's own stop signal.
 */
async function applyChangedFiles(c: ReviewContext): Promise<AbortSignal> {
    const { args, report } = c;
    const { signal } = args;
    const concurrency = Math.min(32, Math.max(1, Math.floor(args.concurrency ?? DEFAULT_REVIEW_CONCURRENCY) || 1));
    // Read-ahead is wider than the pool (PRD 41 A10 rule 8) so workers never wait for the ordered writer.
    const readAhead = readAheadOf(args, concurrency);
    const judgeWidth = Math.min(concurrency, Math.min(16, Math.max(1, Math.floor(args.judgeConcurrency ?? DEFAULT_REVIEW_JUDGE_CONCURRENCY) || 1)));
    // The run's own stop: the caller's abort relays into it, and a thrown error (a local store failure) aborts it,
    // so queued and in-flight judge calls and pool jobs stop and nothing outlives the run.
    const stopper = new AbortController();
    // An abort that lands while the writer is mid-file ends the run at that file's boundary, as a sequential run
    // does (`cut`); one that lands anywhere else (a read, a model call) keeps every file whose work completed.
    const gate: WriterGate = { inWriter: false };
    let cut = false;
    const relay = () => { if (gate.inWriter) cut = true; stopper.abort(signal?.reason); };
    if (signal?.aborted) relay(); else signal?.addEventListener('abort', relay, { once: true });
    const runSignal = stopper.signal;
    const prefetch = createPrefetch({
        args, signal: runSignal, rootId: c.rootId, packs: args.packs, store: args.store, composed: c.composed, knownKeys: c.knownKeys,
        width: readAhead, extractWidth: concurrency,
        ...(c.detail.extractor.extractorKeyNow ? { extractorKey: c.detail.extractor.extractorKeyNow } : {}),
    });
    const dispatch = {
        // A mechanical run (pass 1) makes no model call: judged files stay owed.
        port: args.mechanical ? undefined : args.judgement, signal: runSignal, breaker: c.breaker, budget: c.budget, judgedPrior: c.judgedPrior,
        lanes: new Lanes(judgeWidth), limiter: createModelLimiter(args.maxModelCalls ?? JUDGE_CONCURRENCY),
    };
    const apply: ApplyContext = {
        c, gate, runSignal,
        judgeCtx: {
            batch: c.batch, entities: c.entities, stored: c.stored, judgedNow: c.judgedNow, newClaims: c.newClaims, measured: c.measured,
            detail: c.detail, report, ...(args.onStep ? { onStep: args.onStep } : {}),
        },
    };
    const queue: Slot[] = [];
    let applied = 0;
    const drain = async (keep: number) => {
        while (!cut && queue.length > keep) if (await applyFile(apply, queue.shift()!)) applied += 1;
    };
    let completed = false;
    try {
        for await (const { item: f, i, result } of prefetchOrdered(c.work, readAhead, prefetch, runSignal)) {
            if (result.status === 'rejected') throw result.reason; // a prefetch job settles to a Prefetch; this is a broken invariant
            const job = result.value;
            queue.push({ f, i, job, judge: job.kind === 'extracted' ? startJudge(dispatch, f, job.extraction) : undefined });
            await drain(judgeWidth - 1);
        }
        await drain(0);
        completed = true;
    } finally {
        if (!completed) stopper.abort(); // a thrown error stops every queued and in-flight call
        signal?.removeEventListener('abort', relay);
        await Promise.allSettled(queue.map((s) => s.judge?.result)); // nothing outlives the run
    }
    if (applied < c.work.length) report.aborted = true;
    return runSignal;
}
