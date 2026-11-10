/**
 * knowledge/reviewApply.ts — the ordered writer of a review (PRD 41 A8): applies ONE settled file, in plan order — its failure,
 * its asset node, its "nothing to extract", or its extract cache entry, judgement, write and markStale. Moved out of review.ts.
 */

import type { ReviewContext } from './reviewContext.js';
import { writeBatch } from './reviewContext.js';
import { owedCalls } from './owedFiles.js';
import { putExtract, putNoExtract } from './extractFile.js';
import { recordFailure } from './reviewFailure.js';
import { applyJudge, judgesPack, splitSections, type JudgeJob } from './reviewJudge.js';
import type { Prefetch, PrefetchFile } from './reviewPrefetch.js';
import { putAsset } from './assetFile.js';
import { srcRef } from './refs.js';

/** One changed file between dispatch and the ordered writer: what its prefetch found and its judging in flight. */
export interface Slot {
    f: PrefetchFile;
    i: number;
    job: Prefetch;
    judge: JudgeJob | undefined;
}

/** Whether the writer is mid-file (an abort then cuts the run at this file's boundary), shared with the run's abort relay. */
export interface WriterGate { inWriter: boolean }

export interface ApplyContext {
    c: ReviewContext;
    gate: WriterGate;
    /** The run's own stop signal. */
    runSignal: AbortSignal;
    /** What `applyJudge` merges into. */
    judgeCtx: Parameters<typeof applyJudge>[0];
}

/**
 * Applies one file, in plan order. False when its work was cut short by an abort (extracted, judging
 * unfinished): nothing of it is applied and it stays as it was, owed. A file whose work is complete is
 * applied even after an abort — the work is done, so it is kept.
 */
export async function applyFile(a: ApplyContext, slot: Slot): Promise<boolean> {
    a.gate.inWriter = true;
    try { return await applyInner(a, slot); } finally { a.gate.inWriter = false; }
}

async function applyInner(a: ApplyContext, { f, i, job, judge }: Slot): Promise<boolean> {
    const { c } = a;
    const { store } = c.args;
    c.progress({ phase: 'extract', done: i, total: c.work.length, path: f.path });
    // A read that throws (permissions, a lock, a race with a delete) or an extraction that throws says nothing
    // about the file: skipped with its knowledge and manifest hash intact, tried again next review.
    if (job.kind === 'read-error' || job.kind === 'failed') {
        if (job.kind === 'read-error') c.report.readErrors += 1; else c.report.extractErrors += 1;
        c.report.failures.push({ path: f.path, phase: 'extract', reason: job.kind === 'read-error' ? `read: ${job.reason}` : job.reason });
        c.measured.set(f.path, owedCalls(c.detail.extractor, f.path)); // learns nothing: keep what it was owed
        // The ONE failure path (reviewFailure.ts): a transient failure keeps the re-check request and makes the file owed; only a proven-final
        // one (or the third in a row at the same bytes) is answered with a terminal `[]`.
        const ctx = { store, detail: c.detail, stopped: () => a.runSignal.aborted, unreadable: c.unreadable, failures: c.failures, settledRecheck: c.settledRecheck };
        if (await recordFailure(ctx, f, job)) c.finished.add(f.path); // final: settled, not re-read while its bytes stand
        return true;
    }
    c.failures.delete(f.path); // it read and extracted (or was decided): the run of failures at this hash ends
    // An asset (planned binary, or a text-extension file found binary on read): never extracted, but it IS a file of the root, so
    // its file node is stored (assetFile.ts) — a replica tells it from "not extracted yet", and references to its path resolve.
    // Recorded, so a file found binary on read is not re-read by every later review.
    if (job.kind === 'binary' || job.kind === 'asset') {
        c.filesNow.set(f.path, f.hash);
        c.finished.add(f.path);
        await putAsset(store, c.rootId, f);
        await c.settledRecheck(f.hash);
        await store.markStale(c.scope, srcRef(c.rootId, f.path), f.hash);
        c.report.staleMarked += 1;
        return true;
    }
    if (job.kind === 'none') {
        // No pack accepts the file: nothing will ever be extracted for it, so say so (and answer any re-check of it).
        c.finished.add(f.path);
        await putNoExtract(store, f.hash);
        await c.settledRecheck(f.hash);
        return true;
    }
    c.report.extracted += 1;
    // The extract cache is written HERE, in plan order, so the cache, the counts and the order match a sequential run.
    if (!job.extraction.cached) await putExtract(store, f.hash, job.extraction.extracted);
    await c.settledRecheck(f.hash);
    let complete = true;
    let deferred = false;
    if (c.args.mechanical && judgesPack(c.args.judgement, job.extraction.pack)) {
        // Pass 1: the map has this file; its judgement is pass 2's. With sections at hand (in-thread parse) a file whose sections are all
        // still judged costs no model call: its facts are refreshed and it is complete. A pool returns none in a mechanical run, so those files are
        // owed and pass 2 finds nothing to judge (a cache hit, no call).
        const { reuse, toJudge } = splitSections(c.judgedPrior, job.extraction);
        if (job.extraction.sections.length > 0 && toJudge.length === 0) {
            for (const ref of reuse) c.stored.refresh(ref, f.path, f.hash, c.batch, c.entities);
        } else {
            complete = false;
            deferred = true;
            c.owedJudge.push(f.path);
            c.measured.set(f.path, Math.max(1, toJudge.length));
        }
    }
    if (judge) {
        c.progress({ phase: 'judge', done: i, total: c.work.length, path: f.path });
        a.gate.inWriter = false; // waiting for the model calls is not the writer working
        const outcome = await applyJudge(a.judgeCtx, f, judge);
        a.gate.inWriter = true;
        if (outcome === 'aborted') return false;
        if (outcome === 'failed') {
            c.owedJudge.push(f.path);
            // Judging failed as a whole: leave the file exactly as it was (facts, manifest hash, owed) so the next review retries it.
            c.measured.set(f.path, owedCalls(c.detail.extractor, f.path));
            return true;
        }
        complete = outcome.complete;
        if (!complete) c.owedJudge.push(f.path);
    }
    c.progress({ phase: 'write', done: i, total: c.work.length, path: f.path });
    const t0 = c.now();
    await writeBatch(c);
    // An owed file keeps its old judgement current until pass 2 replaces it: nothing of it is marked stale yet.
    if (!deferred) {
        await store.markStale(c.scope, srcRef(c.rootId, f.path), f.hash);
        c.report.staleMarked += 1;
    }
    c.args.onStep?.({ path: f.path, phase: 'write', ms: c.now() - t0, outcome: 'ok' });
    if (complete) { c.filesNow.set(f.path, f.hash); c.finished.add(f.path); }
    return true;
}
