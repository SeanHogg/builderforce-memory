/**
 * knowledge/review.ts — `reviewDirectory`: point it at a folder and it learns what the
 * folder says (PRD 38 §4.3).
 *
 *   plan (walk + manifest diff) → per changed file: read · parse (the ONE ParserRegistry,
 *   composed from the packs) · extract (cached by content hash) · judge (changed sections
 *   of judged kinds, within budget) · write · markStale → analyzers → compare near-duplicate
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
 */

import { sha256Hex } from '../retrieval/knowledge-index/sha256.js';
import type { ReviewArgs, ReviewReport, ScopeRecord, StoredFact, StoredRelation } from './model.js';
import { owedAfterRun, owedCalls, saveExtractorState } from './owedFiles.js';
import { planDetail, refreshPlan } from './plan.js';
import { composePacks, filesManifestKey, judgedManifestKey, manifestMap, saveManifestMap } from './sources.js';
import { knownKeysOfEntities, putExtract, putNoExtract } from './extractFile.js';
import { backfillCandidates, backfillNoExtract } from './backfillNoExtract.js';
import { prefetchOrdered } from './orderedPool.js';
import {
    DEFAULT_REVIEW_CONCURRENCY, DEFAULT_REVIEW_JUDGE_CONCURRENCY, DEFAULT_REVIEW_MODEL_CALLS, JUDGE_CONCURRENCY, JudgedBatch,
    type CompareOutcome, type JudgeBudget,
} from './judge.js';
import { createModelLimiter } from './modelLimiter.js';
import { JudgeBreaker } from './judgeCalls.js';
import { applyJudge, judgesPack, splitSections, startJudge, type JudgeJob } from './reviewJudge.js';
import { createPrefetch, type Prefetch } from './reviewPrefetch.js';
import { Lanes } from './reviewRunModel.js';
import { compareNew, pendingClaims, saveAwaiting } from './reviewCompare.js';
import { factsWithoutAnchor, StoredIndex } from './reviewStored.js';
import { canonicalJson, compareStr, srcRef } from './refs.js';

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

/** One changed file between dispatch and the ordered writer: what its prefetch found and its judging in flight. */
interface Slot {
    f: { path: string; hash: string };
    i: number;
    job: Prefetch;
    judge: JudgeJob | undefined;
}

/** Throws if a batch carries `extracted` items — perception is never memory. */
export function assertStoredBatch(batch: { facts: readonly StoredFact[]; relations: readonly StoredRelation[] }): void {
    const bad = [...batch.facts, ...batch.relations].find((x) => (x.origin as string) === 'extracted');
    if (bad) throw new Error('knowledge: extracted output is never written to a KnowledgeStore');
}

/**
 * A store that can hold persistence (`batched`) gets the whole review as one batch, so a snapshot
 * store writes a bounded number of times instead of once per file; any other store is unaffected.
 */
export async function reviewDirectory(args: ReviewArgs): Promise<ReviewReport> {
    return args.store.batched ? args.store.batched((flush) => reviewRun(args, flush)) : reviewRun(args, async () => {});
}

async function reviewRun(args: ReviewArgs, flush: () => Promise<void>): Promise<ReviewReport> {
    const now = () => Date.now();
    const started = now();
    const { source, store, packs, manifest, signal } = args;
    const rootId = source.rootId;
    const scope = { rootId };
    const progress = (p: Parameters<NonNullable<ReviewArgs['onProgress']>>[0]) => args.onProgress?.(p);
    const report: ReviewReport = {
        rootId, digest: '', extracted: 0, judged: 0, unjudged: 0, invalidJudgements: 0,
        factsWritten: 0, relationsWritten: 0, staleMarked: 0, readErrors: 0, extractErrors: 0, failures: [], awaitingJudge: [], aborted: false, durationMs: 0,
    };

    const detail = args.plan
        ? await refreshPlan(args.plan, manifest)
        : await planDetail(source, manifest, packs, { ...(args.force ? { force: true } : {}), compare: !!args.similar && !!args.judgement });
    progress({ phase: 'plan', done: detail.files.length, total: detail.files.length });
    const composed = composePacks(packs);
    const packIds = packs.map((p) => p.id).sort(compareStr);
    const scopeRecord: ScopeRecord = { rootId, title: source.title, packs: packIds };
    const [storedFacts, storedRelations] = await Promise.all([store.facts(scope), store.relations(scope)]);
    const stored = new StoredIndex(storedFacts, storedRelations);
    const entities = new Map(storedFacts.filter((f) => f.key.startsWith('ent:')).map((f) => [f.key, f] as const));
    const knownKeys = knownKeysOfEntities(entities.keys());
    const budget: JudgeBudget = {
        modelCalls: args.budget?.modelCalls ?? DEFAULT_REVIEW_MODEL_CALLS,
        deadline: started + (Number.isFinite(args.budget?.ms) ? args.budget.ms : Number.POSITIVE_INFINITY),
        used: 0, now,
    };
    const batch = new JudgedBatch();
    const breaker = new JudgeBreaker();
    const owedJudge: string[] = [];
    const judgedPrior = await manifestMap(manifest, judgedManifestKey(rootId));
    const judgedNow = new Map(judgedPrior);
    const filesNow = new Map(detail.prior);
    // A root reviewed before binary files were recorded: settle files with no body: binary-by-extension ones (no read), and the ones a replica asked to re-check.
    const asked = (await store.recheckHashes?.()) ?? [];
    // ONE bulk fetch of the extract cache for the whole run instead of a round trip per file: the files this run extracts and the ones the backfill looks up.
    const preChanged = detail.files.filter((f) => f.changed && (!args.only || args.only.includes(f.path)));
    const preload = [...new Set([...preChanged.map((f) => f.hash), ...backfillCandidates(detail, asked)])];
    if (preload.length > 0) await store.preloadExtracted?.(preload);
    await backfillNoExtract({ store, source, detail, asked, only: args.only, signal });
    // A re-check request is answered only when the work it asked for is done: the body stored, the binary recorded, or the failure recorded.
    const askedSet = new Set(asked);
    const settledRecheck = async (hash: string): Promise<void> => { if (askedSet.has(hash)) await store.clearRecheck?.([hash]); };
    // Claims awaiting comparison: those a previous run could not compare, then this run's.
    const newClaims = await pendingClaims(manifest, rootId, stored.facts);

    const write = async (withScope: ScopeRecord = scopeRecord) => {
        const out = batch.take();
        assertStoredBatch(out);
        if (out.facts.length === 0 && out.relations.length === 0 && withScope === scopeRecord) return;
        await store.write({ scope: withScope, ...out });
        report.factsWritten += out.facts.length;
        report.relationsWritten += out.relations.length;
    };

    // A source that is gone: its knowledge goes stale at no hash, its judged sections are forgotten, and
    // facts other sources still back drop its anchor. `rewritten` is the run's one record of such facts,
    // so a later retire() never restores an anchor an earlier one dropped.
    const rewritten = new Map<string, StoredFact>();
    const retire = async (paths: readonly string[]) => {
        const live = new Map([...stored.facts, ...entities, ...rewritten]);
        for (const path of paths) {
            await store.markStale(scope, srcRef(rootId, path), '');
            report.staleMarked += 1;
            filesNow.delete(path);
            const sectionPrefix = `${srcRef(rootId, path)}#`;
            for (const section of [...judgedNow.keys()]) if (section.startsWith(sectionPrefix)) judgedNow.delete(section);
            for (const fact of factsWithoutAnchor(live.values(), path)) {
                batch.fact(fact);
                live.set(fact.key, fact);
                rewritten.set(fact.key, fact);
                if (entities.has(fact.key)) entities.set(fact.key, fact);
            }
        }
        await write();
    };

    const changed = detail.files.filter((f) => f.changed);
    // `only` (pass 2) narrows what this run works on; the other changed files are untouched and stay owed (owedAfterRun sees them all).
    // Pass 2 works in the order it was given (priority), so a model-call ceiling is spent on the most important files first.
    const byPath = new Map(changed.map((f) => [f.path, f] as const));
    const work = args.only ? args.only.flatMap((p) => byPath.get(p) ?? []) : changed;
    // Changed files this run finished (or had nothing to do for); an owed file not in it stays owed,
    // at the model calls it still costs where this run could measure that (see owedFiles.ts).
    const finished = new Set<string>();
    const measured = new Map<string, number>();
    // Read + binary check + extract run ahead with bounded concurrency (extraction only reads `knownKeys`,
    // fixed above, and the content-hash extract cache); with `args.extractPool` the width is the pool's.
    // Judging runs `judgeWidth` files at once; ONE writer (`applyFile`) consumes them strictly in plan order.
    const concurrency = Math.min(32, Math.max(1, Math.floor(args.concurrency ?? DEFAULT_REVIEW_CONCURRENCY) || 1));
    // Read-ahead is wider than the pool (PRD 41 A10 rule 8) so workers never wait for the ordered writer.
    const readAhead = readAheadOf(args, concurrency);
    const judgeWidth = Math.min(concurrency, Math.min(16, Math.max(1, Math.floor(args.judgeConcurrency ?? DEFAULT_REVIEW_JUDGE_CONCURRENCY) || 1)));
    // The run's own stop: the caller's abort relays into it, and a thrown error (a local store failure) aborts it,
    // so queued and in-flight judge calls and pool jobs stop and nothing outlives the run.
    const stopper = new AbortController();
    // An abort that lands while the writer is mid-file ends the run at that file's boundary, as a sequential run
    // does (`cut`); one that lands anywhere else (a read, a model call) keeps every file whose work completed.
    let inWriter = false;
    let cut = false;
    const relay = () => { if (inWriter) cut = true; stopper.abort(signal?.reason); };
    if (signal?.aborted) relay(); else signal?.addEventListener('abort', relay, { once: true });
    const runSignal = stopper.signal;
    const prefetch = createPrefetch({ args, signal: runSignal, rootId, packs, store, composed, knownKeys, width: readAhead, extractWidth: concurrency });
    const dispatch = {
        // A mechanical run (pass 1) makes no model call: judged files stay owed.
        port: args.mechanical ? undefined : args.judgement, signal: runSignal, breaker, budget, judgedPrior, lanes: new Lanes(judgeWidth),
        limiter: createModelLimiter(args.maxModelCalls ?? JUDGE_CONCURRENCY),
    };
    const judgeCtx = { batch, entities, stored, judgedNow, newClaims, measured, detail, report, ...(args.onStep ? { onStep: args.onStep } : {}) };
    const fail = (path: string, phase: 'extract' | 'judge', reason: string) => report.failures.push({ path, phase, reason });

    /**
     * Applies one file, in plan order. False when its work was cut short by an abort (extracted, judging
     * unfinished): nothing of it is applied and it stays as it was, owed. A file whose work is complete is
     * applied even after an abort — the work is done, so it is kept.
     */
    const applyInner = async ({ f, i, job, judge }: Slot): Promise<boolean> => {
        progress({ phase: 'extract', done: i, total: work.length, path: f.path });
        // A read that throws (permissions, a lock, a race with a delete) or an extraction that throws says nothing
        // about the file: skipped with its knowledge and manifest hash intact, tried again next review.
        if (job.kind === 'read-error' || job.kind === 'failed') {
            if (job.kind === 'read-error') report.readErrors += 1; else report.extractErrors += 1;
            fail(f.path, 'extract', job.kind === 'read-error' ? `read: ${job.reason}` : job.reason);
            measured.set(f.path, owedCalls(detail.extractor, f.path)); // learns nothing: keep what it was owed
            // The manifest keeps the file at its old hash, which is what a replica syncs: if no body is stored for it, say so once (a terminal
            // answer, `[]` = nothing extracted), so the file is not held back and re-asked for forever. A later successful read replaces it.
            // (`[]` rather than a sibling "failed" member: the reason is already in `report.failures`, and a new store member would have no reader.)
            const known = detail.prior.get(f.path);
            if (known) { await putNoExtract(store, known); await settledRecheck(known); }
            return true;
        }
        // Recorded, so a file found binary on read is not re-read by every later review.
        if (job.kind === 'binary') {
            filesNow.set(f.path, f.hash);
            finished.add(f.path);
            await putNoExtract(store, f.hash); // the explicit "nothing extracted", so a replica can tell it from "not extracted yet"
            await settledRecheck(f.hash);
            await store.markStale(scope, srcRef(rootId, f.path), f.hash);
            report.staleMarked += 1;
            return true;
        }
        if (job.kind === 'none') { finished.add(f.path); return true; }
        report.extracted += 1;
        // The extract cache is written HERE, in plan order, so the cache, the counts and the order match a sequential run.
        if (!job.extraction.cached) await putExtract(store, f.hash, job.extraction.extracted);
        await settledRecheck(f.hash);
        let complete = true;
        let deferred = false;
        if (args.mechanical && judgesPack(args.judgement, job.extraction.pack)) {
            // Pass 1: the map has this file; its judgement is pass 2's. With sections at hand (in-thread parse) a file whose sections are all
            // still judged costs no model call: its facts are refreshed and it is complete. A pool returns none in a mechanical run, so those files are
            // owed and pass 2 finds nothing to judge (a cache hit, no call).
            const { reuse, toJudge } = splitSections(judgedPrior, job.extraction);
            if (job.extraction.sections.length > 0 && toJudge.length === 0) {
                for (const ref of reuse) stored.refresh(ref, f.path, f.hash, batch, entities);
            } else {
                complete = false;
                deferred = true;
                owedJudge.push(f.path);
                measured.set(f.path, Math.max(1, toJudge.length));
            }
        }
        if (judge) {
            progress({ phase: 'judge', done: i, total: work.length, path: f.path });
            inWriter = false; // waiting for the model calls is not the writer working
            const outcome = await applyJudge(judgeCtx, f, judge);
            inWriter = true;
            if (outcome === 'aborted') return false;
            if (outcome === 'failed') {
                owedJudge.push(f.path);
                // Judging failed as a whole: leave the file exactly as it was (facts, manifest hash, owed) so the next review retries it.
                measured.set(f.path, owedCalls(detail.extractor, f.path));
                return true;
            }
            complete = outcome.complete;
            if (!complete) owedJudge.push(f.path);
        }
        progress({ phase: 'write', done: i, total: work.length, path: f.path });
        const t0 = now();
        await write();
        // An owed file keeps its old judgement current until pass 2 replaces it: nothing of it is marked stale yet.
        if (!deferred) {
            await store.markStale(scope, srcRef(rootId, f.path), f.hash);
            report.staleMarked += 1;
        }
        args.onStep?.({ path: f.path, phase: 'write', ms: now() - t0, outcome: 'ok' });
        if (complete) { filesNow.set(f.path, f.hash); finished.add(f.path); }
        return true;
    };

    const applyFile = async (slot: Slot): Promise<boolean> => {
        inWriter = true;
        try { return await applyInner(slot); } finally { inWriter = false; }
    };
    const queue: Slot[] = [];
    let applied = 0;
    const drain = async (keep: number) => {
        while (!cut && queue.length > keep) if (await applyFile(queue.shift()!)) applied += 1;
    };
    let completed = false;
    try {
        for await (const { item: f, i, result } of prefetchOrdered(work, readAhead, prefetch, runSignal)) {
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
    if (applied < work.length) report.aborted = true;

    // Pass 2 (`only`) is judging alone: the map pass already retired the removed files and ran the analyzers.
    if (!report.aborted && !args.only) {
        await retire(detail.plan.removed);
        // Analyzers run in parallel and a failing one never aborts the review — its extracted
        // output is simply not stored (the file-level facts above are already durable), so a
        // single bad pack cannot sink the whole run.
        const analysed = await Promise.all(
            packs.filter((p) => p.analyzers).map(async (p) => {
                try { return await p.analyzers!.analyse(rootId, signal); } catch { return []; }
            }),
        );
        for (const packResults of analysed) {
            for (const ex of packResults) {
                await store.putExtracted?.(ex.sourceHash, [ex]);
                report.extracted += 1;
            }
        }
        progress({ phase: 'resolve', done: 0, total: newClaims.size });
    }

    // The compare phase needs a pack asking for it AND both ports.
    const wantsCompare = packs.some((p) => p.judgement?.contradictions);
    const canCompare = wantsCompare && !!args.judgement && !!args.similar;
    let outcome: CompareOutcome | undefined;
    if (!report.aborted && canCompare && !args.mechanical) {
        outcome = await compareNew({
            args, pending: newClaims, known: stored.facts, budget, batch,
            onPairs: (total) => progress({ phase: 'compare', done: 0, total }),
        });
        // Aborted only if the abort landed before the last pair was compared.
        if (outcome?.aborted) report.aborted = true;
    }
    // DURABILITY ORDER: everything the run produced is stored and durable BEFORE a manifest says its files are
    // done. A manifest ahead of the store loses knowledge for good (the next review sees those files unchanged
    // and skips them); a store ahead of the manifest only re-reviews files. `write()` first stores what the compare
    // phase left in the batch, then one flush covers the awaiting set (it names stored claims) and the files
    // manifest: nothing mutates the store between them. The trailing write only stamps lastDigest; the end of
    // the batch persists it.
    await write();
    await flush();
    // No pack asks for contradictions: nothing waits, and an earlier run's awaiting set is left as it was.
    if (wantsCompare && !args.mechanical) await saveAwaiting(manifest, rootId, newClaims, outcome);

    // Pass 2: what is still owed is every file of its slice it did not finish (an abort leaves the unreached ones owed too).
    report.awaitingJudge = args.only ? work.filter((f) => !finished.has(f.path)).map((f) => f.path) : owedJudge;
    report.awaitingJudgeCount = report.awaitingJudge.length;
    report.judgedFiles = args.only ? work.length - report.awaitingJudge.length : 0;
    if (breaker.stopped) report.judgeStopped = breaker.stopped;
    report.judged = batch.judged;
    report.unjudged = batch.unjudged;
    report.invalidJudgements = batch.invalid;
    report.digest = sha256Hex(canonicalJson({ rootId, packs: packIds, files: [...filesNow].sort((a, b) => compareStr(a[0], b[0])) }));
    await saveManifestMap(manifest, filesManifestKey(rootId), filesNow);
    await saveManifestMap(manifest, judgedManifestKey(rootId), judgedNow);
    // After every run: today's version and every file still owed a re-extraction — the unfinished ones,
    // content-changed or not, and the owed paths the plan did not list that the manifest still holds.
    await saveExtractorState(manifest, rootId, owedAfterRun({
        state: detail.extractor, changed, prior: detail.prior, finished, carried: detail.carried, live: filesNow, measured,
    }));
    await write(report.aborted ? scopeRecord : { ...scopeRecord, lastDigest: report.digest, lastReviewedAt: new Date(now()).toISOString() });
    report.durationMs = now() - started;
    progress({ phase: 'done', done: work.length, total: work.length });
    return report;
}
