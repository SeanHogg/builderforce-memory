/**
 * knowledge/reviewFinish.ts — the end of a review: the near-duplicate compare, then everything the run produced is stored and
 * durable BEFORE the manifests say its files are done, then the report's totals, the manifests and the scope's digest stamp.
 * Moved out of review.ts.
 */

import { sha256Hex } from '../retrieval/knowledge-index/sha256.js';
import type { ReviewReport } from './model.js';
import type { CompareOutcome } from './judge.js';
import { owedAfterRun, saveExtractorState } from './owedFiles.js';
import { filesManifestKey, judgedManifestKey, saveManifestMap } from './sources.js';
import { compareNew, saveAwaiting } from './reviewCompare.js';
import { writeBatch, type ReviewContext } from './reviewContext.js';
import { canonicalJson, compareStr } from './refs.js';

export async function finishReview(c: ReviewContext, flush: () => Promise<void>): Promise<ReviewReport> {
    const { args, report } = c;
    const { manifest } = args;
    // The compare phase needs a pack asking for it AND both ports.
    const wantsCompare = args.packs.some((p) => p.judgement?.contradictions);
    const canCompare = wantsCompare && !!args.judgement && !!args.similar;
    let outcome: CompareOutcome | undefined;
    if (!report.aborted && canCompare && !args.mechanical) {
        outcome = await compareNew({
            args, pending: c.newClaims, known: c.stored.facts, budget: c.budget, batch: c.batch,
            onPairs: (total) => c.progress({ phase: 'compare', done: 0, total }),
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
    await writeBatch(c);
    await flush();
    // No pack asks for contradictions: nothing waits, and an earlier run's awaiting set is left as it was.
    if (wantsCompare && !args.mechanical) await saveAwaiting(manifest, c.rootId, c.newClaims, outcome);

    // Pass 2: what is still owed is every file of its slice it did not finish (an abort leaves the unreached ones owed too).
    report.awaitingJudge = args.only ? c.work.filter((f) => !c.finished.has(f.path)).map((f) => f.path) : c.owedJudge;
    report.awaitingJudgeCount = report.awaitingJudge.length;
    report.judgedFiles = args.only ? c.work.length - report.awaitingJudge.length : 0;
    if (c.breaker.stopped) report.judgeStopped = c.breaker.stopped;
    report.judged = c.batch.judged;
    report.unjudged = c.batch.unjudged;
    report.invalidJudgements = c.batch.invalid;
    report.digest = sha256Hex(canonicalJson({ rootId: c.rootId, packs: c.packIds, files: [...c.filesNow].sort((a, b) => compareStr(a[0], b[0])) }));
    await saveManifestMap(manifest, filesManifestKey(c.rootId), c.filesNow);
    await saveManifestMap(manifest, judgedManifestKey(c.rootId), c.judgedNow);
    // After every run: today's version and every file still owed a re-extraction — the unfinished ones,
    // content-changed or not, and the owed paths the plan did not list that the manifest still holds.
    await saveExtractorState(manifest, c.rootId, owedAfterRun({
        state: c.detail.extractor, changed: c.changed, prior: c.detail.prior, finished: c.finished, carried: c.detail.carried,
        live: c.filesNow, measured: c.measured, unreadable: c.unreadable,
    }), c.failures, c.detail.extractor.documentsNow,
    // An aborted run keeps the key the root was last reviewed under: every file stays owed until a run under the new key finishes.
    report.aborted ? c.detail.extractor.extractorKey : c.detail.extractor.extractorKeyNow);
    await writeBatch(c, report.aborted ? c.scopeRecord : { ...c.scopeRecord, lastDigest: report.digest, lastReviewedAt: new Date(c.now()).toISOString() });
    report.durationMs = c.now() - c.started;
    c.progress({ phase: 'done', done: c.work.length, total: c.work.length });
    return report;
}
