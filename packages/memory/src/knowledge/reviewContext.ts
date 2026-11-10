/**
 * knowledge/reviewContext.ts — the state ONE review run shares between its stages (review.ts): the plan and what the store
 * holds (opened by {@link openReview}), the run's books (files finished, owed, failed), and the two store writes every stage
 * uses: {@link writeBatch} (the judged batch, with the scope) and {@link retire} (sources that are gone).
 *
 * Moved out of review.ts's one long closure so each stage is a function of this context. No behaviour of its own beyond that.
 */

import type { ReviewArgs, ReviewReport, ScopeRecord, StoredFact, StoredRelation } from './model.js';
import { planDetail, refreshPlan, type PlanDetail, type PlannedFile } from './plan.js';
import { composePacks, judgedManifestKey, manifestMap } from './sources.js';
import { knownKeysOfEntities } from './extractFile.js';
import { backfillCandidates, backfillNoExtract } from './backfillNoExtract.js';
import { DEFAULT_REVIEW_MODEL_CALLS, JudgedBatch, type JudgeBudget } from './judge.js';
import { JudgeBreaker } from './judgeCalls.js';
import { pendingClaims, type PendingClaims } from './reviewCompare.js';
import { factsWithoutAnchor, StoredIndex } from './reviewStored.js';
import { compareStr, srcRef } from './refs.js';
import type { ComposedPacks } from './sources.js';
import type { NodeRef } from './model.js';

export type ReviewProgressEvent = Parameters<NonNullable<ReviewArgs['onProgress']>>[0];

export interface ReviewContext {
    args: ReviewArgs;
    rootId: string;
    scope: { rootId: string };
    scopeRecord: ScopeRecord;
    packIds: string[];
    report: ReviewReport;
    detail: PlanDetail;
    composed: ComposedPacks;
    stored: StoredIndex;
    entities: Map<string, StoredFact>;
    knownKeys: ReadonlyMap<string, NodeRef>;
    budget: JudgeBudget;
    batch: JudgedBatch;
    breaker: JudgeBreaker;
    judgedPrior: Map<string, string>;
    judgedNow: Map<string, string>;
    /** The files manifest as this run leaves it. */
    filesNow: Map<string, string>;
    newClaims: PendingClaims;
    /** Changed files of the plan, and the ones this run works on (`only` narrows them, in its order). */
    changed: PlannedFile[];
    work: PlannedFile[];
    /** Files whose judging is still owed. */
    owedJudge: string[];
    /** Changed files this run finished (or had nothing to do for). */
    finished: Set<string>;
    /** Files that failed for a reason that says nothing final about them, at the manifest's own bytes. */
    unreadable: Set<string>;
    /** Consecutive transient failures by path at one hash. */
    failures: Map<string, { hash: string; count: number }>;
    /** Model calls each unfinished file is still owed. */
    measured: Map<string, number>;
    /** Facts a retire rewrote: a later retire never restores an anchor an earlier one dropped. */
    rewritten: Map<string, StoredFact>;
    started: number;
    now(): number;
    progress(p: ReviewProgressEvent): void;
    /** Answers a replica's re-check request for `hash`, once its work is done. */
    settledRecheck(hash: string): Promise<void>;
}

const emptyReport = (rootId: string): ReviewReport => ({
    rootId, digest: '', extracted: 0, judged: 0, unjudged: 0, invalidJudgements: 0,
    factsWritten: 0, relationsWritten: 0, staleMarked: 0, readErrors: 0, extractErrors: 0, failures: [], awaitingJudge: [], aborted: false, durationMs: 0,
});

/** Plans the run, reads what the store holds of the root, preloads the extract cache and runs the no-extract backfill. */
export async function openReview(args: ReviewArgs): Promise<ReviewContext> {
    const now = () => Date.now();
    const started = now();
    const { source, store, packs, manifest, signal } = args;
    const rootId = source.rootId;
    const scope = { rootId };
    const progress = (p: ReviewProgressEvent) => args.onProgress?.(p);

    const detail = args.plan
        ? await refreshPlan(args.plan, manifest, args.extractorKey ?? args.plan.extractor.extractorKeyNow)
        : await planDetail(source, manifest, packs, {
            ...(args.force ? { force: true } : {}), compare: !!args.similar && !!args.judgement,
            ...(args.extractorKey ? { extractorKey: args.extractorKey } : {}),
        });
    progress({ phase: 'plan', done: detail.files.length, total: detail.files.length });
    const packIds = packs.map((p) => p.id).sort(compareStr);
    const [storedFacts, storedRelations] = await Promise.all([store.facts(scope), store.relations(scope)]);
    const stored = new StoredIndex(storedFacts, storedRelations);
    const entities = new Map(storedFacts.filter((f) => f.key.startsWith('ent:')).map((f) => [f.key, f] as const));
    const judgedPrior = await manifestMap(manifest, judgedManifestKey(rootId));
    // Files with no stored body: assets the manifest names (queued so their node is written, no read) and the ones a replica asked to re-check (queued; this review stores their body or asset node).
    const asked = (await store.recheckHashes?.()) ?? [];
    // ONE bulk fetch of the extract cache for the whole run instead of a round trip per file: the files this run extracts and the ones the backfill looks up.
    const preChanged = detail.files.filter((f) => f.changed && (!args.only || args.only.includes(f.path)));
    const preload = [...new Set([...preChanged.map((f) => f.hash), ...backfillCandidates(detail, asked)])];
    if (preload.length > 0) await store.preloadExtracted?.(preload);
    await backfillNoExtract({ store, detail, asked, only: args.only, signal });
    // A re-check request is answered only when the work it asked for is done: the body stored, the binary recorded, or the failure recorded.
    const askedSet = new Set(asked);
    // Claims awaiting comparison: those a previous run could not compare, then this run's.
    const newClaims = await pendingClaims(manifest, rootId, stored.facts);

    const changed = detail.files.filter((f) => f.changed);
    // `only` (pass 2) narrows what this run works on; the other changed files are untouched and stay owed (owedAfterRun sees them all).
    // Pass 2 works in the order it was given (priority), so a model-call ceiling is spent on the most important files first.
    const byPath = new Map(changed.map((f) => [f.path, f] as const));
    return {
        args, rootId, scope, scopeRecord: { rootId, title: source.title, packs: packIds }, packIds, report: emptyReport(rootId),
        detail, composed: composePacks(packs), stored, entities, knownKeys: knownKeysOfEntities(entities.keys()),
        budget: {
            modelCalls: args.budget?.modelCalls ?? DEFAULT_REVIEW_MODEL_CALLS,
            deadline: started + (Number.isFinite(args.budget?.ms) ? args.budget.ms : Number.POSITIVE_INFINITY),
            used: 0, now,
        },
        batch: new JudgedBatch(), breaker: new JudgeBreaker(),
        judgedPrior, judgedNow: new Map(judgedPrior), filesNow: new Map(detail.prior), newClaims,
        changed, work: args.only ? args.only.flatMap((p) => byPath.get(p) ?? []) : changed,
        owedJudge: [], finished: new Set(), unreadable: new Set(), failures: new Map(detail.extractor.failures ?? []), measured: new Map(),
        rewritten: new Map(), started, now, progress,
        settledRecheck: async (hash) => { if (askedSet.has(hash)) await store.clearRecheck?.([hash]); },
    };
}

/** Throws if a batch carries `extracted` items — perception is never memory. */
export function assertStoredBatch(batch: { facts: readonly StoredFact[]; relations: readonly StoredRelation[] }): void {
    const bad = [...batch.facts, ...batch.relations].find((x) => (x.origin as string) === 'extracted');
    if (bad) throw new Error('knowledge: extracted output is never written to a KnowledgeStore');
}

/** Stores what the judged batch holds (with `withScope`); a no-op when there is nothing and the scope is the run's own. */
export async function writeBatch(c: ReviewContext, withScope: ScopeRecord = c.scopeRecord): Promise<void> {
    const out = c.batch.take();
    assertStoredBatch(out);
    if (out.facts.length === 0 && out.relations.length === 0 && withScope === c.scopeRecord) return;
    await c.args.store.write({ scope: withScope, ...out });
    c.report.factsWritten += out.facts.length;
    c.report.relationsWritten += out.relations.length;
}

/**
 * Sources that are gone: their knowledge goes stale at no hash, their judged sections are forgotten, and facts other
 * sources still back drop their anchor (recorded in `rewritten`, so a later retire never restores it).
 */
export async function retire(c: ReviewContext, paths: readonly string[]): Promise<void> {
    const live = new Map([...c.stored.facts, ...c.entities, ...c.rewritten]);
    for (const path of paths) {
        await c.args.store.markStale(c.scope, srcRef(c.rootId, path), '');
        c.report.staleMarked += 1;
        c.filesNow.delete(path);
        const sectionPrefix = `${srcRef(c.rootId, path)}#`;
        for (const section of [...c.judgedNow.keys()]) if (section.startsWith(sectionPrefix)) c.judgedNow.delete(section);
        for (const fact of factsWithoutAnchor(live.values(), path)) {
            c.batch.fact(fact);
            live.set(fact.key, fact);
            c.rewritten.set(fact.key, fact);
            if (c.entities.has(fact.key)) c.entities.set(fact.key, fact);
        }
    }
    await writeBatch(c);
}
