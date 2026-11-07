/**
 * knowledge/reviewJudge.ts — the judge stage of a review, split in two so files can be judged N at
 * once while ONE writer keeps the result deterministic (PRD 41 A8).
 *
 *   • {@link startJudge} runs at DISPATCH, in plan order: it picks the sections to judge, reserves
 *     their model calls from the shared budget up front (so the budget is spent in file order whatever
 *     the concurrency) and starts the calls under the review's ONE {@link ModelLimiter}. It touches no
 *     shared state but the budget counter.
 *   • {@link applyJudge} runs at WRITE, in plan order: it merges the answers into the batch, the entity
 *     map, the judged-section manifest and the awaiting claims. One writer, so the bytes equal a
 *     one-file-at-a-time review's.
 *
 * A file whose EVERY call failed (a provider error: 401, 429, network) or whose judging threw is a
 * judge failure: nothing of it is applied, so the writer leaves its prior knowledge and manifest hash as
 * they were and the next review retries it. A budget cut, or some calls failing, is the older "Continue"
 * behaviour and is left as it was.
 */

import { hashText } from '../ingest/hash.js';
import { SECTION_INDEX_BADGE } from '../ingest/docsExtract.js';
import { budgetLeft, type JudgeBudget, type JudgedBatch, type SectionToJudge } from './judge.js';
import { applyAnswers, callSections, type SectionAnswer } from './judgeCalls.js';
import type { FileExtraction } from './extractFile.js';
import type { KnowledgePack, NodeRef, ReviewArgs, ReviewReport, StoredFact } from './model.js';
import type { ModelLimiter } from './modelLimiter.js';
import type { PlanDetail } from './plan.js';
import type { StoredIndex } from './reviewStored.js';
import { Lanes, reasonOf, type ReviewStep } from './reviewRunModel.js';

/** Whether the file's pack asks for judgement and a port is there to do it. */
export const judgesPack = (port: unknown, pack: KnowledgePack): boolean =>
    !!port && !!pack.judgement && !!(pack.judgement.claims || pack.judgement.entities);

type CallResult =
    | { status: 'ok'; answers: SectionAnswer[]; unjudged: number; errors: number; ms: number; lane?: number }
    | { status: 'aborted' }
    | { status: 'error'; reason: string; ms: number; lane?: number };

/** A file's judging in flight: what to merge once it settles. */
export interface JudgeJob {
    pack: string;
    contradictions: boolean;
    /** Sections whose judgement is current: their facts are refreshed, not judged. */
    reuse: NodeRef[];
    toJudge: SectionToJudge[];
    /** Never rejects (bar a broken lane invariant). */
    result: Promise<CallResult>;
}

export interface JudgeDispatch {
    port: NonNullable<ReviewArgs['judgement']> | undefined;
    /** The run's signal. */
    signal: AbortSignal;
    budget: JudgeBudget;
    judgedPrior: ReadonlyMap<string, string>;
    lanes: Lanes;
    limiter: ModelLimiter;
}

/** Starts judging a file's changed sections; undefined when its pack asks for no judgement (or there is no port). */
export function startJudge(d: JudgeDispatch, f: { path: string; hash: string }, extraction: FileExtraction): JudgeJob | undefined {
    const { extracted: ex, sections, pack } = extraction;
    const port = d.port;
    if (!port || !judgesPack(port, pack)) return undefined;
    const kinds = new Set(pack.judgement!.nodeKinds);
    const reuse: NodeRef[] = [];
    const toJudge: SectionToJudge[] = [];
    for (const node of ex.nodes) {
        const index = node.badges?.[SECTION_INDEX_BADGE];
        const text = typeof index === 'number' ? sections[index]?.text : undefined;
        if (!kinds.has(node.kind) || !text) continue;
        if (d.judgedPrior.get(node.ref) === hashText(text)) reuse.push(node.ref);
        else toJudge.push({ ref: node.ref, text });
    }
    // Reserved now, in plan order: this file's calls come out of the shared budget before any later file's.
    const allot = budgetLeft(d.budget) ? Math.min(toJudge.length, Math.max(0, d.budget.modelCalls - d.budget.used)) : 0;
    d.budget.used += allot;
    const slice: JudgeBudget = { modelCalls: allot, deadline: d.budget.deadline, used: 0, now: d.budget.now };
    const result = (async (): Promise<CallResult> => {
        if (toJudge.length === 0) return { status: 'ok', answers: [], unjudged: 0, errors: 0, ms: 0 };
        const lane = d.lanes.take();
        const t0 = Date.now();
        try {
            const r = await callSections({ port, fileHash: f.hash, sections: toJudge, budget: slice, limiter: d.limiter, signal: d.signal });
            // Every call that was made failed: the provider is refusing us (key, quota, network), not the file.
            if (r.answers.length === 0 && r.errors > 0) {
                return { status: 'error', reason: `judge: ${r.firstError ?? 'model call failed'}`, ms: Date.now() - t0, lane };
            }
            return { status: 'ok', answers: r.answers, unjudged: r.unjudged, errors: r.errors, ms: Date.now() - t0, lane };
        } catch (error) {
            if (d.signal.aborted) return { status: 'aborted' };
            return { status: 'error', reason: `judge: ${reasonOf(error)}`, ms: Date.now() - t0, lane };
        } finally {
            d.lanes.free(lane);
        }
    })();
    return { pack: pack.id, contradictions: !!pack.judgement!.contradictions, reuse, toJudge, result };
}

export interface JudgeApply {
    batch: JudgedBatch;
    entities: Map<string, StoredFact>;
    stored: StoredIndex;
    judgedNow: Map<string, string>;
    /** Claims awaiting comparison. */
    newClaims: Map<string, { text: string; hash: string }>;
    /** Model calls each unfinished file is still owed. */
    measured: Map<string, number>;
    detail: PlanDetail;
    report: ReviewReport;
    onStep?(step: ReviewStep): void;
}

/**
 * Merges a settled judge job. `aborted`: the file's judging was cut short, nothing was applied. `failed`:
 * judging failed as a whole, nothing was applied (the caller leaves the file as it was). Otherwise whether
 * the file is complete.
 */
export async function applyJudge(
    c: JudgeApply, f: { path: string; hash: string }, job: JudgeJob,
): Promise<'aborted' | 'failed' | { complete: boolean }> {
    const r = await job.result;
    if (r.status === 'aborted') return 'aborted';
    const step = { path: f.path, phase: 'judge' as const, ms: r.ms, ...(r.lane !== undefined ? { lane: r.lane } : {}) };
    if (r.status === 'error') {
        c.batch.unjudged += job.toJudge.length;
        c.report.failures.push({ path: f.path, phase: 'judge', reason: r.reason });
        c.onStep?.({ ...step, outcome: 'failed', reason: r.reason });
        return 'failed';
    }
    const unjudgedBefore = c.batch.unjudged;
    for (const ref of job.reuse) c.stored.refresh(ref, f.path, f.hash, c.batch, c.entities);
    c.batch.unjudged += r.unjudged;
    // Staged per file: a section counts as judged only once its facts are written, so an abort mid-file
    // leaves it to be judged again (written whole or not at all).
    const judgedHere = new Map<NodeRef, string>();
    const claims = applyAnswers({
        pack: job.pack, path: f.path, fileHash: f.hash, batch: c.batch, entities: c.entities,
        onJudged: (ref) => judgedHere.set(ref, hashText(job.toJudge.find((s) => s.ref === ref)!.text)),
    }, r.answers);
    // Only a pack that asks for contradictions has claims to compare (the estimate counts the same).
    // Deleted first so a re-judged claim moves to the newest position (the awaiting cap drops the oldest).
    if (job.contradictions) for (const x of claims) { c.newClaims.delete(x.ref); c.newClaims.set(x.ref, { text: x.text, hash: f.hash }); }
    for (const [ref, h] of judgedHere) c.judgedNow.set(ref, h);
    const reason = r.errors > 0 ? `${r.unjudged} unjudged (${r.errors} errors)` : r.unjudged > 0 ? `${r.unjudged} unjudged` : undefined;
    c.onStep?.({ ...step, outcome: job.toJudge.length === 0 ? 'cached' : 'ok', ...(reason ? { reason } : {}) });
    const complete = c.batch.unjudged === unjudgedBefore;
    // A content-changed file may revert to the manifest bytes: then all it had to judge — and what it was
    // already owed from earlier cuts — is still owed.
    if (!complete) {
        const contentChanged = c.detail.prior.get(f.path) !== f.hash;
        c.measured.set(f.path, contentChanged ? job.toJudge.length + (c.detail.extractor.owed.get(f.path) ?? 0) : c.batch.unjudged - unjudgedBefore);
    }
    return { complete };
}
