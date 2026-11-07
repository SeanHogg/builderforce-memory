/**
 * knowledge/judgeCalls.ts — judging one file's sections, in two stages (PRD 41 A8).
 *
 *   • {@link callSections} — the CALL stage: one model call per section within a budget slice, touching
 *     no shared state, so several files can be in it at once (optionally under ONE shared
 *     {@link ModelLimiter}).
 *   • {@link applyAnswers} — the APPLY stage: merges answers into the batch and the shared entity map.
 *     Synchronous; the review's one ordered writer calls it in plan order, which keeps the stored
 *     bytes independent of how many files were judged at once.
 *
 * {@link judgeSections} is both stages for one file.
 */

import { hashText } from '../ingest/hash.js';
import {
    budgetLeft, call, claimContent, JUDGE_CONCURRENCY, REVIEW_FACT_SOURCE, runPool, validateExtract,
    type ExtractArgs, type JudgeBudget, type SectionToJudge,
} from './judge.js';
import type { ModelLimiter } from './modelLimiter.js';
import type { JudgementPort, NodeRef, StoredFact } from './model.js';
import { compareStr, factRef } from './refs.js';
import { entityKey, normalizeKey } from './resolve.js';
import { reasonOf } from './reviewRunModel.js';

/** A section's validated answer: the output of the call stage, applied later by {@link applyAnswers}. */
export interface SectionAnswer { section: SectionToJudge; v: ReturnType<typeof validateExtract> }

export interface CallSectionsArgs {
    port: JudgementPort;
    fileHash: string;
    sections: readonly SectionToJudge[];
    /** This file's slice of the review budget (its own `used`). */
    budget: JudgeBudget;
    /** Shared across the files of a review: caps the calls in flight and adapts to rate limits. */
    limiter?: ModelLimiter;
    signal?: AbortSignal;
}

export interface CallSectionsResult {
    answers: SectionAnswer[];
    /** Sections cut by the budget or whose call failed. */
    unjudged: number;
    /** Calls the port (or the provider behind it) failed. */
    errors: number;
    /** The first failure's message, single line. */
    firstError?: string;
}


/** Answers come back in section order. Throws on abort. */
export async function callSections(a: CallSectionsArgs): Promise<CallSectionsResult> {
    const slots: Array<SectionAnswer | undefined> = new Array(a.sections.length);
    let unjudged = 0;
    let errors = 0;
    let firstError: string | undefined;
    const hooks = {
        ...(a.limiter ? { limiter: a.limiter } : {}),
        onError: (err: unknown) => { errors += 1; firstError ??= reasonOf(err); },
    };
    // The budget check and `call`'s synchronous `used` increment happen on the same tick (no await
    // between), so concurrency cannot overshoot `modelCalls`.
    const abortError = () => a.signal?.reason ?? new Error('aborted');
    await runPool(a.sections.map((section, idx) => ({ section, idx })), JUDGE_CONCURRENCY, async ({ section, idx }) => {
        // Honoured per section, not only when the port throws: the caller drops this file whole.
        if (a.signal?.aborted) throw abortError();
        if (!budgetLeft(a.budget)) { unjudged += 1; return; }
        const answer = await call(a.port, {
            kind: 'extract', items: [{ ref: section.ref, text: section.text, sourceHash: a.fileHash }],
            ...(a.signal ? { signal: a.signal } : {}),
        }, a.budget, hooks);
        // A failed call judged nothing: not counted judged, so the file stays changed and is retried.
        if (!answer) { unjudged += 1; return; }
        // The abort can land while a call is in flight (the port sees the signal): a section that
        // finished after the abort must not count as judged, or the caller would not re-judge it.
        if (a.signal?.aborted) throw abortError();
        slots[idx] = { section, v: validateExtract(answer.result, section.ref) };
    });
    return { answers: slots.filter((s): s is SectionAnswer => !!s), unjudged, errors, ...(firstError !== undefined ? { firstError } : {}) };
}

/** Judges each section within the budget; the rest — and any whose call failed — are counted `unjudged`. */
export async function judgeSections(a: ExtractArgs): Promise<Array<{ ref: NodeRef; text: string }>> {
    const { answers, unjudged } = await callSections({
        port: a.port, fileHash: a.fileHash, sections: a.sections, budget: a.budget, ...(a.signal ? { signal: a.signal } : {}),
    });
    a.batch.unjudged += unjudged;
    return applyAnswers(a, answers);
}

/** Merges answers into the batch and the shared entities map, in the order given. */
export function applyAnswers(
    a: Pick<ExtractArgs, 'pack' | 'path' | 'fileHash' | 'batch' | 'entities' | 'onJudged'>,
    answers: readonly SectionAnswer[],
): Array<{ ref: NodeRef; text: string }> {
    const claims: Array<{ ref: NodeRef; text: string }> = [];
    const anchor = { path: a.path, contentHash: a.fileHash };
    for (const { section, v } of answers) {
        a.batch.invalid += v.invalid;
        a.batch.judged += 1;
        a.onJudged(section.ref);
        const rel = (from: NodeRef, to: NodeRef, kind: string) =>
            a.batch.relation({ from, to, kind, weight: 1, origin: 'model', sourceHash: a.fileHash });

        const entityRefs = new Map<string, NodeRef>();
        for (const e of v.entities) {
            const key = entityKey(a.pack, e.name);
            const prior = a.entities.get(key);
            const anchors = [...(prior?.fileAnchors ?? []).filter((x) => x.path !== a.path), anchor]
                .sort((x, y) => compareStr(x.path, y.path));
            const fact: StoredFact = {
                key, content: prior?.content ?? `${e.name} (${e.type})`, source: REVIEW_FACT_SOURCE,
                importance: prior?.importance ?? 0.5, origin: prior?.origin ?? 'model', fileAnchors: anchors,
            };
            a.entities.set(key, fact);
            a.batch.fact(fact);
            entityRefs.set(normalizeKey(e.name), key);
            rel(section.ref, key, 'mentions');
        }
        for (const c of v.claims) {
            const key = `claim:${hashText([section.ref, c.subject, c.predicate, c.object].join('\u0000'))}`;
            const text = `${c.subject} ${c.predicate} ${c.object}`;
            a.batch.fact({
                key, content: claimContent(text, c.quote), source: REVIEW_FACT_SOURCE,
                importance: 0.5, origin: 'model', fileAnchors: [anchor],
            });
            const ref = factRef(key);
            rel(ref, section.ref, 'about');
            for (const term of [c.subject, c.object]) {
                const ent = entityRefs.get(normalizeKey(term));
                if (ent) rel(ref, ent, 'mentions');
            }
            claims.push({ ref, text });
        }
    }
    return claims;
}
