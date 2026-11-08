/**
 * knowledge/reviewFailure.ts — what a review does with a file whose read or extraction failed (the ONE failure path).
 *
 * A failure is TRANSIENT unless proven final (see {@link ExtractFailed}): a read, a timeout, a crashed or disposed worker, a failing store, a stop.
 * A replica that holds the file back and asks about it needs an answer, but only a final one may be terminal (`[]` = nothing extracted):
 *
 *   - final: the manifest's bytes are gone (the hash moved), or the parser/extractor threw on them. `[]` is stored under the manifest's hash
 *     and the request is settled — unless an unchanged planned file still carries that hash (its content is readable: do not hide it);
 *   - transient at the manifest's own bytes: the request is kept and the path is made OWED, so the next normal review retries it;
 *     after {@link MAX_TRANSIENT_FAILURES} in a row at the same hash it is treated as final, so a file that never reads cannot hold a root forever;
 *   - never anything when the run was stopped.
 */

import { putNoExtract } from './extractFile.js';
import type { KnowledgeStore } from './model.js';
import type { PlanDetail } from './plan.js';
import type { Prefetch } from './reviewPrefetch.js';

/** Consecutive transient failures at one hash after which the failure is recorded as final. */
export const MAX_TRANSIENT_FAILURES = 3;

export interface FailureContext {
    store: KnowledgeStore;
    detail: PlanDetail;
    stopped: () => boolean;
    /** Paths to record as owed (a transient failure at the manifest's own bytes). */
    unreadable: Set<string>;
    /** Consecutive transient failures by path (persisted beside the owed entry). */
    failures: Map<string, { hash: string; count: number }>;
    settledRecheck: (hash: string) => Promise<void>;
}

export async function recordFailure(ctx: FailureContext, f: { path: string; hash: string }, job: Extract<Prefetch, { kind: 'read-error' | 'failed' }>): Promise<void> {
    const known = ctx.detail.prior.get(f.path);
    if (!known || ctx.stopped()) return;
    const transient = job.kind === 'read-error' || job.transient === true;
    if (known === f.hash && transient) {
        const prev = ctx.failures.get(f.path);
        const count = (prev?.hash === known ? prev.count : 0) + 1;
        if (count < MAX_TRANSIENT_FAILURES) {
            ctx.failures.set(f.path, { hash: known, count });
            ctx.unreadable.add(f.path);
            return;
        }
    }
    ctx.failures.delete(f.path);
    // An unchanged planned file that still carries `known` is readable content: a terminal `[]` would hide it.
    if (ctx.detail.files.some((g) => g.path !== f.path && !g.changed && g.hash === known)) return;
    await putNoExtract(ctx.store, known);
    await ctx.settledRecheck(known);
}
