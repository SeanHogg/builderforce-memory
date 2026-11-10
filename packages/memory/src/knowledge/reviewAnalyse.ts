/**
 * knowledge/reviewAnalyse.ts — the `analyse` stage of a review: every pack's analyzers run in parallel, under their own progress
 * row, and their output goes to the extract cache (perception, never memory). Moved out of review.ts.
 *
 * A failing analyzer never aborts the review — its output is simply not stored (the file-level facts are already durable), so
 * one bad pack cannot sink the run — but it is never silent either: it is recorded in `report.failures` (`kind: 'analyzer'`, phase `analyse`, path
 * `analyzer:<pack id>`) and the other analyzers keep running.
 */

import type { AnalyzerProgress, Extracted } from './model.js';
import type { ReviewContext } from './reviewContext.js';
import { reasonOf } from './reviewRunModel.js';

/** The `path` an analyzer failure is recorded under. */
export const analyzerFailurePath = (packId: string): string => `analyzer:${packId}`;

/**
 * Runs the analyzers and stores what they extracted. Each pack's latest count is summed; only `done` is monotonic. The caller
 * opened the `analyse` row (as 0/1); this closes it.
 */
export async function runAnalyzers(c: ReviewContext): Promise<void> {
    const { packs, signal } = c.args;
    const latest = new Map<number, AnalyzerProgress>();
    const sum = () => {
        let done = 0; let total = 0;
        for (const v of latest.values()) { done += v.done; total += v.total; }
        return { done, total: Math.max(1, total) };
    };
    const analysed = await Promise.all(
        packs.filter((p) => p.analyzers).map(async (p, k): Promise<Extracted[]> => {
            let settled = false;
            // A total may grow as a pack discovers more work. A callback after its pack settled is dropped.
            const onProgress = (a: AnalyzerProgress) => {
                if (settled) return;
                latest.set(k, { done: a.done, total: a.total });
                c.progress({ phase: 'analyse', ...sum(), ...(a.path !== undefined ? { path: a.path } : {}) });
            };
            try {
                return await p.analyzers!.analyse(c.rootId, signal, onProgress);
            } catch (error) {
                c.report.failures.push({ kind: 'analyzer', path: analyzerFailurePath(p.id), phase: 'analyse', reason: reasonOf(error) });
                return [];
            } finally { settled = true; }
        }),
    );
    // Close the row whatever the analyzers reported (none, short, or a throw).
    const closing = Math.max(1, sum().total);
    c.progress({ phase: 'analyse', done: closing, total: closing });
    return storeAnalysed(c, analysed);
}

async function storeAnalysed(c: ReviewContext, analysed: readonly Extracted[][]): Promise<void> {
    for (const packResults of analysed) {
        for (const ex of packResults) {
            await c.args.store.putExtracted?.(ex.sourceHash, [ex]);
            c.report.extracted += 1;
        }
    }
}
