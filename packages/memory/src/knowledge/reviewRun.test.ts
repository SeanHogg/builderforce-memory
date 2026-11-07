/**
 * src/knowledge/reviewRun.test.ts
 * The pure run recorder (PRD 41 A8): phases, lanes, the step ring, failures, rate and ETA.
 */

import { applyRunEvent, REVIEW_RUN_STEP_RING, runEta, runRate, startRun, type ReviewRunEvent } from './reviewRun.js';
import type { ReviewStep } from './reviewRunModel.js';

const step = (i: number, extra: Partial<ReviewStep> = {}): ReviewStep => ({ path: `f${i}.md`, phase: 'extract', ms: 10, outcome: 'ok', lane: i % 2, ...extra });
const run0 = () => startRun({ id: 'r1', root: 'root', kind: 're-review', at: 1000 });
const fold = (events: Array<[ReviewRunEvent, number]>) => events.reduce((run, [e, at]) => applyRunEvent(run, e, at), run0());

describe('reviewRun reducer', () => {
    it('starts empty and never mutates the run it is given', () => {
        const start = run0();
        const next = applyRunEvent(start, { type: 'step', step: step(0) }, 1100);
        expect(start.steps).toHaveLength(0);
        expect(next.steps).toHaveLength(1);
        expect(start).toMatchObject({ id: 'r1', root: 'root', kind: 're-review', startedAt: 1000, phases: [], lanes: [], failures: [] });
    });

    it('tracks phases with done/total, ends the plan, and ends every open phase with the run', () => {
        const run = fold([
            [{ type: 'progress', progress: { phase: 'plan', done: 4, total: 4 } }, 1000],
            [{ type: 'progress', progress: { phase: 'extract', done: 0, total: 4, path: 'a' } }, 1100],
            [{ type: 'progress', progress: { phase: 'extract', done: 2, total: 4, path: 'c' } }, 1300],
            [{ type: 'end', outcome: 'done' }, 1500],
        ]);
        const byName = Object.fromEntries(run.phases.map((p) => [p.phase, p]));
        expect(byName['plan']).toMatchObject({ done: 4, total: 4, endedAt: 1000 });
        expect(byName['extract']).toMatchObject({ startedAt: 1100, done: 3, total: 4, endedAt: 1500 });
        expect(run).toMatchObject({ outcome: 'done', endedAt: 1500 });
    });

    it('keeps the last 200 steps in a ring but every failure', () => {
        let run = run0();
        for (let i = 0; i < 250; i++) {
            run = applyRunEvent(run, { type: 'step', step: step(i, i % 50 === 0 ? { outcome: 'failed', reason: `boom ${i}` } : {}) }, 1000 + i);
        }
        expect(run.steps).toHaveLength(REVIEW_RUN_STEP_RING);
        expect(run.steps[0]!.path).toBe('f50.md');
        expect(run.steps.at(-1)!.path).toBe('f249.md');
        expect(run.failures.map((f) => f.reason)).toEqual(['boom 0', 'boom 50', 'boom 100', 'boom 150', 'boom 200']);
        expect(run.failures[0]).toEqual({ path: 'f0.md', phase: 'extract', reason: 'boom 0' });
    });

    it('keeps an ended phase ended on a stale event, and reopens it only when the total grows', () => {
        const ended = fold([
            [{ type: 'progress', progress: { phase: 'extract', done: 3, total: 4 } }, 1100],
            [{ type: 'progress', progress: { phase: 'extract', done: 0, total: 4 } }, 1200], // stale: an earlier file
        ]);
        expect(ended.phases[0]).toMatchObject({ done: 4, total: 4, endedAt: 1100 });
        const grown = applyRunEvent(ended, { type: 'progress', progress: { phase: 'extract', done: 4, total: 9 } }, 1300);
        expect(grown.phases[0]!.endedAt).toBeUndefined();
    });

    it('tracks compare as its own phase', () => {
        const run = fold([
            [{ type: 'progress', progress: { phase: 'judge', done: 0, total: 5 } }, 1000],
            [{ type: 'progress', progress: { phase: 'compare', done: 0, total: 12 } }, 1100],
        ]);
        expect(run.phases.map((p) => [p.phase, p.total])).toEqual([['judge', 5], ['compare', 12]]);
    });

    it('accumulates lanes per phase: current path, files done, busy time', () => {
        const run = fold([
            [{ type: 'step', step: step(0) }, 1010],
            [{ type: 'step', step: step(1) }, 1020],
            [{ type: 'step', step: step(2, { ms: 30 }) }, 1030],
            [{ type: 'step', step: { path: 'f0.md', phase: 'judge', ms: 99, outcome: 'ok', lane: 0 } }, 1040],
        ]);
        expect(run.lanes).toEqual([
            { phase: 'extract', id: 0, path: 'f2.md', done: 2, busyMs: 40 },
            { phase: 'extract', id: 1, path: 'f1.md', done: 1, busyMs: 10 },
            { phase: 'judge', id: 0, path: 'f0.md', done: 1, busyMs: 99 },
        ]);
    });

    it('measures a rate from the phase clock and an ETA from the rate', () => {
        const run = fold([
            [{ type: 'progress', progress: { phase: 'extract', done: 0, total: 100 } }, 1000],
            [{ type: 'progress', progress: { phase: 'extract', done: 19, total: 100 } }, 3000],
        ]);
        expect(runRate(run, 'extract')).toBe(10); // 20 files in 2 s
        expect(runEta(run, 'extract')).toBe(8000); // 80 left at 10/s
        expect(runRate(run, 'write')).toBe(0);
        expect(runEta(run, 'write')).toBeUndefined();
    });

    it('has no ETA before a rate exists and 0 once a phase is complete', () => {
        const early = fold([[{ type: 'progress', progress: { phase: 'extract', done: 0, total: 10 } }, 1000]]);
        expect(runEta(early, 'extract')).toBeUndefined();
        const done = fold([[{ type: 'progress', progress: { phase: 'extract', done: 9, total: 10 } }, 1000]]);
        expect(runEta(done, 'extract')).toBe(0);
    });
});
