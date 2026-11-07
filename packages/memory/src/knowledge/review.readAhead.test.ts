/**
 * src/knowledge/review.readAhead.test.ts
 * A pool that reads files itself runs a wide read-ahead window, so one slow head file does not stall the
 * other workers; and Stop settles the review at once however many jobs are queued behind the slot limiter.
 */

import { jest } from '@jest/globals';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { extractJob, type ExtractJob, type ExtractPool } from './extractJob.js';
import { DOCS_PACK } from './packs/docs.js';
import { reviewDirectory } from './review.js';
import { LocalSnapshotStore } from './stores/localSnapshotStore.js';
import { args, fakeJudge, memPersistence, memSource } from './testkit.test-util.js';

const FILE_COUNT = 400;
const files = (): Record<string, string> =>
    Object.fromEntries(Array.from({ length: FILE_COUNT }, (_, i) => [`d${String(i).padStart(4, '0')}.md`, `# T${i}\nbody ${i}\n`]));

/** A pool that reads the file itself (a `readByPool` job), then extracts it. */
function readingPool(source: ReturnType<typeof memSource>, hold: (job: ExtractJob, started: number) => Promise<void>): { pool: ExtractPool; started: () => number } {
    let started = 0;
    return {
        started: () => started,
        pool: async (job) => {
            started += 1;
            await hold(job, started);
            const doc = await source.read(job.readByPool!.path);
            const { readByPool: _drop, ...rest } = job;
            return extractJob({ ...rest, doc } as ExtractJob, [DOCS_PACK]);
        },
    };
}

describe('read-ahead with a pool that reads', () => {
    afterEach(() => jest.useRealTimers());

    it('regression: one slow head file does not stop the other files being extracted beyond 64 ahead', async () => {
        const source = memSource(files());
        const head = 'd0000.md';
        let startedWhenHeadSettled = 0;
        let release!: () => void;
        const gate = new Promise<void>((r) => { release = r; });
        const { pool, started } = readingPool(source, async (job, n) => {
            if (job.path === head) {
                // The head settles only once well over 64 other files have started.
                await Promise.race([gate, new Promise<void>((r) => setTimeout(r, 500))]);
                startedWhenHeadSettled = started();
            } else if (n > 100) release();
        });
        const report = await reviewDirectory(args(source, new LocalSnapshotStore(memPersistence()), new InMemoryIngestManifest(), {
            judgement: fakeJudge(), mechanical: true, extractPool: pool, deferRead: true, concurrency: 8,
        }));
        expect(report.aborted).toBe(false);
        expect(startedWhenHeadSettled - 1).toBeGreaterThan(64);
    }, 15_000);

    it('a judging (non-mechanical) review keeps the narrow window: at most 64 files ahead of a slow head', async () => {
        const source = memSource(files());
        const head = 'd0000.md';
        let startedWhenHeadSettled = 0;
        const { pool, started } = readingPool(source, async (job) => {
            if (job.path === head) {
                // Hold the head long enough for the window to fill, then record how far ahead the others got.
                await new Promise<void>((r) => setTimeout(r, 300));
                startedWhenHeadSettled = started();
            }
        });
        const report = await reviewDirectory(args(source, new LocalSnapshotStore(memPersistence()), new InMemoryIngestManifest(), {
            judgement: fakeJudge(), extractPool: pool, deferRead: true, concurrency: 8,
        }));
        expect(report.aborted).toBe(false);
        expect(startedWhenHeadSettled - 1).toBeLessThanOrEqual(64);
    });

    it('regression: Stop settles the review at once with hundreds of jobs queued behind the slots', async () => {
        jest.useFakeTimers();
        const source = memSource(files());
        const ctl = new AbortController();
        // Every pool job hangs and ignores its signal: only the run's own stop can settle them.
        const { pool, started } = readingPool(source, () => new Promise<void>(() => undefined));
        const run = reviewDirectory(args(source, new LocalSnapshotStore(memPersistence()), new InMemoryIngestManifest(), {
            judgement: fakeJudge(), mechanical: true, extractPool: pool, deferRead: true, concurrency: 8, signal: ctl.signal,
        }));
        await jest.advanceTimersByTimeAsync(50);
        expect(started()).toBe(8);
        ctl.abort();
        // No timer is advanced past the 10 s extract deadline: the stop alone settles every in-flight and queued job.
        const report = await run;
        expect(report.aborted).toBe(true);
    });
});
