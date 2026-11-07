/**
 * src/knowledge/review.twoPass.test.ts — PRD 41 A10: the map is complete without the model (pass 1,
 * `mechanical`), judging follows in priority order within a budget (pass 2, `only`), the next run judges only what
 * is still owed, a hanging job is a `timeout` failure that never stalls the run, and consecutive provider failures
 * stop judging with the sections left owed.
 */

import { hashText } from '../ingest/hash.js';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { DOCS_PACK } from './packs/docs.js';
import { extractJob } from './extractJob.js';
import type { ExtractPool } from './extractJob.js';
import { orderForJudging } from './judgePriority.js';
import { reviewDirectory } from './review.js';
import { applyRunEvent, slowestSteps, startRun } from './reviewRun.js';
import { args, fakeJudge, memSource, memStore, quietPack, ROOT, stateOf } from './testkit.test-util.js';
import type { ReviewStep } from './reviewRunModel.js';

const N = 6;
const files = () => Object.fromEntries(Array.from({ length: N }, (_, i) => [`d${i}.md`, `# T${i}\nbody ${i}\n`]));
const byKey = (a: { key: string }, b: { key: string }) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
const factsOf = async (store: ReturnType<typeof memStore>) => [...(await store.facts({ rootId: ROOT }))].sort(byKey);

describe('two-pass review', () => {
    it('pass 1 makes no model call, completes the extract cache and lists every judged file as awaiting judgement', async () => {
        const judge = fakeJudge();
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const report = await reviewDirectory(args(memSource(files()), store, manifest, { judgement: judge, mechanical: true, packs: [quietPack] }));
        expect(judge.requests).toHaveLength(0);
        expect(report.awaitingJudge).toEqual(Object.keys(files()));
        expect(report.extracted).toBe(N);
        expect(await factsOf(store)).toEqual([]);
        expect(await store.getExtracted(hashText(files()['d0.md']!))).toBeDefined(); // the map has the file
        expect((await stateOf(manifest)).owed).toEqual(Object.keys(files()));
    });

    it('pass 2 judges only the files named, and the next run judges only what is still owed', async () => {
        const judge = fakeJudge();
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const source = memSource(files());
        const base = { judgement: judge, packs: [quietPack] };
        await reviewDirectory(args(source, store, manifest, { ...base, mechanical: true }));
        const first = await reviewDirectory(args(source, store, manifest, { ...base, only: ['d3.md', 'd1.md'] }));
        const judgedFirst = new Set(judge.requests.flatMap((r) => r.items.map((it) => it.ref as string)));
        expect(judgedFirst.size).toBe(2);
        expect([...judgedFirst].every((r) => r.includes('d1.md') || r.includes('d3.md'))).toBe(true);
        expect((await stateOf(manifest)).owed).toEqual(['d0.md', 'd2.md', 'd4.md', 'd5.md']);
        expect(first.judged).toBe(2);

        judge.requests.length = 0;
        const next = await reviewDirectory(args(source, store, manifest, base));
        const judgedNext = judge.requests.flatMap((r) => r.items.map((it) => it.ref as string));
        expect(judgedNext).toHaveLength(4);
        expect(judgedNext.some((r) => r.includes('d1.md') || r.includes('d3.md'))).toBe(false);
        expect(next.judged).toBe(4);
        expect((await stateOf(manifest)).owed).toEqual([]);
    });

    it('mechanical pass then pass 2 over every file stores the same facts as one sequential pass', async () => {
        const single = memStore();
        await reviewDirectory(args(memSource(files()), single, new InMemoryIngestManifest(), { judgement: fakeJudge(), packs: [quietPack], concurrency: 1, judgeConcurrency: 1 }));

        const two = memStore();
        const manifest = new InMemoryIngestManifest();
        const source = memSource(files());
        await reviewDirectory(args(source, two, manifest, { judgement: fakeJudge(), packs: [quietPack], mechanical: true, concurrency: 4 }));
        await reviewDirectory(args(source, two, manifest, { judgement: fakeJudge(), packs: [quietPack], only: Object.keys(files()), concurrency: 4 }));
        expect(await factsOf(two)).toEqual(await factsOf(single));
        const relKey = (r: { from: string; to: string; kind: string }) => `${r.from}|${r.to}|${r.kind}`;
        const sorted = async (st: ReturnType<typeof memStore>) => [...(await st.relations({ rootId: ROOT }))].sort((a, b) => (relKey(a) < relKey(b) ? -1 : 1));
        expect(await sorted(two)).toEqual(await sorted(single));
    });
});

describe('per-job timeouts', () => {
    const hang = (path: string): ExtractPool => (job, signal) =>
        job.path === path
            ? new Promise((_, reject) => signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true }))
            : Promise.resolve(extractJob(job, [DOCS_PACK]));

    it('a hanging extractor is one timeout failure (its worker told to stop) and the run continues', async () => {
        const steps: ReviewStep[] = [];
        const started = Date.now();
        const report = await reviewDirectory(args(memSource(files()), memStore(), new InMemoryIngestManifest(), {
            extractPool: hang('d2.md'), concurrency: 2, timeouts: { extractMs: 60 }, onStep: (s) => steps.push(s),
        }));
        expect(Date.now() - started).toBeLessThan(5_000);
        expect(report.aborted).toBe(false);
        expect(report.extractErrors).toBe(1);
        expect(report.failures).toEqual([{ path: 'd2.md', phase: 'extract', reason: 'timeout' }]);
        expect(report.extracted).toBe(N - 1);
        expect(steps.filter((s) => s.phase === 'extract' && s.outcome === 'failed').map((s) => s.reason)).toEqual(['timeout']);
    });

    it('a read that never returns is a failed read with a timeout and the file keeps its prior state', async () => {
        const source = memSource(files());
        const read = source.read.bind(source);
        source.read = (path) => (path === 'd1.md' ? new Promise(() => undefined) : read(path));
        const report = await reviewDirectory(args(source, memStore(), new InMemoryIngestManifest(), { timeouts: { readMs: 40 } }));
        expect(report.readErrors).toBe(1);
        expect(report.failures).toEqual([{ path: 'd1.md', phase: 'extract', reason: 'read: timeout' }]);
        expect(report.extracted).toBe(N - 1);
    });

    it('queued jobs are not timed out while they wait for a pool slot', async () => {
        let busy = 0, peak = 0;
        const slow: ExtractPool = async (job) => {
            busy += 1; peak = Math.max(peak, busy);
            await new Promise((r) => setTimeout(r, 30));
            busy -= 1;
            return extractJob(job, [DOCS_PACK]);
        };
        const report = await reviewDirectory(args(memSource(files()), memStore(), new InMemoryIngestManifest(), {
            extractPool: slow, concurrency: 1, readAhead: 6, timeouts: { extractMs: 200 },
        }));
        expect(peak).toBe(1);
        expect(report.failures).toEqual([]);
        expect(report.extracted).toBe(N);
    });
});

describe('judge breaker', () => {
    it('stops judging after consecutive failures; the unjudged sections stay owed and the next run judges them', async () => {
        let fail = true;
        const calls: string[] = [];
        const judge = {
            async judge(req: Parameters<ReturnType<typeof fakeJudge>['judge']>[0]) {
                calls.push(req.items[0]!.ref as string);
                if (fail) throw new Error('judge: timeout after 60s');
                return fakeJudge().judge(req);
            },
        };
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const source = memSource(files());
        const report = await reviewDirectory(args(source, store, manifest, { judgement: judge, packs: [quietPack], concurrency: 1, judgeConcurrency: 1 }));
        expect(calls.length).toBe(5);
        expect(report.judgeStopped).toMatch(/5 model calls failed in a row/);
        expect(report.judged).toBe(0);
        expect((await stateOf(manifest)).owed).toEqual(Object.keys(files()));

        fail = false;
        calls.length = 0;
        const next = await reviewDirectory(args(source, store, manifest, { judgement: judge, packs: [quietPack] }));
        expect(next.judged).toBe(N);
        expect((await stateOf(manifest)).owed).toEqual([]);
    });
});

describe('judge priority and the slowest steps', () => {
    it('orders entry points, then inbound references (most first), then recent, then the rest, ties by path', () => {
        const paths = ['z.md', 'b.md', 'a.md', 'recent.md', 'hub.md', 'README.md', 'few.md'];
        const order = orderForJudging(paths, {
            entry: (p) => p === 'README.md',
            inbound: new Map([['hub.md', 9], ['few.md', 2]]),
            recent: new Set(['recent.md']),
        });
        expect(order).toEqual(['README.md', 'hub.md', 'few.md', 'recent.md', 'a.md', 'b.md', 'z.md']);
        expect(paths[0]).toBe('z.md');
    });

    it('slowestSteps lists the longest steps first with their reasons', () => {
        let run = startRun({ id: 'r', root: ROOT, kind: 'review', at: 0 });
        for (const [path, ms, reason] of [['a', 5, undefined], ['b', 9000, 'timeout'], ['c', 300, undefined]] as const) {
            run = applyRunEvent(run, { type: 'step', step: { path, phase: 'extract', ms, outcome: reason ? 'failed' : 'ok', ...(reason ? { reason } : {}) } }, 1);
        }
        expect(slowestSteps(run, 2).map((s) => [s.path, s.reason])).toEqual([['b', 'timeout'], ['c', undefined]]);
    });
});
