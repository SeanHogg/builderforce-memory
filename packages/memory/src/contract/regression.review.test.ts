/**
 * src/contract/regression.review.test.ts — bugs fixed in the review pipeline, each pinned by one test named
 * `regression: <what broke>`. A failing one means a fixed bug came back. (Bugs already pinned elsewhere are
 * listed in the CHANGELOG entry of the contract tests, not repeated here.)
 */

import { hashText } from '../ingest/hash.js';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import type { ReviewProgress } from '../knowledge/model.js';
import type { ReviewStep } from '../knowledge/reviewRunModel.js';
import { extractJob, type ExtractJob, type ExtractPool } from '../knowledge/extractJob.js';
import type { JudgementRequest, KnowledgePack, KnowledgeStore } from '../knowledge/model.js';
import { DOCS_PACK } from '../knowledge/packs/docs.js';
import { putNoExtract } from '../knowledge/extractFile.js';
import { planReview } from '../knowledge/plan.js';
import { srcRef } from '../knowledge/refs.js';
import { reviewDirectory } from '../knowledge/review.js';
import { applyRunEvent, failuresOf, failureTotal, REVIEW_RUN_FAILURE_CHUNK, startRun } from '../knowledge/reviewRun.js';
import { extractorManifestKey, filesManifestKey, judgedManifestKey, manifestMap } from '../knowledge/sources.js';
import {
    ALL, args, fakeJudge, FILES, memSource, memStore, quietPack, ROOT, stateOf, type SpySource,
} from '../knowledge/testkit.test-util.js';

const sixFiles = () => Object.fromEntries(Array.from({ length: 6 }, (_, i) => [`d${i}.md`, `# T${i}\nbody ${i}\n`]));

/** A pool that reads each file itself (`deferRead`), optionally slowly, as the worker pool of a host does. */
function readingPool(source: SpySource, onJob: (path: string) => void = () => undefined, delayMs = 0): ExtractPool {
    return async (job) => {
        onJob(job.path);
        const doc = await source.read(job.path);
        if (delayMs > 0 && job.path !== 'd0.md') await new Promise((r) => setTimeout(r, delayMs));
        return extractJob({ ...job, doc } as ExtractJob, [DOCS_PACK]);
    };
}

describe('regression: judging', () => {
    it('regression: a judge call that times out is not recorded as judged, and the next run judges the section again', async () => {
        const source = memSource({ 'a.md': '# A\nalpha\n' });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const timingOut = { async judge(_req: JudgementRequest): Promise<never> { throw new Error('judge: timeout after 60s'); } };

        const report = await reviewDirectory(args(source, store, manifest, { judgement: timingOut, packs: [quietPack] }));
        expect(report).toMatchObject({ judged: 0, unjudged: 1 });
        expect(await manifestMap(manifest, judgedManifestKey(ROOT))).toEqual(new Map()); // no section recorded as judged
        expect(await store.facts({ rootId: ROOT })).toEqual([]);
        expect((await planReview(source, manifest, [quietPack])).changed).toEqual(['a.md']); // still owed

        const judge = fakeJudge();
        const again = await reviewDirectory(args(source, store, manifest, { judgement: judge, packs: [quietPack] }));
        expect(judge.requests.filter((r) => r.kind === 'extract')).toHaveLength(1);
        expect(again.judged).toBe(1);
        expect((await manifestMap(manifest, judgedManifestKey(ROOT))).size).toBe(1);
    });
});

describe('regression: abort', () => {
    it('regression: an abort mid-read through a deferRead pool with read-ahead keeps the finished files clear and owes the unreached ones', async () => {
        const source = memSource(sixFiles());
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        await manifest.delete(extractorManifestKey(ROOT)); // the next review is an upgrade run: every file is owed until it is finished

        const controller = new AbortController();
        const pool = readingPool(source, (path) => { if (path === 'd1.md') controller.abort(); }, 25);
        const report = await reviewDirectory(args(source, store, manifest, {
            judgement: fakeJudge(), signal: controller.signal,
            extractPool: pool, deferRead: true, concurrency: 3, readAhead: 6,
        }));
        expect(report.aborted).toBe(true);

        const owed = (await stateOf(manifest)).owed;
        expect(owed).toContain('d5.md'); // never reached
        expect(owed.every((p) => p in sixFiles())).toBe(true);

        source.reads.length = 0;
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect([...source.reads].sort()).toEqual(owed); // only what was owed is read again
        expect((await stateOf(manifest)).owed).toEqual([]);
    }, 30_000);
});

describe('regression: abort listeners', () => {
    /** Counts the live 'abort' listeners per AbortSignal while a run goes. */
    function trackAbortListeners() {
        const proto = EventTarget.prototype;
        const add = proto.addEventListener;
        const remove = proto.removeEventListener;
        const live = new WeakMap<object, Set<unknown>>();
        let peak = 0;
        proto.addEventListener = function (this: EventTarget, type: string, listener: EventListenerOrEventListenerObject | null, options?: boolean | AddEventListenerOptions) {
            if (type === 'abort' && this instanceof AbortSignal) {
                const set = live.get(this) ?? new Set<unknown>();
                live.set(this, set);
                set.add(listener);
                peak = Math.max(peak, set.size);
            }
            return add.call(this, type, listener, options);
        } as typeof proto.addEventListener;
        proto.removeEventListener = function (this: EventTarget, type: string, listener: EventListenerOrEventListenerObject | null, options?: boolean | EventListenerOptions) {
            if (type === 'abort' && this instanceof AbortSignal) live.get(this)?.delete(listener);
            return remove.call(this, type, listener, options);
        } as typeof proto.removeEventListener;
        return { peak: () => peak, restore: () => { proto.addEventListener = add; proto.removeEventListener = remove; } };
    }

    it('regression: the run signal never carries more than a small constant number of abort listeners while 20,000 jobs run', async () => {
        const N = 20_000;
        const files = Object.fromEntries(Array.from({ length: N }, (_, i) => [`d${Math.floor(i / 100)}/f${i}.md`, `# T${i}\nbody ${i}\n`]));
        const source = memSource(files);
        const tracker = trackAbortListeners();
        try {
            const report = await reviewDirectory(args(source, memStore(), new InMemoryIngestManifest(), {
                mechanical: true, judgement: fakeJudge(), signal: new AbortController().signal,
                extractPool: readingPool(source), deferRead: true, concurrency: 8,
            }));
            expect(report.extracted).toBe(N);
            expect(report.failures).toEqual([]);
            expect(tracker.peak()).toBeLessThanOrEqual(8);
        } finally {
            tracker.restore();
        }
    }, 120_000);
});

describe('regression: pass 2 and pass 1', () => {
    /** A store that records the retirements (markStale at no hash) it is asked for. */
    function spyStore(retired: string[]): KnowledgeStore {
        const local = memStore();
        return {
            scopes: () => local.scopes(), facts: (s) => local.facts(s), relations: (s) => local.relations(s),
            write: (b) => local.write(b),
            markStale: (scope, ref, hash) => { if (hash === '') retired.push(ref); return local.markStale(scope, ref, hash); },
            getExtracted: (h) => local.getExtracted(h), putExtracted: (h, e) => local.putExtracted(h, e),
            batched: (work) => local.batched(work),
        };
    }

    it('regression: pass 2 (only) runs no analyzers and retires nothing; the map pass does both', async () => {
        let analysed = 0;
        const pack: KnowledgePack = { ...quietPack, analyzers: { async analyse() { analysed += 1; return []; } } };
        const retired: string[] = [];
        const store = spyStore(retired);
        const manifest = new InMemoryIngestManifest();
        const source = memSource({ ...FILES });
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), packs: [pack] }));
        expect(analysed).toBe(1);

        delete source.files['c.md'];
        source.files['a.md'] = '# A\nalpha two\n';
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), packs: [pack], only: ['a.md'] }));
        expect(analysed).toBe(1);
        expect(retired).toEqual([]);
        expect((await manifestMap(manifest, filesManifestKey(ROOT))).has('c.md')).toBe(true); // still known: the map pass retires it

        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), packs: [pack] }));
        expect(analysed).toBe(2);
        expect(retired).toEqual([srcRef(ROOT, 'c.md')]);
    });

    it('regression: the review leaves "write" for an "analyse" phase the moment the last file is written, and forwards analyzer progress', async () => {
        const pack: KnowledgePack = {
            ...quietPack,
            analyzers: { async analyse(_root, _signal, onProgress) { onProgress?.({ done: 1, total: 2 }); onProgress?.({ done: 2, total: 2 }); return []; } },
        };
        const events: Array<{ phase: string; done: number; total: number }> = [];
        const source = memSource({ ...FILES });
        await reviewDirectory(args(source, memStore(), new InMemoryIngestManifest(), {
            judgement: fakeJudge(), packs: [pack], onProgress: (p) => events.push({ phase: p.phase, done: p.done, total: p.total }),
        }));
        const lastWrite = events.map((e) => e.phase).lastIndexOf('write');
        const analyse = events.map((e, i) => (e.phase === 'analyse' ? i : -1)).filter((i) => i >= 0);
        const firstResolve = events.findIndex((e) => e.phase === 'resolve');
        expect(analyse.length).toBe(4);
        expect(analyse[0]!).toBeGreaterThan(lastWrite);
        expect(analyse[3]!).toBeLessThan(firstResolve);
        expect(analyse.map((i) => [events[i]!.done, events[i]!.total])).toEqual([[0, 1], [1, 2], [2, 2], [2, 2]]); // open, two forwarded, close
    });

    it('regression: an analyzer that never reports keeps the "analyse" row active while it runs, and the row ends before resolve', async () => {
        let release!: () => void;
        const gate = new Promise<void>((r) => { release = r; });
        let entered!: () => void;
        const inside = new Promise<void>((r) => { entered = r; });
        const pack: KnowledgePack = { ...quietPack, analyzers: { async analyse() { entered(); await gate; return []; } } };
        const events: ReviewProgress[] = [];
        const done = reviewDirectory(args(memSource({ ...FILES }), memStore(), new InMemoryIngestManifest(), {
            judgement: fakeJudge(), packs: [pack], onProgress: (p) => events.push(p),
        }));
        await inside;
        const fold = () => events.reduce((run, progress) => applyRunEvent(run, { type: 'progress', progress }, 1), startRun({ id: 'r', root: 'r', kind: 'review', at: 0 }));
        const during = fold().phases.find((p) => p.phase === 'analyse');
        expect(during).toBeDefined();
        expect(during!.endedAt).toBeUndefined();
        release();
        await done;
        const after = fold().phases.find((p) => p.phase === 'analyse')!;
        expect(after.endedAt).toBeDefined();
        expect(after.done).toBe(after.total);
        expect(events.findIndex((e) => e.phase === 'resolve')).toBeGreaterThan(events.map((e) => e.phase).lastIndexOf('analyse'));
    });

    it('regression: a Stop while the analyzers run ends the review as aborted and never reaches resolve', async () => {
        const stop = new AbortController();
        const pack: KnowledgePack = { ...quietPack, analyzers: { async analyse() { stop.abort(); return []; } } };
        const phases: string[] = [];
        const report = await reviewDirectory(args(memSource({ ...FILES }), memStore(), new InMemoryIngestManifest(), {
            judgement: fakeJudge(), packs: [pack], signal: stop.signal, onProgress: (p) => phases.push(p.phase),
        }));
        expect(report.aborted).toBe(true);
        expect(phases).toContain('analyse');
        expect(phases).not.toContain('resolve');
    });

    it('regression: pass 1 neither owes nor stale-marks a file whose sections are all reused', async () => {
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const source = memSource({ ...FILES });
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), packs: [quietPack] }));
        source.files['a.md'] = '# A\nalpha changed\n'; // a.md's section needs the model again; b.md and c.md only get re-read (force)

        const judge = fakeJudge();
        const report = await reviewDirectory(args(source, store, manifest, { judgement: judge, packs: [quietPack], mechanical: true, force: true }));
        expect(judge.requests).toHaveLength(0);
        expect(report.awaitingJudge).toEqual(['a.md']); // b.md and c.md owe nothing
        expect((await stateOf(manifest)).owed).toEqual(['a.md']);
        expect(ALL.filter((p) => !report.awaitingJudge.includes(p))).toEqual(['b.md', 'c.md']);
        // Nothing is stale: the reused files were refreshed, and the owed one keeps its old judgement current until pass 2.
        expect((await store.relations({ rootId: ROOT })).filter((r) => r.staleAt)).toEqual([]);
    });
});

describe('regression: review run failures', () => {
    it('regression: failure chunks append in constant work and failuresOf reads the flat list in order', () => {
        const N = REVIEW_RUN_FAILURE_CHUNK * 3 + 7;
        let run = startRun({ id: 'r', root: 'r', kind: 'review', at: 0 });
        const fail = (i: number) => ({ type: 'step' as const, step: { path: `f${i}.md`, phase: 'extract' as const, ms: 1, outcome: 'failed' as const, reason: `r${i}` } });
        for (let i = 0; i < N; i++) run = applyRunEvent(run, fail(i), i);

        expect(failureTotal(run)).toBe(N);
        expect(run.failures.map((c) => c.length)).toEqual([REVIEW_RUN_FAILURE_CHUNK, REVIEW_RUN_FAILURE_CHUNK, REVIEW_RUN_FAILURE_CHUNK, 7]);
        expect(failuresOf(run).map((f) => f.path)).toEqual(Array.from({ length: N }, (_, i) => `f${i}.md`));

        // Linear: a full chunk is shared by reference with every later run, never copied again.
        const before = run.failures;
        run = applyRunEvent(run, fail(N), N);
        expect(run.failures[0]).toBe(before[0]);
        expect(run.failures[2]).toBe(before[2]);
        expect(failureTotal(run)).toBe(N + 1);
    });
});

describe('regression: changed since plan', () => {
    const EDITED = '# A\nedited after the plan\n';

    it('regression: a file whose bytes changed between plan and read is a failed step "changed since plan", stores nothing under the planned hash, and is extracted by the next run', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const planned = hashText(FILES['a.md']);
        const read = source.read.bind(source);
        source.read = async (p) => (p === 'a.md' ? { id: p, tenantId: 't', content: { kind: 'text', text: EDITED } } : read(p));
        const steps: ReviewStep[] = [];

        const report = await reviewDirectory(args(source, store, manifest, { onStep: (s) => steps.push(s) }));
        expect(report.failures).toEqual([{ path: 'a.md', phase: 'extract', reason: 'changed since plan' }]);
        expect(report.extractErrors).toBe(1);
        expect(steps.filter((s) => s.path === 'a.md' && s.phase === 'extract')).toMatchObject([{ outcome: 'failed', reason: 'changed since plan' }]);
        expect(await store.getExtracted(planned)).toBeUndefined();
        expect((await manifestMap(manifest, filesManifestKey(ROOT))).has('a.md')).toBe(false); // still owed

        source.read = read;
        source.files['a.md'] = EDITED; // the fresh plan sees the new bytes
        const next = await reviewDirectory(args(source, store, manifest));
        expect(next.failures).toEqual([]);
        expect(await store.getExtracted(hashText(EDITED))).toBeDefined();
        expect((await manifestMap(manifest, filesManifestKey(ROOT))).get('a.md')).toBe(hashText(EDITED));
    });

    it('regression: a pool result whose own hash differs from the plan is a failed step "changed since plan"', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const pool: ExtractPool = async (job) => {
            const out = extractJob({ ...job, doc: await source.read(job.path) } as ExtractJob, [DOCS_PACK]);
            return job.path === 'a.md' ? { ...out, extracted: out.extracted.map((e) => ({ ...e, sourceHash: 'other' })) } : out;
        };
        const report = await reviewDirectory(args(source, store, new InMemoryIngestManifest(), { extractPool: pool, deferRead: true }));
        expect(report.failures).toEqual([{ path: 'a.md', phase: 'extract', reason: 'changed since plan' }]);
        expect(await store.getExtracted(hashText(FILES['a.md']))).toBeUndefined();
    });
});

describe('regression: a file with nothing to extract is told apart from one not extracted yet', () => {
    it('regression: a file found binary on read has an explicit stored [] (and a stale mark), never an absent body', async () => {
        const text = 'bin\u0000ary';
        const store = memStore();
        await reviewDirectory(args(memSource({ 'blob.txt': text }), store, new InMemoryIngestManifest()));
        expect(await store.getExtracted(hashText(text))).toEqual([]);
    });

    it('regression: a text file that is not extracted (here: its review failed) has NO stored body, so it is never mistaken for binary', async () => {
        const store = memStore();
        const text = '# A\nalpha\n';
        expect(await store.getExtracted(hashText(text))).toBeUndefined();
        await store.markStale({ rootId: ROOT }, srcRef(ROOT, 'a.md'), hashText(text)); // staled before it is extracted
        expect(await store.getExtracted(hashText(text))).toBeUndefined();
    });

    it('regression: recording "nothing extracted" never clobbers a hash that already has entries', async () => {
        const store = memStore();
        const text = '# A\nalpha\n';
        await reviewDirectory(args(memSource({ 'a.md': text }), store, new InMemoryIngestManifest()));
        const before = await store.getExtracted(hashText(text));
        expect(before!.length).toBeGreaterThan(0);
        await putNoExtract(store, hashText(text));
        expect(await store.getExtracted(hashText(text))).toEqual(before);
    });
});
