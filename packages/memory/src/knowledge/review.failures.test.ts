/**
 * src/knowledge/review.failures.test.ts
 * PRD 41 A8: no single failure ends a review, every file in every phase is a step event, and a host's
 * extraction pool is a port. A throwing extractor / judge is recorded and skipped; abort still stops.
 */

import { hashText } from '../ingest/hash.js';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { htmlParser, markdownParser, plainTextParser } from '../ingest/parsers.js';
import { extractJob, type ExtractJob, type ExtractPool } from './extractJob.js';
import type { JudgementPort, KnowledgePack, StoredFact } from './model.js';
import { LocalSnapshotStore } from './stores/localSnapshotStore.js';
import type { ReviewStep } from './reviewRunModel.js';
import { DOCS_PACK } from './packs/docs.js';
import { reviewDirectory } from './review.js';
import { filesManifestKey, manifestMap } from './sources.js';
import { args, fakeJudge, FILES, memPersistence, memSource, memStore, ROOT } from './testkit.test-util.js';

/** The docs pack whose markdown extractor throws for `bad.md`. */
const poisoned: KnowledgePack = {
    ...DOCS_PACK,
    parsers: [
        { ...markdownParser, extract: (s, sec, ctx) => {
            if (s.id === 'bad.md') throw new Error('poisoned extractor');
            return markdownParser.extract!(s, sec, ctx);
        } },
        htmlParser, plainTextParser,
    ],
};

const anchoredTo = (facts: readonly StoredFact[], path: string) =>
    facts.filter((f) => f.fileAnchors.some((a) => a.path === path));

describe('a failing file never ends a review', () => {
    it('a throwing extractor is one failed step; the run completes and the prior knowledge is kept', async () => {
        const source = memSource({ ...FILES, 'bad.md': '# Bad\nversion one\n' });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        const before = anchoredTo(await store.facts({ rootId: ROOT }), 'bad.md');
        const priorHash = (await manifestMap(manifest, filesManifestKey(ROOT))).get('bad.md');
        expect(before.length).toBeGreaterThan(0);

        source.files['bad.md'] = '# Bad\nversion two\n';
        const steps: ReviewStep[] = [];
        const report = await reviewDirectory(args(source, store, manifest, {
            packs: [poisoned], judgement: fakeJudge(), onStep: (s) => steps.push(s),
        }));
        expect(report.aborted).toBe(false);
        expect(report.extractErrors).toBe(1);
        expect(report.failures).toEqual([{ path: 'bad.md', phase: 'extract', reason: 'poisoned extractor' }]);
        const failed = steps.filter((s) => s.outcome === 'failed');
        expect(failed).toHaveLength(1);
        expect(failed[0]).toMatchObject({ path: 'bad.md', phase: 'extract', reason: 'poisoned extractor' });
        expect(anchoredTo(await store.facts({ rootId: ROOT }), 'bad.md')).toEqual(before);
        expect((await manifestMap(manifest, filesManifestKey(ROOT))).get('bad.md')).toBe(priorHash);

        // The next review (extractor fixed) retries it.
        const retry = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(retry.extractErrors).toBe(0);
        expect((await manifestMap(manifest, filesManifestKey(ROOT))).get('bad.md')).toBe(hashText(source.files['bad.md']!));
    });

    it('a judge that throws is recorded; the file stays unjudged and the run goes on', async () => {
        const source = memSource({ ...FILES });
        const judge = fakeJudge();
        const good = judge.judge.bind(judge);
        judge.judge = async (req) => {
            if (req.items[0]!.ref.includes('b.md')) return { get invalid(): number { throw new Error('judge exploded'); } } as never;
            return good(req);
        };
        const report = await reviewDirectory(args(source, memStore(), new InMemoryIngestManifest(), { judgement: judge }));
        expect(report.aborted).toBe(false);
        expect(report.failures).toEqual([{ path: 'b.md', phase: 'judge', reason: 'judge: judge exploded' }]);
        expect(report.unjudged).toBeGreaterThan(0);
        expect(report.judged).toBeGreaterThan(0);
    });

    it('when every call of a file fails, it is a judge failure that leaves the file exactly as it was', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        const factsBefore = anchoredTo(await store.facts({ rootId: ROOT }), 'a.md');
        const priorHash = (await manifestMap(manifest, filesManifestKey(ROOT))).get('a.md');
        expect(factsBefore.length).toBeGreaterThan(0);

        source.files['a.md'] = '# A\nalpha changed\n';
        const steps: ReviewStep[] = [];
        const down = { async judge(): Promise<never> { throw Object.assign(new Error('429 rate limited'), { status: 429 }); } };
        const report = await reviewDirectory(args(source, store, manifest, { judgement: down, onStep: (s) => steps.push(s) }));
        expect(report.aborted).toBe(false);
        expect(report.failures).toEqual([{ path: 'a.md', phase: 'judge', reason: 'judge: 429 rate limited' }]);
        expect(steps.filter((s) => s.path === 'a.md' && s.phase === 'judge')).toEqual([
            expect.objectContaining({ outcome: 'failed', reason: 'judge: 429 rate limited' }),
        ]);
        expect(steps.some((s) => s.path === 'a.md' && s.phase === 'write')).toBe(false);
        // Prior knowledge is not stale, the manifest hash is the old one, and the next review retries it.
        expect(anchoredTo(await store.facts({ rootId: ROOT }), 'a.md')).toEqual(factsBefore);
        expect((await store.relations({ rootId: ROOT })).filter((r) => r.kind === 'about').every((r) => !r.staleAt)).toBe(true);
        expect((await manifestMap(manifest, filesManifestKey(ROOT))).get('a.md')).toBe(priorHash);
        const retry = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(retry.failures).toEqual([]);
        expect((await manifestMap(manifest, filesManifestKey(ROOT))).get('a.md')).toBe(hashText(source.files['a.md']!));
    });

    it('when only some calls fail, the step is ok and says how many were unjudged', async () => {
        const source = memSource({ 'two.md': '# Good\ngood body\n\n## Other\nbad body\n' });
        const base = fakeJudge();
        const flaky: JudgementPort = {
            async judge(req) {
                if (req.items[0]!.text.includes('bad body')) throw new Error('flaky');
                return base.judge(req);
            },
        };
        const steps: ReviewStep[] = [];
        const report = await reviewDirectory(args(source, memStore(), new InMemoryIngestManifest(), { judgement: flaky, onStep: (s) => steps.push(s) }));
        expect(report.failures).toEqual([]);
        expect(steps.find((s) => s.phase === 'judge')).toMatchObject({ outcome: 'ok', reason: '1 unjudged (1 errors)' });
    });
});

describe('extractPool', () => {
    it('is called once per changed file with plain data, and its results are stored', async () => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const jobs: ExtractJob[] = [];
        const steps: ReviewStep[] = [];
        const extractPool: ExtractPool = async (job) => {
            jobs.push(job);
            const copy = JSON.parse(JSON.stringify(job)) as ExtractJob; // crosses a thread boundary: plain data only
            const out = extractJob(copy, [DOCS_PACK]);
            return { ...out, extracted: out.extracted.map((e) => ({
                ...e, nodes: e.nodes.map((n) => (n.kind === 'file' ? { ...n, label: `pool:${n.label}` } : n)),
            })) };
        };
        const report = await reviewDirectory(args(source, store, new InMemoryIngestManifest(), {
            extractPool, concurrency: 2, onStep: (s) => steps.push(s),
        }));
        expect(jobs.map((j) => j.path).sort()).toEqual(['a.md', 'b.md', 'c.md']);
        expect(report.extracted).toBe(3);
        const cached = await store.getExtracted!(hashText(FILES['a.md']));
        expect(cached!.flatMap((e) => e.nodes).some((n) => n.label === 'pool:a.md')).toBe(true);
        const extractSteps = steps.filter((s) => s.phase === 'extract');
        expect(extractSteps).toHaveLength(3);
        for (const s of extractSteps) expect([0, 1]).toContain(s.lane);

        // A second review finds every extract cached: the pool is not asked again.
        jobs.length = 0;
        await reviewDirectory(args(source, store, new InMemoryIngestManifest(), { extractPool, force: true }));
        expect(jobs).toHaveLength(0);
    });

    it('asks the pool for sections only when the file is judged, and never parses in-thread', async () => {
        const source = memSource({ ...FILES });
        const jobs: ExtractJob[] = [];
        const extractPool: ExtractPool = async (job) => { jobs.push(job); return extractJob(job, [DOCS_PACK]); };
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { extractPool }));
        expect(jobs.every((j) => !j.needSections && !j.sectionsOnly)).toBe(true);
        jobs.length = 0;
        const judge = fakeJudge();
        const report = await reviewDirectory(args(source, store, manifest, { extractPool, judgement: judge, force: true }));
        expect(report.failures).toEqual([]);
        // The extracts are cached, so only the sections are asked for.
        expect(jobs).toHaveLength(3);
        expect(jobs.every((j) => j.sectionsOnly)).toBe(true);
        expect(judge.requests.length).toBeGreaterThan(0);
    });

    const bad: Array<[string, ExtractPool]> = [
        ['rejects', async () => { throw new Error('worker crashed'); }],
        ['returns nothing although the packs accept the file', async () => ({ extracted: [] })],
        ['returns another file\'s extract', async (job) => {
            const other = extractJob({ ...job, path: 'other.md' }, [DOCS_PACK]);
            return other;
        }],
    ];
    it.each(bad)('a pool that %s is a failed extract step and the prior knowledge is kept', async (_name, pool) => {
        const source = memSource({ ...FILES });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        const before = anchoredTo(await store.facts({ rootId: ROOT }), 'a.md');
        const priorHash = (await manifestMap(manifest, filesManifestKey(ROOT))).get('a.md');

        source.files['a.md'] = '# A\nalpha edited\n';
        const steps: ReviewStep[] = [];
        const report = await reviewDirectory(args(source, store, manifest, { extractPool: pool, judgement: fakeJudge(), onStep: (s) => steps.push(s) }));
        expect(report.aborted).toBe(false);
        expect(report.extractErrors).toBe(1);
        expect(steps.filter((s) => s.path === 'a.md' && s.outcome === 'failed')).toHaveLength(1);
        expect(anchoredTo(await store.facts({ rootId: ROOT }), 'a.md')).toEqual(before);
        expect((await manifestMap(manifest, filesManifestKey(ROOT))).get('a.md')).toBe(priorHash);
    });
});

describe('abort', () => {
    it('an abort mid-extract stops promptly and keeps the work written before it', async () => {
        const N = 16;
        const text = Object.fromEntries(Array.from({ length: N }, (_, i) => [`d${i}.md`, `# T${i}\nbody ${i}\n`]));
        const source = memSource(text);
        const read = source.read.bind(source);
        const controller = new AbortController();
        source.read = async (path) => {
            if (path === 'd4.md') controller.abort();
            return read(path);
        };
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const report = await reviewDirectory(args(source, store, manifest, {
            judgement: fakeJudge(), signal: controller.signal, concurrency: 2,
        }));
        expect(report.aborted).toBe(true);
        expect(source.reads.length).toBeLessThan(N);
        const kept = [...(await manifestMap(manifest, filesManifestKey(ROOT))).keys()];
        expect(kept.length).toBeLessThan(N);
        expect(anchoredTo(await store.facts({ rootId: ROOT }), 'd0.md').length).toBeGreaterThan(0);
    });
});

describe('abort mid-judge', () => {
    it('starts no model call after the abort, stops promptly and does not stamp the digest', async () => {
        const N = 20;
        const text = Object.fromEntries(Array.from({ length: N }, (_, i) => [`d${i}.md`, `# T${i}\nbody ${i}\n\n## S${i}\nmore ${i}\n`]));
        const source = memSource(text);
        const controller = new AbortController();
        const base = fakeJudge();
        let calls = 0;
        let afterAbort = 0;
        const judgement: JudgementPort = {
            async judge(req) {
                if (controller.signal.aborted) afterAbort += 1;
                calls += 1;
                if (calls === 3) controller.abort();
                await new Promise((r) => setTimeout(r, 2));
                return base.judge(req);
            },
        };
        const persistence = memPersistence();
        const store = new LocalSnapshotStore(persistence, { now: () => '2026-10-28T00:00:00.000Z' });
        const report = await reviewDirectory(args(source, store, new InMemoryIngestManifest(), {
            judgement, signal: controller.signal, judgeConcurrency: 4, maxModelCalls: 4,
        }));
        expect(report.aborted).toBe(true);
        expect(afterAbort).toBe(0);
        expect(calls).toBeLessThan(N * 2);
        expect((await store.scopes()).every((s) => s.lastDigest === undefined)).toBe(true);

        // The control: an unaborted review does stamp it.
        const done = new LocalSnapshotStore(memPersistence(), { now: () => '2026-10-28T00:00:00.000Z' });
        await reviewDirectory(args(memSource(text), done, new InMemoryIngestManifest(), { judgement: fakeJudge() }));
        expect((await done.scopes()).some((s) => s.lastDigest !== undefined)).toBe(true);
    });
});

describe('onStep', () => {
    it('reports cached for reused knowledge and extract-cache hits, skipped for a binary file', async () => {
        const source = memSource({ ...FILES, 'blob.md': 'abc\u0000def' });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const first: ReviewStep[] = [];
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), onStep: (s) => first.push(s) }));
        const blob = first.filter((s) => s.path === 'blob.md');
        expect(blob).toEqual([expect.objectContaining({ phase: 'extract', outcome: 'skipped', reason: 'binary' })]);
        for (const p of ['a.md', 'b.md', 'c.md']) {
            const mine = first.filter((s) => s.path === p);
            expect(mine.map((s) => `${s.phase}:${s.outcome}`)).toEqual(['extract:ok', 'judge:ok', 'write:ok']);
        }

        const second: ReviewStep[] = [];
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), force: true, onStep: (s) => second.push(s) }));
        const a = second.filter((s) => s.path === 'a.md');
        expect(a.map((s) => `${s.phase}:${s.outcome}`)).toEqual(['extract:cached', 'judge:cached', 'write:ok']);
    });
});
