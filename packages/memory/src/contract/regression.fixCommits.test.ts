/**
 * src/contract/regression.fixCommits.test.ts — fixes that shipped without a `regression:` test, each now pinned by one:
 *   c1d581c  scripts are code, not docs; `[[x]]` in code is not a link
 *   88fe16a  a Stop never fails the job already under way: the file in hand finishes (its deleted test pinned the reverted behaviour)
 *   cd6d9f9  a pool job's deadline adds only the SOURCE's read override to the extract ceiling, never the default read timeout
 * and the review's analyzers: a failing one is recorded, never silent, and the others still run.
 */

import { jest } from '@jest/globals';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { markdownParser } from '../ingest/parsers.js';
import type { SourceDocument } from '../ingest/types.js';
import { extractJob, type ExtractJob, type ExtractPool } from '../knowledge/extractJob.js';
import { createDeadlines } from '../knowledge/jobDeadline.js';
import type { ExtractContext, Extracted, KnowledgePack } from '../knowledge/model.js';
import { DOCS_PACK } from '../knowledge/packs/docs.js';
import { planReview } from '../knowledge/plan.js';
import { reviewDirectory } from '../knowledge/review.js';
import { analyzerFailurePath } from '../knowledge/reviewAnalyse.js';
import { asPathSource, resolveRelativePath } from '../knowledge/sources.js';
import { args, fakeJudge, FILES, memSource, memStore, quietPack, ROOT } from '../knowledge/testkit.test-util.js';

const SHELL = 'if [[ -f x ]]; then :; fi\n';

describe('regression: c1d581c scripts and code links', () => {
    it('regression: a shell script is not a docs-pack document, and [[x]] inside markdown code is not a link', async () => {
        const listing = {
            rootId: ROOT, title: 't',
            async *list() { yield { path: 'run.sh', hash: 'h1', size: SHELL.length }; yield { path: 'a.md', hash: 'h2', size: 4 }; },
            async read(path: string): Promise<SourceDocument> { return { id: path, tenantId: 't', content: { kind: 'text', text: path === 'run.sh' ? SHELL : '# A\n' } }; },
        };
        const plan = await planReview(listing, new InMemoryIngestManifest(), [DOCS_PACK]);
        expect(plan.skipped).toContainEqual({ path: 'run.sh', reason: 'no-pack' });

        const ctx: ExtractContext = { rootId: ROOT, knownKeys: new Map(), resolvePath: resolveRelativePath };
        const text = 'See [[real]].\n\n```sh\nif [[ -f x ]]; then :; fi\n```\n\nInline `[[nope]]` too.\n';
        const src = asPathSource({ id: 'a.md', tenantId: 't', content: { kind: 'text', text } }, 'a.md', 'h');
        const links = markdownParser.extract!(src, markdownParser.parse(src), ctx).edges.filter((e) => e.kind === 'links');
        expect(links).toHaveLength(1);
    });
});

describe('regression: 88fe16a the job in hand finishes on Stop', () => {
    it('regression: a stop that lands mid-job aborts the job signal but the job resolves with its work, it is not rejected', async () => {
        const parent = new AbortController();
        const withDeadline = createDeadlines(parent.signal);
        let seen: AbortSignal | undefined;
        let finish!: () => void;
        const job = withDeadline(60_000, async (signal) => {
            seen = signal;
            await new Promise<void>((r) => { finish = r; });
            return 'read';
        });
        await Promise.resolve();
        parent.abort(new Error('stopped'));
        await new Promise((r) => setImmediate(r));
        expect(seen?.aborted).toBe(true); // a pool may still recycle its worker
        finish();
        await expect(job).resolves.toBe('read');
    });
});

describe('regression: cd6d9f9 pool deadline', () => {
    /** A pool that reads the file itself and takes `ms` to answer. */
    const slowPool = (source: ReturnType<typeof memSource>, ms: number): ExtractPool => async (job) => {
        await new Promise((r) => setTimeout(r, ms));
        const doc = await source.read(job.path);
        const { readByPool: _drop, ...rest } = job;
        return extractJob({ ...rest, doc } as ExtractJob, [DOCS_PACK]);
    };
    /** Runs the review under fake timers, advancing the clock in steps until it settles: no real time decides the outcome. */
    const run = async (source: ReturnType<typeof memSource>) => {
        jest.useFakeTimers();
        try {
            const review = reviewDirectory(args(source, undefined, new InMemoryIngestManifest(), {
                mechanical: true, extractPool: slowPool(source, 150), deferRead: true, timeouts: { extractMs: 30, readMs: 60_000 },
            }));
            let done = false;
            review.then(() => { done = true; }, () => { done = true; });
            for (let i = 0; i < 1_000 && !done; i++) await jest.advanceTimersByTimeAsync(10);
            return await review;
        } finally {
            jest.useRealTimers();
        }
    };

    it('regression: the default read timeout is NOT added to a pool job deadline (a 60 s readMs once let a stuck job hold its slot)', async () => {
        const report = await run(memSource({ 'a.md': FILES['a.md'] }));
        expect(report.failures).toEqual([{ path: 'a.md', phase: 'extract', reason: 'timeout' }]);
    });

    it('regression: the source read override IS added to a pool job deadline', async () => {
        const source = memSource({ 'a.md': FILES['a.md'] });
        source.readTimeoutMs = () => 5_000;
        const report = await run(source);
        expect(report.failures).toEqual([]);
        expect(report.extracted).toBe(1);
    });
});

describe('regression: analyzer failures', () => {
    it('regression: a failing analyzer is recorded in report.failures (it vanished silently) and the other analyzers still store their output', async () => {
        const out = { sourceRef: `src:${ROOT}::`, sourceHash: '~ok:1', nodes: [], edges: [], keys: [] } as unknown as Extracted;
        const bad: KnowledgePack = { ...quietPack, analyzers: { async analyse() { throw new Error('analyzer exploded'); } } };
        const good: KnowledgePack = { ...quietPack, id: 'good', parsers: [], analyzers: { async analyse() { return [out]; } } };
        const store = memStore();
        const report = await reviewDirectory(args(memSource({ ...FILES }), store, new InMemoryIngestManifest(), { judgement: fakeJudge(), packs: [bad, good] }));
        expect(report.failures).toContainEqual({ kind: 'analyzer', path: analyzerFailurePath(quietPack.id), phase: 'analyse', reason: 'analyzer exploded' });
        expect(await store.getExtracted('~ok:1')).toEqual([out]);
        expect(report.aborted).toBe(false);
    });
});
