/**
 * src/knowledge/review.parallel.test.ts
 * Read + extract are prefetched, judge/write stay in file order: a review whose reads resolve out
 * of order stores exactly what a one-at-a-time review stores, with the same report and progress.
 */

import { hashText } from '../ingest/hash.js';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import type { ReviewProgress } from './model.js';
import { reviewDirectory } from './review.js';
import { filesManifestKey, manifestMap } from './sources.js';
import { args, fakeJudge, memSource, memStore, ROOT, type SpySource } from './testkit.test-util.js';

const N = 12;
const files = () => Object.fromEntries(Array.from({ length: N }, (_, i) => [`d${i}.md`, `# T${i}\nbody ${i}\n`]));

/** A source whose read of file k waits N-k microtask turns: later files resolve first. */
function skewed(): { source: SpySource; maxInFlight: () => number } {
    const source = memSource(files());
    const read = source.read.bind(source);
    let inFlight = 0;
    let max = 0;
    source.read = async (path) => {
        inFlight += 1; max = Math.max(max, inFlight);
        const k = Number(/d(\d+)/.exec(path)![1]);
        for (let t = 0; t < N - k; t++) await Promise.resolve();
        const doc = await read(path);
        inFlight -= 1;
        return doc;
    };
    return { source, maxInFlight: () => max };
}

async function run(concurrency: number) {
    const { source, maxInFlight } = skewed();
    const store = memStore();
    const events: ReviewProgress[] = [];
    // A small budget so judgement is cut part-way: the budget must be spent in file order.
    const report = await reviewDirectory(args(source, store, new InMemoryIngestManifest(), {
        judgement: fakeJudge(), concurrency, budget: { modelCalls: 5, ms: 60_000 }, onProgress: (p) => events.push(p),
    }));
    const byKey = (a: { key: string }, b: { key: string }) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
    return {
        facts: [...(await store.facts({ rootId: ROOT }))].sort(byKey),
        relations: [...(await store.relations({ rootId: ROOT }))],
        report: { ...report, durationMs: 0 },
        events, maxInFlight: maxInFlight(),
    };
}

describe('reviewDirectory prefetch', () => {
    it('stores, reports and progresses identically at concurrency 4 and 1, while reads overlap', async () => {
        const parallel = await run(4);
        const serial = await run(1);
        expect(parallel.maxInFlight).toBeGreaterThan(1);
        expect(serial.maxInFlight).toBe(1);
        expect(parallel.facts.length).toBeGreaterThan(0);
        expect(parallel.facts).toEqual(serial.facts);
        expect(parallel.relations).toEqual(serial.relations);
        expect(parallel.report).toEqual(serial.report);
        expect(parallel.events).toEqual(serial.events);
    });

    it('an abort mid-run at default concurrency writes exactly the consumed prefix and resumes to the same store', async () => {
        const byKey = (a: { key: string }, b: { key: string }) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
        const factsOf = async (store: ReturnType<typeof memStore>) => [...(await store.facts({ rootId: ROOT }))].sort(byKey);

        const clean = memStore();
        await reviewDirectory(args(skewed().source, clean, new InMemoryIngestManifest(), { judgement: fakeJudge() }));

        const { source } = skewed();
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const controller = new AbortController();
        const report = await reviewDirectory(args(source, store, manifest, {
            judgement: fakeJudge(), signal: controller.signal,
            onProgress: (p) => { if (p.phase === 'write' && p.done === 1) controller.abort(); },
        }));
        expect(report.aborted).toBe(true);
        const stored = JSON.stringify(await factsOf(store));
        expect(stored).toContain('d0.md');
        expect(stored).toContain('d1.md');
        for (let k = 2; k < N; k++) expect(stored).not.toContain(`d${k}.md`);
        expect([...(await manifestMap(manifest, filesManifestKey(ROOT))).keys()].sort()).toEqual(['d0.md', 'd1.md']);

        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(await factsOf(store)).toEqual(await factsOf(clean));
    });

    it('files with identical content keep every sourceRef in the extract cache at any concurrency', async () => {
        const SAME = '# Same\nsame body\n';
        const text = { ...files(), 'x1.md': SAME, 'x2.md': SAME, 'x3.md': SAME };
        const go = async (concurrency: number) => {
            const source = memSource(text);
            const store = memStore();
            await reviewDirectory(args(source, store, new InMemoryIngestManifest(), { judgement: fakeJudge(), concurrency }));
            const hash = hashText(SAME);
            const refs = ((await store.getExtracted!(hash)) ?? []).map((e) => e.sourceRef).sort();
            const byKey = (a: { key: string }, b: { key: string }) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
            return { refs, facts: [...(await store.facts({ rootId: ROOT }))].sort(byKey) };
        };
        const parallel = await go(4);
        expect(parallel.refs).toHaveLength(3);
        expect(parallel).toEqual(await go(1));
    });
});
