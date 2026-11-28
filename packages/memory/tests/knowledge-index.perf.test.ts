import { makeDocs, makeEmbedder, makeQueries, buildIndex } from './knowledge-index.fixture.js';

function percentile(samples: readonly number[], fraction: number): number {
    return [...samples].sort((a, b) => a - b)[Math.ceil(samples.length * fraction) - 1]!;
}

// Opt in to the CPU gate so ordinary unit-test concurrency does not affect its timings.
(process.env.BF_KNOWLEDGE_INDEX_PERF === '1' ? describe : describe.skip)('50,000-card knowledge index', () => {
    let docs: ReturnType<typeof makeDocs>;
    let embedder: ReturnType<typeof makeEmbedder>;
    let index: Awaited<ReturnType<typeof buildIndex>>;

    beforeAll(async () => {
        docs = makeDocs(50_000);
        embedder = makeEmbedder(32);
        index = await buildIndex(docs, embedder, {});
    }, 1_800_000);

    it('meets the warm CPU retrieval and exact lookup budgets', async () => {
        expect(index.storage().nodes).toBe(50_000);
        const queries = makeQueries(100);
        for (const query of queries) await index.search(query, { k: 10 });
        const calls = embedder.calls;
        const searchMs: number[] = [];
        const exactMs: number[] = [];
        for (const [i, query] of queries.entries()) {
            const searchStart = performance.now();
            const results = await index.search(query, { k: 10 });
            searchMs.push(performance.now() - searchStart);
            expect(results).toHaveLength(10);

            const doc = docs[i * 500]!;
            const exactStart = performance.now();
            const exact = await index.search(doc.keys![0]!, { k: 1 });
            exactMs.push(performance.now() - exactStart);
            expect(exact[0]).toMatchObject({ id: doc.id, tier: 'exact' });
        }
        // The query itself is embedded each time. Only rerank card vectors are cached.
        expect(embedder.calls - calls).toBe(queries.length);
        const p50 = percentile(searchMs, 0.5), p95 = percentile(searchMs, 0.95);
        const exactMax = Math.max(...exactMs);
        console.info({ cards: docs.length, dimensions: embedder.dims, samples: queries.length, p50, p95, exactMax });
        expect(p50).toBeLessThanOrEqual(20);
        expect(p95).toBeLessThanOrEqual(60);
        expect(exactMax).toBeLessThan(1);
    }, 30_000);
});
