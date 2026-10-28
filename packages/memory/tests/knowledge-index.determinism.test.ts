/**
 * tests/knowledge-index.determinism.test.ts — the knowledge index's determinism and
 * storage done-checks: order-free builds, integer traversal under float noise, lean
 * mode recall, and the int8 + CSR storage budget.
 */

import { createKnowledgeIndex, type DeterministicKnowledgeIndexType, type IndexDoc, type IndexParams } from '../src/retrieval/knowledge-index/index.js';
import { fitScale, quantize, toUnit } from '../src/retrieval/knowledge-index/quantize.js';
import { makeDocs, makeEmbedder, makeQueries, shuffled, unitVector } from './knowledge-index.fixture.js';

const FAST: Partial<IndexParams> = { m: 8, efConstruction: 48 };

async function build(
    docs: readonly IndexDoc[],
    embedder: ReturnType<typeof makeEmbedder>,
    params: Partial<IndexParams>,
    seed?: number,
): Promise<DeterministicKnowledgeIndexType> {
    const index = createKnowledgeIndex({ embedder, params });
    const order = seed === undefined ? [...docs] : shuffled(docs, seed);
    for (let i = 0; i < order.length; i += 137) await index.upsert(order.slice(i, i + 137));
    await index.compact();
    return index;
}

describe('order-free builds', () => {
    it('3 shuffled insertion orders → the same digest and the same results for 50 queries', async () => {
        const docs = makeDocs(2000);
        const queries = makeQueries(50);
        const built = [];
        for (const seed of [1, 2, 3]) built.push(await build(docs, makeEmbedder(32), FAST, seed));
        const [a, b, c] = built as [DeterministicKnowledgeIndexType, DeterministicKnowledgeIndexType, DeterministicKnowledgeIndexType];
        expect(a.tiers()).toEqual(['exact', 'lexical', 'dense']);
        expect(b.digest()).toBe(a.digest());
        expect(c.digest()).toBe(a.digest());
        let dense = 0;
        for (const q of queries) {
            const ra = await a.search(q, { k: 10 });
            expect(await b.search(q, { k: 10 })).toEqual(ra);
            expect(await c.search(q, { k: 10 })).toEqual(ra);
            dense += ra.filter((h) => h.tier === 'dense').length;
        }
        expect(dense).toBeGreaterThan(0);
    }, 120_000);
});

describe('integer traversal', () => {
    it('vectors that differ by float noise below quantization give identical candidates', async () => {
        const dims = 32;
        const docs = makeDocs(400, 3);
        const queries = makeQueries(20, 5);
        const unitA = (text: string) => toUnit(Float32Array.from(unitVector(text, dims)));
        const scale = fitScale(docs.map((d) => unitA(d.text)));
        let maxAbs = 0;
        for (const d of docs) for (const x of unitA(d.text)) maxAbs = Math.max(maxAbs, Math.abs(x));

        // Move each component 5% of the way to its code's centre: a different float,
        // the same int8 code. The largest components stay put so the scale cannot move.
        const noisy = (u: Float64Array): Float64Array => Float64Array.from(Float32Array.from(u), (x) => {
            if (Math.abs(x) >= 0.9 * maxAbs) return x;
            const centre = Math.round(x * scale) / scale;
            return x + 0.05 * (centre - x);
        });
        const embA = makeEmbedder(dims, 'fixture/noise');
        const embB = makeEmbedder(dims, 'fixture/noise', noisy);

        // The fixture really is "same codes, different floats".
        const unitB = (text: string) => toUnit(Float32Array.from(noisy(unitVector(text, dims))));
        for (const text of [...docs.map((d) => d.text), ...queries]) {
            expect(Array.from(unitB(text))).not.toEqual(Array.from(unitA(text)));
            expect(quantize(unitB(text), scale)).toEqual(quantize(unitA(text), scale));
        }
        expect(fitScale(docs.map((d) => unitB(d.text)))).toBe(scale);

        const a = await build(docs, embA, FAST);
        const b = await build(docs, embB, FAST);
        expect(b.digest()).toBe(a.digest());
        expect(b.serialize()).toEqual(a.serialize());
        for (const q of queries) expect(await b.denseCandidates(q, 40)).toEqual(await a.denseCandidates(q, 40));
    }, 60_000);
});

describe('lean mode', () => {
    it('recall@10 stays within 2 points of full mode, with fewer stored codes', async () => {
        const dims = 32;
        const docs = makeDocs(1000, 21);
        const queries = makeQueries(50, 23);
        const full = await build(docs, makeEmbedder(dims), FAST);
        const lean = await build(docs, makeEmbedder(dims), { ...FAST, lean: true });

        const truth = (q: string): Set<string> => {
            const qv = unitVector(q, dims);
            const scored = docs.map((d) => {
                const v = unitVector(d.text, dims);
                let dot = 0;
                for (let i = 0; i < dims; i++) dot += qv[i]! * v[i]!;
                return { id: d.id, dot };
            });
            scored.sort((x, y) => y.dot - x.dot || (x.id < y.id ? -1 : 1));
            return new Set(scored.slice(0, 10).map((s) => s.id));
        };
        const recall = async (index: DeterministicKnowledgeIndexType): Promise<number> => {
            let found = 0;
            for (const q of queries) {
                const want = truth(q);
                const got = await index.search(q, { k: 10, tiers: ['dense'] });
                found += got.filter((h) => want.has(h.id)).length;
            }
            return found / (queries.length * 10);
        };
        const fullRecall = await recall(full);
        const leanRecall = await recall(lean);
        expect(Math.abs(fullRecall - leanRecall)).toBeLessThanOrEqual(0.02);
        expect(lean.storage().codeBytes).toBeLessThan(full.storage().codeBytes / 4);
        expect(lean.digest()).not.toBe(full.digest()); // params differ, so the state does
    }, 60_000);
});

describe('storage', () => {
    it('int8 codes + CSR graph take at most 30% of float32 vectors', async () => {
        const index = await build(makeDocs(500, 31), makeEmbedder(384), { efConstruction: 64 });
        const s = index.storage();
        expect(s.nodes).toBe(500);
        expect(s.codeBytes).toBe(500 * 384);
        expect((s.codeBytes + s.graphBytes) / s.float32Bytes).toBeLessThanOrEqual(0.3);
    }, 60_000);
});
