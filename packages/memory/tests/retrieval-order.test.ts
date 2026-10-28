/**
 * tests/retrieval-order.test.ts — the ONE total order (score desc, then id asc) and
 * its migration into every ranker, plus the insertion-order-free HNSW.
 */

import {
    compareHits,
    compareDenseHits,
    compareScoreThenId,
    quantizeScore,
    DENSE_SCORE_QUANTUM,
} from '../src/retrieval/order.js';
import { bm25Search } from '../src/retrieval/bm25.js';
import { reciprocalRankFusion } from '../src/retrieval/fusion.js';
import { fuseHybridArms } from '../src/retrieval/weightedFusion.js';
import { maximalMarginalRelevance, mmrSelect, textMaximalMarginalRelevance } from '../src/retrieval/mmr.js';
import { HnswIndex, denseSearch, levelOf } from '../src/retrieval/hnsw.js';
import { hybridRetrieve } from '../src/retrieval/HybridRetriever.js';
import { SemanticRecall } from '../src/memory/SemanticRecall.js';
import { MemoryVectorStore } from '../src/vectorstore/MemoryVectorStore.js';
import type { MemoryEntry } from '../src/memory/types.js';

const unit = (theta: number, dim = 8): Float32Array => {
    const v = new Float32Array(dim);
    v[0] = Math.cos(theta);
    v[1] = Math.sin(theta);
    return v;
};

/** Seeded PRNG for fixtures. */
function rng(seed: number): () => number {
    let s = seed >>> 0 || 1;
    return () => ((s = (Math.imul(1103515245, s) + 12345) >>> 0) / 0x1_0000_0000);
}

function shuffled<T>(items: readonly T[], seed: number): T[] {
    const r = rng(seed);
    const out = [...items];
    for (let i = out.length - 1; i > 0; i--) {
        const j = Math.floor(r() * (i + 1));
        [out[i], out[j]] = [out[j]!, out[i]!];
    }
    return out;
}

describe('compareHits', () => {
    it('orders by score descending, then id ascending', () => {
        const hits = [
            { id: 'b', score: 0.5 },
            { id: 'c', score: 0.9 },
            { id: 'a', score: 0.5 },
        ];
        expect([...hits].sort(compareHits).map((h) => h.id)).toEqual(['c', 'a', 'b']);
    });

    it('treats scores inside one quantum as tied', () => {
        const a = { id: 'a', score: 0.5 };
        const b = { id: 'b', score: 0.5 + 1e-9 };
        expect(compareHits(b, a)).toBeGreaterThan(0); // tied → id decides, a first
        expect(compareDenseHits({ id: 'z', score: 0.70001 }, { id: 'y', score: 0.70002 })).toBeGreaterThan(0);
        expect(compareHits({ id: 'z', score: 0.7 }, { id: 'y', score: 0.70002 })).toBeGreaterThan(0); // 2e-5 > 1e-6
        expect(compareHits({ id: 'y', score: 0.70002 }, { id: 'z', score: 0.7 })).toBeLessThan(0);
    });

    it('sorts NaN last and compares ids by code unit', () => {
        expect(quantizeScore(Number.NaN)).toBe(-Infinity);
        expect(compareScoreThenId(Number.NaN, 'a', 0, 'b')).toBeGreaterThan(0);
        expect(compareScoreThenId(1, 'B', 1, 'a')).toBeLessThan(0); // 'B' (66) < 'a' (97)
        expect(quantizeScore(0.12346, DENSE_SCORE_QUANTUM)).toBe(1235);
    });
});

describe('every ranker breaks equal scores by id ascending', () => {
    it('bm25Search', () => {
        const docs = ['d', 'b', 'c', 'a'].map((id) => ({ id, text: 'alpha beta' }));
        expect(bm25Search('alpha', docs).map((h) => h.id)).toEqual(['a', 'b', 'c', 'd']);
    });

    it('reciprocalRankFusion', () => {
        const fused = reciprocalRankFusion([{ ids: ['y', 'x'] }, { ids: ['x', 'y'] }]);
        expect(fused.map((h) => h.id)).toEqual(['x', 'y']);
    });

    it('fuseHybridArms', () => {
        const out = fuseHybridArms({
            vector: [{ id: 'q' }, { id: 'p' }],
            text: [],
            vectorScoreOf: () => 0.5,
            textScoreOf: () => 0,
            merge: (row) => row.id,
        });
        expect(out).toEqual(['p', 'q']);
    });

    it('mmrSelect / maximalMarginalRelevance / textMaximalMarginalRelevance', () => {
        const picked = mmrSelect([{ id: 'n' }, { id: 'm' }], { relevance: () => 1, similarity: () => 0, idOf: (x) => x.id });
        expect(picked.map((x) => x.id)).toEqual(['m', 'n']);
        const cands = [{ id: 'k2', vector: unit(0) }, { id: 'k1', vector: unit(0) }];
        expect(maximalMarginalRelevance(unit(0), cands, 1)).toEqual(['k1']);
        const text = [
            { id: 't2', score: 1, content: 'same words' },
            { id: 't1', score: 1, content: 'same words' },
        ];
        expect(textMaximalMarginalRelevance(text, 1).map((i) => i.id)).toEqual(['t1', 't2']);
        expect(textMaximalMarginalRelevance(text, 0.5)[0]!.id).toBe('t1');
    });

    it('denseSearch, exact scan and HNSW path', () => {
        const items = ['p3', 'p1', 'p2'].map((id) => ({ id, vector: unit(0.3) }));
        expect(denseSearch(unit(0.3), items, 3).map((h) => h.id)).toEqual(['p1', 'p2', 'p3']);
        expect(denseSearch(unit(0.3), items, 3, 1, { efConstruction: 16 }).map((h) => h.id)).toEqual(['p1', 'p2', 'p3']);
    });

    it('HnswIndex.search', () => {
        const idx = new HnswIndex({ M: 4, efConstruction: 16 });
        for (const id of ['h4', 'h2', 'h3', 'h1']) idx.add(id, unit(1));
        expect(idx.search(unit(1), 4).map((h) => h.id)).toEqual(['h1', 'h2', 'h3', 'h4']);
    });

    it('hybridRetrieve', () => {
        const cands = ['r3', 'r1', 'r2'].map((id) => ({ id, text: 'gamma delta' }));
        expect(hybridRetrieve({ text: 'gamma' }, cands, { topK: 3 }).map((h) => h.id)).toEqual(['r1', 'r2', 'r3']);
    });

    it('SemanticRecall (lexical path)', async () => {
        const entries: MemoryEntry[] = ['s3', 's1', 's2'].map((key) => ({ key, content: 'same fact', timestamp: 1 }));
        const recall = new SemanticRecall({ annThreshold: 256, embedCacheMax: 16, minRelevance: 0 });
        const hits = await recall.similar(entries, 'same fact', 3);
        expect(hits.map((h) => h.entry.key)).toEqual(['s1', 's2', 's3']);
    });

    it('MemoryVectorStore (exact scan)', async () => {
        const store = new MemoryVectorStore();
        await store.upsert(['v3', 'v1', 'v2'].map((id) => ({ id, text: id, vector: unit(0.2), metadata: {} })));
        const hits = await store.query({ vector: unit(0.2), topK: 3 });
        expect(hits.map((h) => h.id)).toEqual(['v1', 'v2', 'v3']);
    });
});

describe('insertion-order-free HNSW', () => {
    it('levelOf is a pure function of the id with a geometric distribution', () => {
        expect(levelOf('node-1', 16)).toBe(levelOf('node-1', 16));
        let upper = 0;
        for (let i = 0; i < 4000; i++) if (levelOf(`n${i}`, 16) > 0) upper++;
        // P(level ≥ 1) = 1/16 → ~250 of 4000.
        expect(upper).toBeGreaterThan(150);
        expect(upper).toBeLessThan(350);
    });

    it('canonicalBuild gives the same graph for any input order', () => {
        const r = rng(99);
        const items = Array.from({ length: 300 }, (_, i) => {
            const v = new Float32Array(12);
            for (let d = 0; d < 12; d++) v[d] = r() * 2 - 1;
            return { id: `c${i}`, vector: v };
        });
        const graphs = [1, 2, 3].map((seed) =>
            JSON.stringify(HnswIndex.canonicalBuild(shuffled(items, seed), { M: 8, efConstruction: 40 }).graph()));
        expect(graphs[1]).toBe(graphs[0]);
        expect(graphs[2]).toBe(graphs[0]);
    });
});
