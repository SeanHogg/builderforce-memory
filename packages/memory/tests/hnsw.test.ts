/**
 * tests/hnsw.test.ts — EVM-1: pure-TS HNSW ANN index + denseSearch fast path.
 */

import { HnswIndex, denseSearch } from '../src/retrieval/hnsw.js';

/** Deterministic unit vector pointing at angle θ in a 2D plane embedded in dim D. */
function unit(theta: number, dim = 8): Float32Array {
    const v = new Float32Array(dim);
    v[0] = Math.cos(theta);
    v[1] = Math.sin(theta);
    return v;
}

/** Brute-force exact top-k cosine for an oracle. */
function exactTopK(query: Float32Array, items: Array<{ id: string; vector: Float32Array }>, k: number): string[] {
    const cos = (a: Float32Array, b: Float32Array) => {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) { dot += a[i]! * b[i]!; na += a[i]! ** 2; nb += b[i]! ** 2; }
        return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
    };
    return items.map(it => ({ id: it.id, s: cos(query, it.vector) }))
        .sort((a, b) => b.s - a.s).slice(0, k).map(h => h.id);
}

describe('HnswIndex', () => {
    it('returns empty before anything is added', () => {
        const idx = new HnswIndex();
        expect(idx.size).toBe(0);
        expect(idx.search(unit(0), 5)).toEqual([]);
    });

    it('finds the nearest neighbour on a ring of vectors', () => {
        const idx = new HnswIndex({ seed: 1 });
        const N = 64;
        for (let i = 0; i < N; i++) idx.add(`p${i}`, unit((2 * Math.PI * i) / N));
        expect(idx.size).toBe(N);
        expect(idx.has('p0')).toBe(true);

        // Query very close to p0's angle → p0 should be the top hit.
        const hits = idx.search(unit(0.001), 3);
        expect(hits[0]!.id).toBe('p0');
        expect(hits[0]!.score).toBeGreaterThan(0.99);
        expect(hits).toHaveLength(3);
    });

    it('re-adding a known id is a no-op', () => {
        const idx = new HnswIndex();
        idx.add('a', unit(0));
        idx.add('a', unit(1));
        expect(idx.size).toBe(1);
    });

    it('clear() empties the index', () => {
        const idx = new HnswIndex();
        idx.add('a', unit(0));
        idx.clear();
        expect(idx.size).toBe(0);
        expect(idx.search(unit(0), 1)).toEqual([]);
    });

    it('high recall vs exact search on a random set', () => {
        const idx = new HnswIndex({ seed: 7, M: 8, efConstruction: 100 });
        const items: Array<{ id: string; vector: Float32Array }> = [];
        // Deterministic pseudo-random vectors.
        let s = 12345;
        const rnd = () => ((s = (Math.imul(1103515245, s) + 12345) >>> 0) / 0x1_0000_0000);
        for (let i = 0; i < 400; i++) {
            const v = new Float32Array(16);
            for (let d = 0; d < 16; d++) v[d] = rnd() * 2 - 1;
            items.push({ id: `v${i}`, vector: v });
            idx.add(`v${i}`, v);
        }
        const query = items[0]!.vector;
        const got = new Set(idx.search(query, 10, 100).map(h => h.id));
        const want = exactTopK(query, items, 10);
        const overlap = want.filter(id => got.has(id)).length;
        // ANN: expect strong recall, and the exact nearest (itself) must be found.
        expect(got.has('v0')).toBe(true);
        expect(overlap).toBeGreaterThanOrEqual(8);
    });
});

describe('denseSearch fast path', () => {
    const items = Array.from({ length: 10 }, (_, i) => ({ id: `p${i}`, vector: unit((Math.PI * i) / 10) }));

    it('returns empty for empty items or k<=0', () => {
        expect(denseSearch(unit(0), [], 5)).toEqual([]);
        expect(denseSearch(unit(0), items, 0)).toEqual([]);
    });

    it('uses exact scan below threshold (default)', () => {
        const hits = denseSearch(unit(0.01), items, 3);
        expect(hits[0]!.id).toBe('p0');
        expect(hits).toHaveLength(3);
    });

    it('uses HNSW at/above threshold and still finds the nearest', () => {
        // threshold 5 → 10 items triggers the ANN path; nearest must still be p0.
        const hits = denseSearch(unit(0.01), items, 3, 5, { seed: 3, efConstruction: 50 });
        expect(hits[0]!.id).toBe('p0');
    });
});
