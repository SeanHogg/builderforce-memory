/**
 * tests/knowledge-index.exact.test.ts — exact mode (fewer than `exactBelow` cards, no
 * graph): recall against a brute-force float reference, an order-free digest, the
 * threshold, exact-mode pairs (all pairs and blocked), serialization, and loading a
 * version-1 blob (every v1 index is a graph).
 */

import {
    DEFAULT_INDEX_PARAMS,
    EXACT_PAIR_BUDGET,
    hashDocText,
    loadKnowledgeIndex,
    type DeterministicKnowledgeIndexType,
    type EmbedderPort,
    type IndexDoc,
} from '../src/retrieval/knowledge-index/index.js';
import { compareDenseHits } from '../src/retrieval/order.js';
import { cosineSimilarity } from '../src/similarity/index.js';
import { fitScale, quantize, toUnit } from '../src/retrieval/knowledge-index/quantize.js';
import { canonicalKeys } from '../src/retrieval/knowledge-index/state.js';
import { ByteWriter } from '../src/retrieval/knowledge-index/bytes.js';
import { writeGraph } from '../src/retrieval/knowledge-index/digest.js';
import { buildIndex, makeDocs, makeEmbedder, makeQueries } from './knowledge-index.fixture.js';

const FAST = { m: 8, efConstruction: 48 };

/** Top `k` ids by float cosine over every card — the engine's own vectors and order. */
async function bruteForce(embedder: EmbedderPort, docs: readonly IndexDoc[], q: string, k: number): Promise<string[]> {
    const vecs = (await embedder.embed(docs.map((d) => d.text))).map(toUnit);
    const qv = toUnit((await embedder.embed([q]))[0]!);
    return docs
        .map((d, i) => ({ id: d.id, score: cosineSimilarity(qv, vecs[i]!) }))
        .sort(compareDenseHits)
        .slice(0, k)
        .map((h) => h.id);
}

describe('exact mode', () => {
    it('is the default below exactBelow and builds no graph', async () => {
        expect(DEFAULT_INDEX_PARAMS.exactBelow).toBe(20000);
        const index = await buildIndex(makeDocs(300), makeEmbedder(16), FAST);
        expect(index.storage()).toEqual(expect.objectContaining({ mode: 'exact', nodes: 300, graphBytes: 0, codeBytes: 300 * 16 }));
    });

    it('dense recall@10 is 1.0 against a brute-force float reference', async () => {
        const dims = 32;
        const docs = makeDocs(1000, 21);
        const embedder = makeEmbedder(dims);
        const index = await buildIndex(docs, embedder, FAST);
        let found = 0;
        const queries = makeQueries(50, 23);
        for (const q of queries) {
            const want = new Set(await bruteForce(embedder, docs, q, 10));
            const got = await index.search(q, { k: 10, tiers: ['dense'] });
            expect(got).toHaveLength(10);
            found += got.filter((h) => want.has(h.id)).length;
        }
        expect(found / (queries.length * 10)).toBe(1);
    }, 60_000);

    it('3 shuffled insertion orders → the same digest, bytes and results', async () => {
        const docs = makeDocs(2000);
        const built: DeterministicKnowledgeIndexType[] = [];
        for (const seed of [1, 2, 3]) built.push(await buildIndex(docs, makeEmbedder(32), FAST, seed));
        const [a, b, c] = built as [DeterministicKnowledgeIndexType, DeterministicKnowledgeIndexType, DeterministicKnowledgeIndexType];
        expect(a.storage().mode).toBe('exact');
        expect(b.digest()).toBe(a.digest());
        expect(c.digest()).toBe(a.digest());
        expect(b.serialize()).toEqual(a.serialize());
        for (const q of makeQueries(20)) {
            const ra = await a.search(q, { k: 10 });
            expect(await b.search(q, { k: 10 })).toEqual(ra);
            expect(await c.search(q, { k: 10 })).toEqual(ra);
        }
    }, 60_000);

    it('switches to the graph at exactBelow, and exactBelow: 0 always builds it', async () => {
        const docs = makeDocs(100);
        const embedder = makeEmbedder(16);
        const index = await buildIndex(docs.slice(0, 99), embedder, { ...FAST, exactBelow: 100 });
        expect(index.storage().mode).toBe('exact');
        await index.upsert(docs.slice(99));
        await index.compact();
        expect(index.storage().mode).toBe('graph');
        expect(index.storage().graphBytes).toBeGreaterThan(0);
        expect((await buildIndex(docs.slice(0, 10), embedder, { ...FAST, exactBelow: 0 })).storage().mode).toBe('graph');
    });

    it('ignores lean: every code is kept', async () => {
        const index = await buildIndex(makeDocs(200), makeEmbedder(16), { ...FAST, lean: true });
        expect(index.storage()).toEqual(expect.objectContaining({ mode: 'exact', codeBytes: 200 * 16 }));
    });

    it('serialize round-trips at version 2 to the same digest, bytes and results', async () => {
        const docs = makeDocs(220);
        const index = await buildIndex(docs.slice(0, 200), makeEmbedder(24), FAST);
        await index.upsert(docs.slice(200)); // pending, not yet compacted
        index.remove(['doc00001', 'doc00210']);
        const bytes = index.serialize();
        expect(new DataView(bytes.buffer, bytes.byteOffset).getUint32(4, true)).toBe(2);
        const loaded = loadKnowledgeIndex(bytes, makeEmbedder(24));
        expect(loaded.digest()).toBe(index.digest());
        expect(loaded.serialize()).toEqual(bytes);
        expect(loaded.storage().mode).toBe('exact');
        for (const q of makeQueries(10, 3)) expect(await loaded.search(q, { k: 8 })).toEqual(await index.search(q, { k: 8 }));
    });
});

describe('exact-mode similarPairs', () => {
    const withTwins = (n: number): IndexDoc[] => {
        const docs = [...makeDocs(n, 41)];
        for (let i = 0; i < 6; i++) {
            const text = `src/dup${i}/twin.ts term${i} term${i + 50} term${i + 100} term${i + 150}`;
            docs.push({ id: `dup${i}a`, text }, { id: `dup${i}b`, text });
        }
        return docs;
    };

    // 300 cards score every pair; 800 cards exceed the budget and go through blocking.
    for (const n of [300, 800]) {
        it(`finds every twin deterministically over ${n} cards`, async () => {
            const docs = withTwins(n);
            if (n === 800) expect((docs.length * (docs.length - 1)) / 2).toBeGreaterThan(EXACT_PAIR_BUDGET);
            const x = await buildIndex(docs, makeEmbedder(32), FAST, 4);
            const y = await buildIndex(docs, makeEmbedder(32), FAST, 5);
            expect(x.storage().mode).toBe('exact');
            const pairs = x.similarPairs({ minSimilarity: 0.99, maxPairs: 50 });
            expect(y.similarPairs({ minSimilarity: 0.99, maxPairs: 50 })).toEqual(pairs);
            for (let i = 0; i < 6; i++) expect(pairs.some((p) => p.a === `dup${i}a` && p.b === `dup${i}b`)).toBe(true);
            for (const p of pairs) expect(p.a < p.b).toBe(true);
            for (let i = 1; i < pairs.length; i++) {
                expect(Math.round(pairs[i - 1]!.similarity / 1e-4) >= Math.round(pairs[i]!.similarity / 1e-4)).toBe(true);
            }
            expect(x.similarPairs({ minSimilarity: 0.99, maxPairs: 50, idPrefix: 'dup1' }))
                .toEqual([expect.objectContaining({ a: 'dup1a', b: 'dup1b' })]);
        }, 60_000);
    }
});

describe('version-1 blobs', () => {
    /** A v1 index written field by field: no `exactBelow`, no mode byte, a full 3-node graph. */
    async function v1Blob(embedder: EmbedderPort, docs: readonly IndexDoc[]): Promise<Uint8Array> {
        const units = (await embedder.embed(docs.map((d) => d.text))).map(toUnit);
        const scale = fitScale(units);
        const w = new ByteWriter();
        w.raw(Uint8Array.from([0x42, 0x46, 0x4b, 0x49])).u32(1);
        w.str(embedder.fingerprint).u32(embedder.dims).f64(scale);
        w.u32(8).u32(48).u32(64).f64(0.02).u32(4).u8(0).u32(60); // v1 params
        w.u32(docs.length);
        for (const d of docs) {
            const keys = canonicalKeys(d.keys);
            w.str(d.id).str(d.text).str(hashDocText(d.text)).u32(keys.length);
            for (const key of keys) w.str(key);
            w.u8(0);
        }
        w.u8(1).u32(docs.length);
        for (const d of docs) w.str(d.id).str(hashDocText(d.text));
        w.bytes(new Uint8Array(docs.length));
        w.bytes(Uint8Array.from(docs, () => 1));
        for (const u of units) w.bytes(quantize(u, scale));
        writeGraph(w, { entry: 0, layers: [{ offsets: Uint32Array.from([0, 2, 4, 6]), neighbours: Uint32Array.from([1, 2, 0, 2, 0, 1]) }] });
        return w.finish();
    }

    it('still loads, as a graph that keeps building a graph', async () => {
        const embedder = makeEmbedder(8);
        const docs = makeDocs(3);
        const loaded = loadKnowledgeIndex(await v1Blob(embedder, docs), embedder);
        expect(loaded.size()).toBe(3);
        expect(loaded.tiers()).toEqual(['exact', 'lexical', 'dense']);
        expect(loaded.storage()).toEqual(expect.objectContaining({ mode: 'graph', nodes: 3 }));
        expect(loaded.storage().graphBytes).toBeGreaterThan(0);
        expect(await loaded.search('file1.ts', { k: 1 })).toEqual([{ id: 'doc00001', score: 1, tier: 'exact' }]);
        const q = makeQueries(1)[0]!;
        expect((await loaded.search(q, { k: 3, tiers: ['dense'] })).map((h) => h.id)).toEqual(await bruteForce(embedder, docs, q, 3));

        const again = loadKnowledgeIndex(loaded.serialize(), embedder);
        expect(again.digest()).toBe(loaded.digest());
        const calls = embedder.calls;
        await loaded.compact();
        expect(loaded.storage().mode).toBe('graph');
        expect(embedder.calls).toBe(calls); // every v1 code is reused, nothing re-embedded
    });
});
