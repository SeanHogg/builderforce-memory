/**
 * tests/knowledge-index.test.ts — the knowledge index's contract: explicit tiers,
 * the exact tier, serialize round-trip, re-embed on a fingerprint change,
 * `expand`, `similarPairs`, the tokenizer and the sha-256 it names its state with.
 */

import {
    createKnowledgeIndex,
    loadKnowledgeIndex,
    sha256Hex,
    Sha256,
    type GraphNeighbours,
    type IndexHit,
} from '../src/retrieval/knowledge-index/index.js';
import { codeTokenize } from '../src/retrieval/knowledge-index/tokenize.js';
import { makeDocs, makeEmbedder, makeQueries, shuffled } from './knowledge-index.fixture.js';

const FAST = { m: 8, efConstruction: 48 };

describe('sha256', () => {
    it('matches the FIPS 180-4 test vectors, incrementally too', () => {
        expect(sha256Hex('')).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
        expect(sha256Hex('abc')).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
        expect(sha256Hex('abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'))
            .toBe('248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1');
        expect(sha256Hex('a'.repeat(1_000_000))).toBe('cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0');
        const h = new Sha256();
        for (const part of ['ab', 'c']) h.update(part);
        expect(h.hex()).toBe(sha256Hex('abc'));
    });
});

describe('codeTokenize', () => {
    it('splits paths, snake_case and camelCase, and keeps the compound', () => {
        const tokens = codeTokenize('api/src/retryPolicy.ts stripe_webhook parseHTTPResponse');
        for (const t of ['api', 'src', 'retry', 'policy', 'retrypolicy', 'ts', 'stripe', 'webhook', 'stripe_webhook', 'parse', 'http', 'response', 'parsehttpresponse']) {
            expect(tokens).toContain(t);
        }
    });
});

describe('tiers', () => {
    it('with no embedder only exact and lexical tiers appear', async () => {
        const index = createKnowledgeIndex({ params: FAST });
        await index.upsert(makeDocs(200));
        await index.compact();
        expect(index.tiers()).toEqual(['exact', 'lexical']);
        for (const q of makeQueries(10)) {
            const hits = await index.search(q, { k: 10 });
            expect(hits.length).toBeGreaterThan(0);
            for (const h of hits) expect(['exact', 'lexical']).toContain(h.tier);
        }
        expect(await index.search('term1 term2', { tiers: ['dense'] })).toEqual([]);
        expect(index.similarPairs({ minSimilarity: 0, maxPairs: 10 })).toEqual([]);
    });

    it('the exact tier is case-folded, scores 1.0 and returns alone when it fills k', async () => {
        const embedder = makeEmbedder(16);
        const index = createKnowledgeIndex({ embedder, params: FAST });
        await index.upsert(makeDocs(50));
        expect(await index.search('SRC/Area3/FILE3.TS', { k: 1 })).toEqual([{ id: 'doc00003', score: 1, tier: 'exact' }]);
        const mixed = await index.search('file4.ts', { k: 5 });
        expect(mixed[0]).toEqual({ id: 'doc00004', score: 1, tier: 'exact' });
        expect(mixed.slice(1).every((h) => h.tier !== 'exact' && h.score < 1)).toBe(true);
    });

    it('skips unchanged hashes on upsert', async () => {
        const embedder = makeEmbedder(16);
        const index = createKnowledgeIndex({ embedder, params: FAST });
        const docs = makeDocs(30);
        await index.upsert(docs);
        const calls = embedder.calls;
        const digest = index.digest();
        await index.upsert(docs);
        expect(embedder.calls).toBe(calls);
        expect(index.digest()).toBe(digest);
    });
});

describe('serialize', () => {
    it('round-trips to the same digest, bytes and results', async () => {
        const embedder = makeEmbedder(24);
        const docs = makeDocs(320);
        const index = createKnowledgeIndex({ embedder, params: FAST });
        await index.upsert(docs.slice(0, 300));
        await index.compact();
        await index.upsert(docs.slice(300)); // pending, not yet in the graph
        index.remove(['doc00001', 'doc00002', 'doc00310']);
        const bytes = index.serialize();
        const loaded = loadKnowledgeIndex(bytes, makeEmbedder(24));
        expect(loaded.digest()).toBe(index.digest());
        expect(loaded.serialize()).toEqual(bytes);
        expect(loaded.size()).toBe(317);
        for (const q of makeQueries(10, 3)) expect(await loaded.search(q, { k: 8 })).toEqual(await index.search(q, { k: 8 }));
    });

    it('a different fingerprint drops the dense tier until compact() re-embeds', async () => {
        const index = createKnowledgeIndex({ embedder: makeEmbedder(24), params: FAST });
        await index.upsert(makeDocs(120));
        await index.compact();
        const other = makeEmbedder(24, 'another-model/2/24');
        const loaded = loadKnowledgeIndex(index.serialize(), other);
        expect(loaded.tiers()).toEqual(['exact', 'lexical']);
        expect(await loaded.search('term5 term9', { tiers: ['dense'] })).toEqual([]);
        await loaded.compact();
        expect(loaded.tiers()).toEqual(['exact', 'lexical', 'dense']);
        expect(other.calls).toBe(120);
        expect(loaded.digest()).not.toBe(index.digest());
    });
});

describe('expand', () => {
    const edges: Record<string, Array<{ id: string; kind: string; weight: number }>> = {
        a: [{ id: 'd', kind: 'mentions', weight: 0.4 }, { id: 'c', kind: 'cites', weight: 0.8 }],
        b: [{ id: 'c', kind: 'cites', weight: 1 }, { id: 'a', kind: 'cites', weight: 1 }],
        c: [{ id: 'e', kind: 'cites', weight: 0.5 }],
    };
    const graph: GraphNeighbours = {
        neighbours: (id, kinds) => (edges[id] ?? []).filter((e) => !kinds || kinds.includes(e.kind)),
    };
    const hits: IndexHit[] = [{ id: 'b', score: 0.5, tier: 'lexical' }, { id: 'a', score: 1, tier: 'exact' }];
    const index = createKnowledgeIndex();

    it('scores neighbours as hit × weight × 0.5 per hop, best path wins, compareHits order', () => {
        const one = index.expand(hits, graph);
        expect(one.map((h) => h.id)).toEqual(['a', 'b', 'c', 'd']);
        expect(one[0]).toEqual({ id: 'a', score: 1, tier: 'exact' }); // an original hit keeps its score
        expect(one[2]!.score).toBeCloseTo(0.4); // via a (1 × 0.8 × 0.5) beats via b (0.25)
        expect(one[2]!.tier).toBe('exact');
        expect(one[3]!.score).toBeCloseTo(0.2);

        const two = index.expand(hits, graph, { hops: 2 });
        expect(two.map((h) => h.id)).toEqual(['a', 'b', 'c', 'd', 'e']);
        expect(two[4]!.score).toBeCloseTo(0.1); // 0.4 × 0.5 × 0.5
    });

    it('follows only the chosen edge kinds and ignores the graph listing order', () => {
        expect(index.expand(hits, graph, { edgeKinds: ['cites'] }).map((h) => h.id)).toEqual(['a', 'b', 'c']);
        const reversed: GraphNeighbours = { neighbours: (id, kinds) => [...graph.neighbours(id, kinds)].reverse() };
        expect(index.expand([...hits].reverse(), reversed, { hops: 2 })).toEqual(index.expand(hits, graph, { hops: 2 }));
    });
});

describe('similarPairs', () => {
    it('finds near-duplicates in a deterministic order', async () => {
        const docs = [...makeDocs(300, 41)];
        for (let i = 0; i < 6; i++) {
            const text = `src/dup${i}/twin.ts term${i} term${i + 50} term${i + 100} term${i + 150}`;
            docs.push({ id: `dup${i}a`, text }, { id: `dup${i}b`, text });
        }
        const runs = [];
        for (const seed of [4, 5]) {
            const index = createKnowledgeIndex({ embedder: makeEmbedder(32), params: FAST });
            await index.upsert(shuffled(docs, seed));
            await index.compact();
            runs.push(index);
        }
        const [x, y] = runs;
        const pairs = x!.similarPairs({ minSimilarity: 0.99, maxPairs: 50 });
        expect(y!.similarPairs({ minSimilarity: 0.99, maxPairs: 50 })).toEqual(pairs);
        for (let i = 0; i < 6; i++) expect(pairs.some((p) => p.a === `dup${i}a` && p.b === `dup${i}b`)).toBe(true);
        for (const p of pairs) expect(p.a < p.b).toBe(true);
        for (let i = 1; i < pairs.length; i++) {
            const prev = pairs[i - 1]!;
            const cur = pairs[i]!;
            expect(Math.round(prev.similarity / 1e-4) >= Math.round(cur.similarity / 1e-4)).toBe(true);
        }
        expect(x!.similarPairs({ minSimilarity: 0.99, maxPairs: 50, idPrefix: 'dup1' }))
            .toEqual([expect.objectContaining({ a: 'dup1a', b: 'dup1b' })]);
        expect(x!.similarPairs({ minSimilarity: 0.99, maxPairs: 2 })).toEqual(pairs.slice(0, 2));
    }, 30_000);
});
