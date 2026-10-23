/**
 * tests/retrieval-signals.test.ts — importance weighting and half-life recency
 * decay applied AFTER rank fusion in `hybridRetrieve`, and carried through
 * `MemoryStore.recallRanked`.
 *
 * Pinned: both signals are OFF by default (the fused ordering is unchanged), the
 * importance factor is `0.5 + importance` (the cloud recall's factor), recency
 * uses the shared half-life helper, and MMR cannot undo the weighting.
 */

import 'fake-indexeddb/auto';
import { hybridRetrieve, type RetrievalCandidate } from '../src/retrieval/HybridRetriever.js';
import { MemoryStore } from '../src/memory/MemoryStore.js';

const DAY = 86_400_000;
const NOW = 1_800_000_000_000;

// Two docs that tie on BM25 for the query; only the signals can separate them.
const tied = (a: Partial<RetrievalCandidate>, b: Partial<RetrievalCandidate>): RetrievalCandidate[] => [
    { id: 'a', text: 'deploy pipeline notes', ...a },
    { id: 'b', text: 'deploy pipeline notes', ...b },
];

/** Each id's score with no signals — RRF still separates exact BM25 ties by rank. */
const baseScores = (cands: RetrievalCandidate[]) =>
    new Map(hybridRetrieve({ text: 'deploy pipeline' }, cands, { topK: cands.length }).map(h => [h.id, h.score]));

test('signals are off by default — the fused ordering is unchanged', () => {
    const cands = tied({ importance: 0, timestamp: NOW - 400 * DAY }, { importance: 1, timestamp: NOW });
    const plain = hybridRetrieve({ text: 'deploy pipeline' }, cands, { topK: 2 });
    const explicitOff = hybridRetrieve({ text: 'deploy pipeline' }, cands, {
        topK: 2,
        importanceWeight: false,
        recencyHalfLifeDays: 0,
        now: NOW,
    });
    expect(explicitOff).toEqual(plain);
});

test('importance multiplies the fused score by 0.5 + importance', () => {
    const cands = tied({ importance: 0 }, { importance: 1 });
    const base = baseScores(cands);
    const hits = hybridRetrieve({ text: 'deploy pipeline' }, cands, { topK: 2, importanceWeight: true });
    expect(hits.map(h => h.id)).toEqual(['b', 'a']);
    expect(hits[0]!.score).toBeCloseTo(base.get('b')! * 1.5, 12);
    expect(hits[1]!.score).toBeCloseTo(base.get('a')! * 0.5, 12);
});

test('missing importance counts as 0.5 (factor 1); out-of-range importance is clamped', () => {
    const cands = tied({}, { importance: 7 });
    const base = baseScores(cands);
    const hits = hybridRetrieve({ text: 'deploy pipeline' }, cands, { topK: 2, importanceWeight: true });
    const byId = new Map(hits.map(h => [h.id, h.score]));
    expect(hits[0]!.id).toBe('b');
    expect(byId.get('b')!).toBeCloseTo(base.get('b')! * 1.5, 12);
    expect(byId.get('a')!).toBeCloseTo(base.get('a')!, 12);
});

test('recency decay halves a score one half-life old; undated candidates do not decay', () => {
    const cands: RetrievalCandidate[] = [
        ...tied({ timestamp: NOW - 30 * DAY }, { timestamp: NOW }),
        { id: 'c', text: 'deploy pipeline notes' },
    ];
    const base = baseScores(cands);
    const hits = hybridRetrieve({ text: 'deploy pipeline' }, cands, { topK: 3, recencyHalfLifeDays: 30, now: NOW });
    const byId = new Map(hits.map(h => [h.id, h.score]));
    expect(hits[hits.length - 1]!.id).toBe('a');
    expect(byId.get('a')!).toBeCloseTo(base.get('a')! * 0.5, 12);
    expect(byId.get('b')!).toBeCloseTo(base.get('b')!, 12);
    expect(byId.get('c')!).toBeCloseTo(base.get('c')!, 12);
});

test('recency defaults its clock to Date.now()', () => {
    const cands = tied({ timestamp: Date.now() - 365 * DAY }, { timestamp: Date.now() });
    const hits = hybridRetrieve({ text: 'deploy pipeline' }, cands, { topK: 2, recencyHalfLifeDays: 30 });
    expect(hits[0]!.id).toBe('b');
});

test('MMR relevance carries the signal factor, so the rerank cannot undo it', () => {
    const v = new Float32Array([1, 0]);
    const cands = tied({ vector: v, importance: 0 }, { vector: v, importance: 1 });
    const hits = hybridRetrieve({ text: 'deploy pipeline', vector: v }, cands, { topK: 2, importanceWeight: true });
    expect(hits[0]!.id).toBe('b');
});

test('MemoryStore.recallRanked passes importance and write time to the retriever', async () => {
    const store = new MemoryStore({ dbName: `signals-${Date.now()}` });
    await store.remember('low', 'deploy pipeline notes', { importance: 0.1, timestamp: NOW });
    await store.remember('high', 'deploy pipeline notes', { importance: 0.9, timestamp: NOW });
    await store.remember('old', 'deploy pipeline notes', { importance: 0.9, timestamp: NOW - 365 * DAY });

    const ranked = await store.recallRanked('deploy pipeline', 3, undefined, {
        importanceWeight: true,
        recencyHalfLifeDays: 90,
        now: NOW,
    });
    expect(ranked.hits.map(h => h.entry.key)).toEqual(['high', 'low', 'old']);
});
