/**
 * tests/retrieval.test.ts
 * Chunking, BM25, rank fusion, and the HybridRetriever — the RAG retrieval layer.
 */

import { chunkText } from '../src/retrieval/chunk.js';
import { chunkedTextSummary } from '../src/working/textSummary.js';
import { bm25Search, bm25Idf, bm25LengthNorm, bm25TermScore } from '../src/retrieval/bm25.js';
import { reciprocalRankFusion } from '../src/retrieval/fusion.js';
import { maximalMarginalRelevance } from '../src/retrieval/mmr.js';
import { hybridRetrieve } from '../src/retrieval/HybridRetriever.js';

// ── chunkText ──────────────────────────────────────────────────────────────────

test('chunkText returns [] for empty / whitespace input', () => {
    expect(chunkText('')).toEqual([]);
    expect(chunkText('   \n  ')).toEqual([]);
});

test('chunkText returns a single chunk when text fits chunkSize', () => {
    const chunks = chunkText('short text', { chunkSize: 100 });
    expect(chunks).toEqual([{ text: 'short text', index: 0, start: 0 }]);
});

test('chunkText splits on paragraph boundaries with overlap', () => {
    const para = 'A'.repeat(40);
    const text = `${para}\n\n${para}\n\n${para}`;
    const chunks = chunkText(text, { chunkSize: 50, chunkOverlap: 10 });
    expect(chunks.length).toBeGreaterThan(1);
    // Indices are contiguous from 0.
    chunks.forEach((c, i) => expect(c.index).toBe(i));
    // Every chunk respects (roughly) the size budget.
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(50);
});

test('chunkText hard-splits a single oversized piece with no separators', () => {
    const text = 'x'.repeat(120);
    const chunks = chunkText(text, { chunkSize: 50, chunkOverlap: 0 });
    expect(chunks.length).toBeGreaterThanOrEqual(3);
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(50);
});

test('chunkText exhausts custom separators without a final "" splitter', () => {
    // No newline in text + separators lacking "" → splitRecursive hits sep===undefined.
    const text = 'y'.repeat(120);
    const chunks = chunkText(text, { chunkSize: 50, chunkOverlap: 5, separators: ['\n'] });
    expect(chunks.length).toBeGreaterThanOrEqual(2);
});

test('chunkText hard-splits an oversized piece with zero overlap', () => {
    // separators lacking "" → an oversized atomic piece reaches the hard-split while
    // loop; overlap 0 exercises the no-carry branch inside it.
    const chunks = chunkText('z'.repeat(120), { chunkSize: 50, chunkOverlap: 0, separators: ['\n'] });
    expect(chunks.length).toBeGreaterThanOrEqual(3);
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(50);
});

test('chunkText with zero overlap does not carry context', () => {
    const text = `${'a'.repeat(30)}\n\n${'b'.repeat(30)}`;
    const chunks = chunkText(text, { chunkSize: 35, chunkOverlap: 0 });
    expect(chunks.length).toBeGreaterThan(1);
});

const EXACT_SEPARATORS = ['\n\n', '\n', ''];

test('regression: chunkOverlap 0 round-trips the input exactly', () => {
    const text = `${'p'.repeat(5)}\n\n${'l'.repeat(4)}\n${'m'.repeat(4)}\n\n${'h'.repeat(12)}`;
    const chunks = chunkText(text, { chunkSize: 8, chunkOverlap: 0, separators: EXACT_SEPARATORS, exact: true });
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(8);
    // Nothing is trimmed or swallowed: separators stay on their chunk.
    expect(chunks.map((c) => c.text).join('')).toBe(text);
    for (const c of chunks) expect(text.slice(c.start, c.start + c.text.length)).toBe(c.text);
    chunks.forEach((c, i) => expect(c.index).toBe(i));
    // A trailing whitespace-only remainder is the one thing dropped.
    expect(chunkText(`abc\n\n${' '.repeat(10)}`, { chunkSize: 5, exact: true, separators: EXACT_SEPARATORS }).map((c) => c.text)).toEqual(['abc\n\n', ' '.repeat(5)]);
    // Whitespace-only input is still [] and the default (non-exact) output is unchanged by the new option.
    expect(chunkText('  \n ', { exact: true })).toEqual([]);
    expect(chunkText('short text', { chunkSize: 100, exact: true })).toEqual([{ text: 'short text', index: 0, start: 0 }]);
});

test('regression: a surrogate pair is never split', () => {
    const text = '😀'.repeat(9); // 18 code units
    for (const max of [1, 2, 3, 5, 7]) {
        for (const exact of [true, false]) {
            const chunks = chunkText(text, { chunkSize: max, chunkOverlap: 0, exact });
            expect(chunks.map((c) => c.text).join('')).toBe(text);
            for (const c of chunks) expect(/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/.test(c.text)).toBe(false);
        }
    }
});

test('regression: a surrogate pair is never split (overlap carry lands mid-pair)', () => {
    const text = '😀'.repeat(20);
    for (const [size, overlap] of [[7, 3], [9, 5], [10, 1], [6, 3]] as const) {
        const chunks = chunkText(text, { chunkSize: size, chunkOverlap: overlap });
        expect(chunks.length).toBeGreaterThan(1);
        for (const c of chunks) expect(/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/.test(c.text)).toBe(false);
    }
});

test('regression: overlap = chunkSize-1 on astral text terminates and never splits a pair', () => {
    // Distinct astral characters (2 code units each) so order and coverage are checkable by position.
    const distinct = Array.from({ length: 200 }, (_, i) => String.fromCodePoint(0x1f600 + i)).join('');
    const same = '😀'.repeat(20);
    const cases: Array<[string, { chunkSize: number; chunkOverlap?: number }]> = [
        [same, { chunkSize: 7, chunkOverlap: 6 }],
        [same, { chunkSize: 2, chunkOverlap: 1 }],
        [same, { chunkSize: 3, chunkOverlap: 2 }],
        [distinct, { chunkSize: 7, chunkOverlap: 6 }],
        [distinct, { chunkSize: 3, chunkOverlap: 2 }],
        [distinct, { chunkSize: 50 }], // default overlap 200 clamps to chunkSize - 1
    ];
    for (const [text, opts] of cases) {
        const chunks = chunkText(text, opts);
        expect(chunks.length).toBeGreaterThan(1);
        for (const c of chunks) {
            expect(c.text.length).toBeGreaterThan(0);
            expect(c.text.length).toBeLessThanOrEqual(opts.chunkSize);
            expect(/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/.test(c.text)).toBe(false);
            expect(c.text.length % 2).toBe(0); // only whole pairs
        }
        expect(chunks[0]!.text.startsWith(text.slice(0, 2))).toBe(true);
        expect(chunks[chunks.length - 1]!.text.endsWith(text.slice(-2))).toBe(true);
    }
    // Ordered, gap-free coverage by each chunk's own `start`.
    for (const opts of [{ chunkSize: 7, chunkOverlap: 6 }, { chunkSize: 3, chunkOverlap: 2 }, { chunkSize: 50 }]) {
        let prevStart = -1;
        let prevEnd = 0;
        for (const c of chunkText(distinct, opts)) {
            expect(distinct.slice(c.start, c.start + c.text.length)).toBe(c.text);
            expect(c.start).toBeGreaterThan(prevStart);
            expect(c.start).toBeLessThanOrEqual(prevEnd); // no gap
            expect(c.start + c.text.length).toBeGreaterThan(prevEnd);
            prevStart = c.start;
            prevEnd = c.start + c.text.length;
        }
        expect(prevEnd).toBe(distinct.length);
    }
});

test("regression: every chunk's start is its true offset", () => {
    const words = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu';
    const text = `${words}. ${'😀🎉'.repeat(15)}\n\n${'x'.repeat(60)}\n${words} ${'é'.repeat(25)}${'🚀'.repeat(12)}`;
    for (const [size, overlap] of [[10, 0], [10, 3], [10, 9], [17, 8], [30, 29], [30, 200], [64, 10], [3, 2], [2, 1]] as const) {
        const chunks = chunkText(text, { chunkSize: size, chunkOverlap: overlap });
        expect(chunks.length).toBeGreaterThan(1);
        for (const c of chunks) {
            expect(text.slice(c.start, c.start + c.text.length)).toBe(c.text);
            expect(c.text.length).toBeLessThanOrEqual(size);
        }
    }
    // The offset is into the ORIGINAL text, leading whitespace included.
    const padded = chunkText(`  \n${'ab '.repeat(40)}`, { chunkSize: 20, chunkOverlap: 5 });
    for (const c of padded) expect(`  \n${'ab '.repeat(40)}`.slice(c.start, c.start + c.text.length)).toBe(c.text);
});

test('regression: no chunk repeats the previous chunk', () => {
    const distinct = Array.from({ length: 120 }, (_, i) => String.fromCodePoint(0x1f600 + i)).join('');
    const ascii = 'abcdefghijklmnopqrstuvwxyz0123456789'.repeat(6);
    for (const text of [distinct, ascii]) {
        for (const [size, overlap] of [[7, 6], [6, 5], [10, 9], [4, 3], [3, 2], [20, 100]] as const) {
            const chunks = chunkText(text, { chunkSize: size, chunkOverlap: overlap });
            let prevEnd = 0;
            for (const c of chunks) {
                expect(c.start + c.text.length).toBeGreaterThan(prevEnd); // each chunk reaches past the previous one
                prevEnd = c.start + c.text.length;
            }
            for (let i = 1; i < chunks.length; i++) expect(chunks[i]!.text).not.toBe(chunks[i - 1]!.text);
        }
    }
});

test('regression: chunkedTextSummary pins the summarizer chunk boundaries (paragraph, line, hard cut, greedy pack)', async () => {
    const text = 'aaaaaaaa\n\n' + 'bbbb\ncccc\ndddd\n\n' + 'eeeeeeeeee\nffffffffff\ngggggggggg\n\n' + 'h'.repeat(45);
    const seen: string[] = [];
    await chunkedTextSummary({
        text,
        limit: 20,
        measure: async (t) => t.length,
        maxChunks: 10,
        summarizeChunk: async (c) => { seen.push(c); return 'S'; },
    });
    // The first eight calls are the leaf chunks of `text`; later calls are the merge rounds.
    expect(seen.slice(0, 8)).toEqual([
        'aaaaaaaa\n\n',
        'bbbb\ncccc\ndddd\n\n',
        'eeeeeeeeee\n',
        'ffffffffff\n',
        'gggggggggg\n\n',
        'h'.repeat(20),
        'h'.repeat(20),
        'hhhhh',
    ]);
});

test('regression: chunkedTextSummary uses the shared splitter', async () => {
    const text = Array.from({ length: 6 }, (_, i) => `${i}`.padEnd(30, 'x')).join('\n\n');
    const seen: string[] = [];
    await chunkedTextSummary({
        text,
        limit: 50,
        measure: async (t) => t.length,
        maxChunks: 8,
        summarizeChunk: async (c) => { seen.push(c); return 'S'; },
    });
    const expected = chunkText(text, { chunkSize: 50, exact: true, separators: EXACT_SEPARATORS }).map((c) => c.text);
    expect(expected.length).toBeGreaterThan(1);
    expect(seen.slice(0, expected.length)).toEqual(expected);
});

// ── bm25Search ─────────────────────────────────────────────────────────────────

test('bm25Search returns [] for an empty corpus', () => {
    expect(bm25Search('query', [])).toEqual([]);
});

test('bm25Search returns [] when the query has no usable terms', () => {
    expect(bm25Search('   !!! ', [{ id: '1', text: 'hello world' }])).toEqual([]);
});

test('bm25Search ranks documents by term relevance and drops non-matches', () => {
    const docs = [
        { id: 'a', text: 'the quick brown fox' },
        { id: 'b', text: 'lazy dogs sleep all day' },
        { id: 'c', text: 'a quick fox is a clever fox' },
    ];
    const hits = bm25Search('quick fox', docs);
    const ids = hits.map((h) => h.id);
    expect(ids.sort()).toEqual(['a', 'c']); // both match; 'b' has no overlap → dropped
});

test('bm25Search handles a corpus of empty-tokenizing docs (avgdl guard)', () => {
    // '!!!' tokenizes to [] → totalLen 0 → avgdl falls back to 1; query term absent
    // from every doc → no match → [].
    expect(bm25Search('alpha', [{ id: 'a', text: '!!!' }])).toEqual([]);
});

test('bm25Search honours custom k1/b options', () => {
    const docs = [{ id: 'a', text: 'alpha beta gamma' }];
    const hits = bm25Search('alpha', docs, { k1: 1.2, b: 0.5 });
    expect(hits[0]!.id).toBe('a');
    expect(hits[0]!.score).toBeGreaterThan(0);
});

test('bm25Search accepts an injected tokenizer for both query and documents', () => {
    // A stemming tokenizer: "run" in the query must reach "runs" in the doc, which
    // the plain word-split tokenizer treats as two unrelated terms.
    const stem = (text: string): string[] =>
        (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).map((t) =>
            t.endsWith('s') && !t.endsWith('ss') && t.length > 3 ? t.slice(0, -1) : t,
        );

    const docs = [
        { id: 'match', text: 'the process runs nightly' },
        { id: 'other', text: 'unrelated prose about weather' },
    ];
    expect(bm25Search('run', docs)).toEqual([]);
    expect(bm25Search('run', docs, { tokenize: stem })[0]!.id).toBe('match');
});

test('an injected tokenizer that drops everything yields no hits, not a throw', () => {
    const docs = [{ id: 'a', text: 'alpha beta' }];
    expect(bm25Search('alpha', docs, { tokenize: () => [] })).toEqual([]);
});

// ── the exported scoring core (shared with the api's BM25F ranker) ─────────────

test('bm25Idf is non-negative even for a term present in every document', () => {
    expect(bm25Idf(10, 10)).toBeGreaterThanOrEqual(0);
    expect(bm25Idf(10, 1)).toBeGreaterThan(bm25Idf(10, 9));
});

test('bm25Idf clamps a nonsense corpus rather than returning NaN', () => {
    expect(Number.isFinite(bm25Idf(0, 0))).toBe(true);
    expect(Number.isFinite(bm25Idf(5, 99))).toBe(true);
    expect(Number.isFinite(bm25Idf(5, -1))).toBe(true);
});

test('bm25LengthNorm is 1 at average length and grows with document length', () => {
    expect(bm25LengthNorm(100, 100, 0.75)).toBeCloseTo(1, 10);
    expect(bm25LengthNorm(200, 100, 0.75)).toBeGreaterThan(1);
    expect(bm25LengthNorm(50, 100, 0.75)).toBeLessThan(1);
    // A zero-length corpus must not divide by zero.
    expect(Number.isFinite(bm25LengthNorm(10, 0, 0.75))).toBe(true);
});

test('bm25TermScore saturates: doubling frequency does not double the score', () => {
    const idf = 2;
    const one = bm25TermScore(1, idf, 1, 1.5);
    const two = bm25TermScore(2, idf, 1, 1.5);
    expect(two).toBeGreaterThan(one);
    expect(two).toBeLessThan(one * 2);
    expect(bm25TermScore(0, idf, 1, 1.5)).toBe(0);
});

test('the core reproduces bm25Search exactly — one formula, two index shapes', () => {
    // What the api's BM25F ranker does: read precomputed per-term stats and score
    // them with the shared core. Over a single-term single-doc corpus it must match
    // the inline index bm25Search builds, or the two rankers have drifted.
    const docs = [{ id: 'a', text: 'alpha beta gamma' }];
    const inline = bm25Search('alpha', docs)[0]!.score;
    const fromStats = bm25TermScore(1, bm25Idf(1, 1), bm25LengthNorm(3, 3, 0.75), 1.5);
    expect(fromStats).toBeCloseTo(inline, 12);
});

// ── reciprocalRankFusion ─────────────────────────────────────────────────────

test('reciprocalRankFusion merges rankings and rewards top positions', () => {
    const fused = reciprocalRankFusion([
        { ids: ['a', 'b', 'c'] },
        { ids: ['b', 'a', 'd'] },
    ]);
    // 'a' (ranks 1,2) and 'b' (ranks 2,1) appear in both → top two.
    const top2 = fused.slice(0, 2).map((f) => f.id).sort();
    expect(top2).toEqual(['a', 'b']);
});

test('reciprocalRankFusion respects per-list weight', () => {
    const fused = reciprocalRankFusion([
        { ids: ['a'], weight: 0.1 },
        { ids: ['b'], weight: 10 },
    ]);
    expect(fused[0]!.id).toBe('b');
});

// ── maximalMarginalRelevance ─────────────────────────────────────────────────

test('mmr selects the most relevant first, then diversifies', () => {
    const q = new Float32Array([1, 0]);
    const cands = [
        { id: 'near1', vector: new Float32Array([1, 0]) },        // exact match (rel 1)
        { id: 'near2', vector: new Float32Array([0.99, 0.141]) }, // near-duplicate (rel ~0.99)
        { id: 'div', vector: new Float32Array([0, 1]) },          // orthogonal / diverse
    ];
    // λ=0.3 weights diversity heavily: after the top match, the near-duplicate is
    // penalised for redundancy and the diverse item is promoted.
    const out = maximalMarginalRelevance(q, cands, 2, 0.3);
    expect(out[0]).toBe('near1');     // most relevant picked first
    expect(out[1]).toBe('div');       // diversity beats the near-duplicate
});

test('mmr stops when candidates run out before topK', () => {
    const q = new Float32Array([1, 0]);
    const cands = [{ id: 'a', vector: new Float32Array([1, 0]) }];
    expect(maximalMarginalRelevance(q, cands, 5)).toEqual(['a']);
});

test('mmr keeps the max similarity across multiple already-selected items', () => {
    // topK=3 forces a third selection while two are already selected, exercising the
    // inner "max similarity to selected" loop (the no-update branch included).
    const q = new Float32Array([1, 0]);
    const cands = [
        { id: 'a', vector: new Float32Array([1, 0]) },
        { id: 'b', vector: new Float32Array([0, 1]) },
        { id: 'c', vector: new Float32Array([0.7, 0.7]) },
    ];
    const out = maximalMarginalRelevance(q, cands, 3, 0.5);
    expect(out.sort()).toEqual(['a', 'b', 'c']);
});

// ── hybridRetrieve ───────────────────────────────────────────────────────────

const vec = (x: number, y: number) => Float32Array.from([x, y]);

test('hybridRetrieve returns [] for no candidates', () => {
    expect(hybridRetrieve({ text: 'q' }, [])).toEqual([]);
});

test('hybridRetrieve fuses dense + sparse and reranks', () => {
    const candidates = [
        { id: 'a', text: 'quick brown fox', vector: vec(1, 0) },
        { id: 'b', text: 'slow green turtle', vector: vec(0, 1) },
        { id: 'c', text: 'quick clever fox', vector: vec(0.9, 0.1) },
    ];
    const hits = hybridRetrieve({ text: 'quick fox', vector: vec(1, 0) }, candidates, { topK: 2 });
    expect(hits.length).toBe(2);
    expect(hits.map((h) => h.id)).toContain('a');
});

test('hybridRetrieve works BM25-only when no query vector is supplied', () => {
    const candidates = [
        { id: 'a', text: 'alpha beta' },
        { id: 'b', text: 'gamma delta' },
    ];
    const hits = hybridRetrieve({ text: 'alpha' }, candidates, { topK: 5 });
    expect(hits[0]!.id).toBe('a');
});

test('hybridRetrieve returns [] when nothing matches either signal', () => {
    // No query vector AND no lexical overlap → both rankings empty → fused empty.
    const candidates = [{ id: 'a', text: 'zzz' }];
    expect(hybridRetrieve({ text: 'nomatch' }, candidates)).toEqual([]);
});

test('hybridRetrieve skips MMR when candidates have no vectors', () => {
    const candidates = [
        { id: 'a', text: 'alpha beta' },
        { id: 'b', text: 'alpha gamma' },
    ];
    // query has a vector but candidates do not → mmrCands empty → fused order kept.
    const hits = hybridRetrieve({ text: 'alpha', vector: vec(1, 0) }, candidates, { topK: 2 });
    expect(hits.length).toBe(2);
});

test('hybridRetrieve with rerank disabled keeps fused order', () => {
    const candidates = [
        { id: 'a', text: 'quick fox', vector: vec(1, 0) },
        { id: 'b', text: 'quick fox', vector: vec(0.99, 0.01) },
    ];
    const hits = hybridRetrieve({ text: 'quick fox', vector: vec(1, 0) }, candidates, {
        topK: 2,
        rerank: false,
    });
    expect(hits.length).toBe(2);
});

test('hybridRetrieve appends fused tail when MMR ranks only the vectored subset', () => {
    const candidates = [
        { id: 'withvec', text: 'quick fox', vector: vec(1, 0) },
        { id: 'novec', text: 'quick fox lexical' }, // matches BM25 but has no vector
    ];
    const hits = hybridRetrieve({ text: 'quick fox', vector: vec(1, 0) }, candidates, { topK: 5 });
    const ids = hits.map((h) => h.id);
    expect(ids).toContain('withvec');
    expect(ids).toContain('novec'); // appended after the MMR-ranked vectored subset
});
