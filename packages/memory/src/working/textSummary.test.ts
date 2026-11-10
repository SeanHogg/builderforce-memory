/**
 * src/working/textSummary.test.ts — chunkText / chunkedTextSummary, the text-level summarizer brain-embedded's on-device
 * `chunkedSummary` binds to. The cases are that module's own (its reference tests), run against this port with its constants
 * (an 80% chunk share, at most 8 leaf chunks), plus the surrogate-pair rule and the B7 detector export.
 */

import { jest } from '@jest/globals';
import { chunkedTextSummary, chunkText, isRepeatedNarration, wordContained, wordOverlap, type ChunkedTextSummaryOptions } from './index.js';

const MAX_CHUNKS = 8;
/** 1 char = 1 token, a 100-token window: chunks must stay under 80. */
const base = { limit: 100, measure: async (t: string) => t.length, maxChunks: MAX_CHUNKS, budgetShare: 0.8 };
const paragraphs = (n: number, size = 20) => Array.from({ length: n }, (_, i) => `${i}`.padEnd(size, 'x')).join('\n\n');
const summarize = (o: Partial<ChunkedTextSummaryOptions> & Pick<ChunkedTextSummaryOptions, 'text' | 'summarizeChunk'>) => chunkedTextSummary({ ...base, ...o });

describe('chunkedTextSummary', () => {
    it('summarizes an in-window text in one call', async () => {
        const summarizeChunk = jest.fn(async (_c: string) => 'S');
        expect(await summarize({ text: 'short', summarizeChunk })).toBe('S');
        expect(summarizeChunk).toHaveBeenCalledTimes(1);
    });

    it('splits a 5x-window text on paragraph boundaries, then merges', async () => {
        const seen: string[] = [];
        const text = paragraphs(20, 22);
        expect(await summarize({ text, summarizeChunk: async (c) => { seen.push(c); return 'sum'; } })).toBe('sum');
        const leaves = seen.filter((c) => /^\d/.test(c));
        expect(leaves.length).toBeGreaterThan(1);
        for (const chunk of seen) expect(chunk.length).toBeLessThanOrEqual(80);
        expect(leaves.every((c) => c.split('\n\n').every((p) => p.trim().length === 22 || p.trim() === ''))).toBe(true);
    });

    it('recurses while the merged summaries are still over budget', async () => {
        const out = await summarize({ text: paragraphs(20, 22), summarizeChunk: async (c) => c.slice(0, Math.ceil(c.length / 2)) });
        expect(out).not.toBeNull();
        expect(out!.length).toBeLessThanOrEqual(80);
    });

    it('returns null past maxChunks leaf chunks without summarizing any', async () => {
        const summarizeChunk = jest.fn(async (_c: string) => 'S');
        expect(await summarize({ text: paragraphs(MAX_CHUNKS * 4 + 4, 40), summarizeChunk })).toBeNull();
        expect(summarizeChunk).not.toHaveBeenCalled();
    });

    it('is null when a chunk is declined', async () => {
        expect(await summarize({ text: paragraphs(6, 40), summarizeChunk: async () => null })).toBeNull();
    });

    it('is null when a merge does not shrink the text', async () => {
        expect(await summarize({ text: paragraphs(6, 40), summarizeChunk: async (c) => c })).toBeNull();
    });

    it('is null past maxMergeDepth', async () => {
        expect(await summarize({ text: paragraphs(20, 22), maxMergeDepth: 0, summarizeChunk: async () => 'S' })).toBeNull();
    });

    it('terminates on a paragraph longer than the window followed by a short one, preserving content', async () => {
        const seen: string[] = [];
        const text = `${'a'.repeat(300)}\n\nshort`;
        expect(await summarize({ text, summarizeChunk: async (c) => { seen.push(c); return 'S'; } })).toBe('S');
        const leaves = seen.slice(0, -1);
        expect(leaves.length).toBeGreaterThan(1);
        for (const c of leaves) expect(c.length).toBeLessThanOrEqual(80);
        expect(leaves.join('')).toBe(text);
    });

    it('reserves output tokens out of the window', async () => {
        const seen: string[] = [];
        await summarize({ text: paragraphs(10, 22), reserveTokens: 50, summarizeChunk: async (c) => { seen.push(c); return 's'; } });
        for (const c of seen.filter((c) => /^\d/.test(c))) expect(c.length).toBeLessThanOrEqual(40);
    });

    it('re-measures a chunk and splits again when the estimate under-cut', async () => {
        const seen: string[] = [];
        const measure = async (t: string) => (t.includes('9') ? t.length * 2 : t.length);
        await summarize({ text: paragraphs(12, 22), measure, summarizeChunk: async (c) => { seen.push(c); return 's'; } });
        for (const c of seen.filter((c) => /^\d/.test(c))) expect(await measure(c)).toBeLessThanOrEqual(80);
    });

    it('throws only AbortError', async () => {
        const controller = new AbortController();
        controller.abort();
        await expect(summarize({ text: 'x', summarizeChunk: async () => 'S', signal: controller.signal })).rejects.toMatchObject({ name: 'AbortError' });
    });
});

describe('chunkText', () => {
    it('cuts on paragraphs, then lines, then hard; joined, the chunks are the text', () => {
        const text = `${'p'.repeat(5)}\n\n${'l'.repeat(4)}\n${'m'.repeat(4)}\n\n${'h'.repeat(12)}`;
        const chunks = chunkText(text, 8);
        for (const c of chunks) expect(c.length).toBeLessThanOrEqual(8);
        expect(chunks.join('')).toBe(text);
    });

    it('regression: a hard cut never splits a surrogate pair', () => {
        const text = '😀'.repeat(9); // 18 code units
        for (const max of [1, 2, 3, 5, 7]) {
            const chunks = chunkText(text, max);
            expect(chunks.join('')).toBe(text);
            for (const c of chunks) expect(/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/.test(c)).toBe(false);
        }
    });
});

describe('B7 narration detector exports', () => {
    it('regression: wordOverlap / wordContained / isRepeatedNarration are the detector foldRepeatedNarration uses', () => {
        expect(wordOverlap('read the file', 'read the file')).toBe(1);
        expect(wordOverlap('', 'x')).toBe(0);
        expect(wordContained('read the file again', 'read the file')).toBe(true);
        expect(isRepeatedNarration('Now I will read the file.', 'Now I will read the file again.')).toBe(true);
        expect(isRepeatedNarration('Now edit the parser.', 'Read the docs first.')).toBe(false);
    });
});
