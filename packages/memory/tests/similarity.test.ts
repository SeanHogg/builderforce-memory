/**
 * tests/similarity.test.ts
 * The shared similarity primitives used by MemoryStore + SemanticCache.
 */

import { cosineSimilarity, jaccardSimilarity, tokenize } from '../src/similarity/index.js';

test('tokenize lowercases and splits on punctuation/whitespace', () => {
    expect(tokenize('The Quick, brown-FOX!')).toEqual(['the', 'quick', 'brown', 'fox']);
    expect(tokenize('   ')).toEqual([]);
});

test('jaccardSimilarity over token sets', () => {
    expect(jaccardSimilarity(new Set(['a', 'b']), new Set(['a', 'b']))).toBe(1);
    expect(jaccardSimilarity(new Set(['a', 'b']), new Set(['b', 'c']))).toBeCloseTo(1 / 3);
    expect(jaccardSimilarity(new Set(['a']), new Set(['b']))).toBe(0);
    expect(jaccardSimilarity(new Set(), new Set())).toBe(1); // both empty → identical
});

test('cosineSimilarity is 1 for aligned, 0 for orthogonal, 0 for a zero vector', () => {
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([2, 0]))).toBeCloseTo(1);
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([0, 1]))).toBe(0);
    expect(cosineSimilarity(new Float32Array([0, 0]), new Float32Array([1, 1]))).toBe(0);
});

test('cosineSimilarity compares over the shorter length when lengths differ', () => {
    // Only the first component is compared → aligned.
    expect(cosineSimilarity(new Float32Array([1, 0, 0]), new Float32Array([1]))).toBeCloseTo(1);
});
