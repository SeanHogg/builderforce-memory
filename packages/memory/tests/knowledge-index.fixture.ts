/**
 * tests/knowledge-index.fixture.ts — synthetic cards, a deterministic bag-of-words
 * embedder and query set shared by the knowledge-index tests.
 */

import type { EmbedderPort, IndexDoc } from '../src/retrieval/knowledge-index/index.js';
import { codeTokenize } from '../src/retrieval/knowledge-index/tokenize.js';

/** Seeded PRNG in [0,1). */
export function rng(seed: number): () => number {
    let s = seed >>> 0 || 1;
    return () => ((s = (Math.imul(1103515245, s) + 12345) >>> 0) / 0x1_0000_0000);
}

export function shuffled<T>(items: readonly T[], seed: number): T[] {
    const r = rng(seed);
    const out = [...items];
    for (let i = out.length - 1; i > 0; i--) {
        const j = Math.floor(r() * (i + 1));
        [out[i], out[j]] = [out[j]!, out[i]!];
    }
    return out;
}

function fnv(text: string): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193) >>> 0;
    return h;
}

const VOCAB = Array.from({ length: 240 }, (_, i) => `term${i}`);

/** `n` cards: a path plus eight vocabulary words; keys are the path and basename. */
export function makeDocs(n: number, seed = 7): IndexDoc[] {
    const r = rng(seed);
    return Array.from({ length: n }, (_, i) => {
        const id = `doc${String(i).padStart(5, '0')}`;
        const path = `src/area${i % 20}/file${i}.ts`;
        const words = Array.from({ length: 8 }, () => VOCAB[Math.floor(r() * VOCAB.length)]!);
        return { id, text: `${path} ${words.join(' ')}`, keys: [path, `file${i}.ts`] };
    });
}

export function makeQueries(n: number, seed = 11): string[] {
    const r = rng(seed);
    return Array.from({ length: n }, () =>
        Array.from({ length: 3 }, () => VOCAB[Math.floor(r() * VOCAB.length)]!).join(' '));
}

/** Unit (float64) bag-of-words vector: each token adds a fixed pseudo-random direction. */
export function unitVector(text: string, dims: number): Float64Array {
    const v = new Float64Array(dims);
    for (const token of codeTokenize(text)) {
        const r = rng(fnv(token));
        for (let d = 0; d < dims; d++) v[d] += r() * 2 - 1;
    }
    let sum = 0;
    for (let d = 0; d < dims; d++) sum += v[d]! * v[d]!;
    const norm = Math.sqrt(sum) || 1;
    for (let d = 0; d < dims; d++) v[d] = v[d]! / norm;
    return v;
}

/** An embedder over {@link unitVector}; `transform` post-processes each unit vector. */
export function makeEmbedder(
    dims: number,
    fingerprint = `fixture-bow/1/${dims}`,
    transform: (unit: Float64Array) => ArrayLike<number> = (u) => u,
): EmbedderPort & { calls: number } {
    const port = {
        fingerprint,
        dims,
        calls: 0,
        async embed(texts: string[]): Promise<Float32Array[]> {
            port.calls += texts.length;
            return texts.map((t) => Float32Array.from(transform(unitVector(t, dims))));
        },
    };
    return port;
}
