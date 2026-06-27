/**
 * BM25 (Okapi) lexical ranking.
 *
 * The keyword half of hybrid retrieval. Dense vector search matches meaning but
 * misses exact tokens (identifiers, error codes, rare names); BM25 catches those.
 * Fusing the two (see {@link ./fusion}) is what lifts the memory layer from
 * "cosine only" to a hybrid retriever on par with Weaviate-style search.
 *
 * Pure and zero-dependency — reuses the shared `tokenize` from ../similarity.
 */

import { tokenize } from '../similarity/index.js';

export interface Bm25Options {
    /** Term-frequency saturation. Higher = TF matters more. Default 1.5. */
    k1?: number;
    /** Length normalisation, 0..1. Higher = penalise long docs more. Default 0.75. */
    b?: number;
}

export interface Bm25Doc {
    id: string;
    text: string;
}

export interface Bm25Hit {
    id: string;
    score: number;
}

/**
 * Scores every document against `query` with Okapi BM25, returning hits sorted by
 * descending score (documents with no query-term overlap score 0 and are dropped).
 * Builds the index inline — for a recall over a bounded candidate set (the memory
 * store / a vector pre-filter) this is O(N·terms) and needs no persistence.
 */
export function bm25Search(query: string, docs: Bm25Doc[], opts: Bm25Options = {}): Bm25Hit[] {
    const k1 = opts.k1 ?? 1.5;
    const b = opts.b ?? 0.75;
    const N = docs.length;
    if (N === 0) return [];

    const queryTerms = new Set(tokenize(query));
    if (queryTerms.size === 0) return [];

    // Per-doc term frequencies + document lengths.
    const docTerms: { id: string; tf: Map<string, number>; len: number }[] = [];
    const df = new Map<string, number>();
    let totalLen = 0;

    for (const doc of docs) {
        const tokens = tokenize(doc.text);
        const tf = new Map<string, number>();
        for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
        for (const t of tf.keys()) if (queryTerms.has(t)) df.set(t, (df.get(t) ?? 0) + 1);
        docTerms.push({ id: doc.id, tf, len: tokens.length });
        totalLen += tokens.length;
    }
    const avgdl = totalLen / N || 1;

    // idf with the +1 smoothing variant (always non-negative).
    const idf = new Map<string, number>();
    for (const term of queryTerms) {
        const n = df.get(term) ?? 0;
        idf.set(term, Math.log(1 + (N - n + 0.5) / (n + 0.5)));
    }

    const hits: Bm25Hit[] = [];
    for (const d of docTerms) {
        let score = 0;
        for (const term of queryTerms) {
            const f = d.tf.get(term);
            if (!f) continue;
            const numer = f * (k1 + 1);
            const denom = f + k1 * (1 - b + b * (d.len / avgdl));
            score += idf.get(term)! * (numer / denom);
        }
        if (score > 0) hits.push({ id: d.id, score });
    }

    hits.sort((a, b) => b.score - a.score);
    return hits;
}
