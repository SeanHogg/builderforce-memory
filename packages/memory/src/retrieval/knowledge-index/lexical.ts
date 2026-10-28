/**
 * lexical.ts — the knowledge index's BM25 tier, as an incremental inverted index.
 *
 * `bm25Search` rebuilds its term table per call, which is right for a bounded
 * candidate set and wrong for 50k cards queried on every prompt. This keeps the
 * postings across upserts and scores with the SAME Okapi core (`bm25Idf`,
 * `bm25LengthNorm`, `bm25TermScore`) — the formula is not restated here.
 */

import { bm25Idf, bm25LengthNorm, bm25TermScore } from '../bm25.js';
import { compareHits } from '../order.js';
import { codeTokenize } from './tokenize.js';

export interface LexicalHit {
    id: string;
    score: number;
}

export class LexicalIndex {
    /** term → (doc id → term frequency). */
    private readonly _postings = new Map<string, Map<string, number>>();
    /** doc id → its distinct terms and token count. */
    private readonly _docs = new Map<string, { terms: string[]; len: number }>();
    private _totalLen = 0;

    get size(): number {
        return this._docs.size;
    }

    /** Indexes (or re-indexes) one document. */
    set(id: string, text: string): void {
        this.delete(id);
        const tokens = codeTokenize(text);
        const tf = new Map<string, number>();
        for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
        for (const [term, f] of tf) {
            let posting = this._postings.get(term);
            if (!posting) this._postings.set(term, (posting = new Map()));
            posting.set(id, f);
        }
        this._docs.set(id, { terms: [...tf.keys()], len: tokens.length });
        this._totalLen += tokens.length;
    }

    delete(id: string): void {
        const doc = this._docs.get(id);
        if (!doc) return;
        for (const term of doc.terms) {
            const posting = this._postings.get(term);
            posting?.delete(id);
            if (posting && posting.size === 0) this._postings.delete(term);
        }
        this._totalLen -= doc.len;
        this._docs.delete(id);
    }

    /** BM25 hits for `query`, best first ({@link compareHits}), at most `limit`. */
    search(query: string, limit: number): LexicalHit[] {
        const n = this._docs.size;
        if (n === 0 || limit <= 0) return [];
        const terms = [...new Set(codeTokenize(query))].sort();
        const avgdl = this._totalLen / n || 1;
        const scores = new Map<string, number>();
        // Terms in sorted order, so each doc's sum is accumulated in the same order everywhere.
        for (const term of terms) {
            const posting = this._postings.get(term);
            if (!posting) continue;
            const idf = bm25Idf(n, posting.size);
            for (const [id, f] of posting) {
                const norm = bm25LengthNorm(this._docs.get(id)!.len, avgdl);
                scores.set(id, (scores.get(id) ?? 0) + bm25TermScore(f, idf, norm));
            }
        }
        const hits: LexicalHit[] = [];
        for (const [id, score] of scores) if (score > 0) hits.push({ id, score });
        return hits.sort(compareHits).slice(0, limit);
    }
}
