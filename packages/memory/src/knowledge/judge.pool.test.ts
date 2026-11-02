/**
 * src/knowledge/judge.pool.test.ts
 * Bounded-parallelism contract for the judgement pass (PRD 38 §4.3): model calls are the slow
 * phase of a review, so `judgeSections` and `compareClaims` fan out over a bounded worker pool
 * (`JUDGE_CONCURRENCY`) instead of one call at a time. These tests pin the bound with a port that
 * records its max observed in-flight count, and pin the budget cap (a synchronous `used`
 * increment plus the `budgetLeft` check cannot overshoot `modelCalls` under concurrency).
 */

import { compareClaims, JUDGE_CONCURRENCY, judgeSections, JudgedBatch } from './judge.js';
import type { JudgementPort, JudgementRequest, JudgementResult, NodeRef } from './model.js';
import type { SimilarPair } from '../retrieval/knowledge-index/types.js';

/** A port that tracks concurrent in-flight calls: increment on entry, decrement on return, record max. */
function trackingPort(opts?: { delayMs?: number; relation?: 'contradicts' | 'agrees' | 'unrelated' }) {
    let inFlight = 0;
    let max = 0;
    let calls = 0;
    const port: JudgementPort = {
        async judge(req: JudgementRequest): Promise<JudgementResult> {
            inFlight += 1;
            max = Math.max(max, inFlight);
            calls += 1;
            try {
                if (opts?.delayMs) await new Promise((r) => setTimeout(r, opts.delayMs));
                if (req.kind === 'extract') {
                    return {
                        invalid: 0,
                        extract: req.items.map((it) => ({
                            ref: it.ref,
                            claims: [{
                                subject: 's', predicate: 'p', object: 'o', quote: 'q',
                            }],
                            entities: [],
                        })),
                    };
                }
                const [a, b] = req.items;
                return {
                    invalid: 0,
                    compare: [{ a: a.ref, b: b.ref, relation: opts?.relation ?? 'agrees', confidence: 0.9 }],
                };
            } finally {
                inFlight -= 1;
            }
        },
    };
    return {
        port,
        stats: {
            get max() { return max; },
            get calls() { return calls; },
        },
    };
}

const budget = (modelCalls: number) => ({
    modelCalls,
    deadline: Date.now() + 60_000,
    used: 0,
    now: () => Date.now(),
});

const extractArgs = (over: Partial<Parameters<typeof judgeSections>[0]>) => ({
    pack: 'docs',
    path: 'a.md',
    fileHash: 'hash-a',
    sections: [] as ReadonlyArray<{ ref: NodeRef; text: string }>,
    budget: budget(200),
    batch: new JudgedBatch(),
    entities: new Map(),
    onJudged: () => {},
    ...over,
});

describe('judgeSections bounded pool', () => {
    it('judges all sections with max in-flight bounded by JUDGE_CONCURRENCY', async () => {
        const { port, stats } = trackingPort({ delayMs: 5 });
        const sections = Array.from({ length: 12 }, (_, i) => ({
            ref: `src:root::a.md#sec-${i}` as NodeRef,
            text: `section ${i}`,
        }));
        const batch = new JudgedBatch();
        const judged: NodeRef[] = [];
        const claims = await judgeSections(extractArgs({
            port,
            sections,
            batch,
            onJudged: (ref) => judged.push(ref),
        }));

        expect(stats.max).toBeLessThanOrEqual(JUDGE_CONCURRENCY);
        expect(stats.max).toBeGreaterThan(1); // actually parallel, not serial
        expect(batch.judged).toBe(12);
        expect(batch.unjudged).toBe(0);
        expect(judged).toHaveLength(12);
        expect(claims).toHaveLength(12);
        expect(stats.calls).toBe(12);
    });

    it('respects the modelCalls budget exactly under concurrency', async () => {
        const { port, stats } = trackingPort({ delayMs: 5 });
        const sections = Array.from({ length: 12 }, (_, i) => ({
            ref: `src:root::a.md#sec-${i}` as NodeRef,
            text: `section ${i}`,
        }));
        const batch = new JudgedBatch();
        await judgeSections(extractArgs({ port, sections, batch, budget: budget(3) }));

        expect(stats.calls).toBe(3);
        expect(batch.judged).toBe(3);
        expect(batch.unjudged).toBe(9);
    });

    it('produces identical claims content regardless of completion order', async () => {
        const { port } = trackingPort({ delayMs: 1 });
        const sections = Array.from({ length: 10 }, (_, i) => ({
            ref: `src:root::a.md#sec-${i}` as NodeRef,
            text: `section ${i}`,
        }));
        const claims = await judgeSections(extractArgs({ port, sections }));
        expect(claims).toHaveLength(10);
        for (const c of claims) {
            expect(c.text).toBe('s p o');
            expect(c.ref.startsWith('fact:claim:')).toBe(true);
        }
    });
});

describe('compareClaims bounded pool', () => {
    const claimRef = (i: number) => `fact:claim:${String(i).padStart(3, '0')}` as NodeRef;

    const compareArgs = (over: Partial<Parameters<typeof compareClaims>[0]>) => {
        const claims = new Map<string, { text: string; hash: string }>();
        for (let i = 0; i < 8; i++) claims.set(claimRef(i), { text: `claim ${i}`, hash: 'h' });
        const pairs: SimilarPair[] = [];
        for (let i = 0; i < 8; i++) {
            const j = (i + 1) % 8;
            pairs.push({ a: claimRef(i), b: claimRef(j), similarity: 0.9 - i * 0.01 });
        }
        return {
            port: trackingPort().port,
            pairs,
            claims,
            budget: budget(200),
            batch: new JudgedBatch(),
            ...over,
        };
    };

    it('compares all pairs with max in-flight bounded by JUDGE_CONCURRENCY', async () => {
        const { port, stats } = trackingPort({ delayMs: 5 });
        const batch = new JudgedBatch();
        const out = await compareClaims(compareArgs({ port, batch }));

        expect(stats.max).toBeLessThanOrEqual(JUDGE_CONCURRENCY);
        expect(stats.max).toBeGreaterThan(1);
        expect(batch.judged).toBe(8);
        expect(out.aborted).toBe(false);
        expect(out.skipped.size).toBe(0);
        expect(stats.calls).toBe(8);
    });

    it('respects the modelCalls budget exactly under concurrency', async () => {
        const { port, stats } = trackingPort({ delayMs: 5 });
        const batch = new JudgedBatch();
        const out = await compareClaims(compareArgs({ port, batch, budget: budget(3) }));

        expect(stats.calls).toBe(3);
        expect(batch.judged).toBe(3);
        expect(batch.unjudged).toBe(5);
        expect(out.skipped.size).toBeGreaterThan(0);
    });

    it('dedupes identical pairs synchronously so no pair is scheduled twice', async () => {
        const { port, stats } = trackingPort({ delayMs: 1 });
        const batch = new JudgedBatch();
        const claims = new Map<string, { text: string; hash: string }>();
        for (const i of [0, 1]) claims.set(claimRef(i), { text: `claim ${i}`, hash: 'h' });
        // The same unordered pair twice (b/a swapped) plus an exact duplicate.
        const pairs: SimilarPair[] = [
            { a: claimRef(0), b: claimRef(1), similarity: 0.9 },
            { a: claimRef(1), b: claimRef(0), similarity: 0.9 },
            { a: claimRef(0), b: claimRef(1), similarity: 0.8 },
        ];
        const out = await compareClaims(compareArgs({ port, batch, pairs, claims }));

        expect(stats.calls).toBe(1);
        expect(batch.judged).toBe(1);
        expect(out.aborted).toBe(false);
    });
});
