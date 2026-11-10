/**
 * src/knowledge/transitions.test.ts
 * PRD 38 P7: station transition probabilities — row normalisation, dropped self-loops, the per-row choice of traces over
 * reference weights, intervals only on observed (trace) rows, and the greedy most-likely walk.
 */

import {
    mostLikelyWalk, stationTransitions, transitionOf, transitionsFromSequences, transitionsFromWeights,
} from './transitions.js';

const W = (from: string, to: string, weight: number) => ({ from, to, weight });
const same = (a: unknown, b: unknown) => expect(JSON.stringify(a)).toBe(JSON.stringify(b));
/** A trace history with `k` hops A → B and `n − k` hops A → C. */
const traces = (k: number, n: number) => [...Array.from({ length: k }, () => ['A', 'B']), ...Array.from({ length: n - k }, () => ['A', 'C'])];

describe('transitionsFromWeights', () => {
    const t = transitionsFromWeights([W('A', 'B', 2), W('A', 'C', 1), W('A', 'B', 1), W('A', 'A', 5), W('B', 'A', 2), W('C', 'B', 0)]);

    it('sums duplicate edges, drops self-loops and non-positive weights, and row-normalises', () => {
        expect([...t.keys()]).toEqual(['A', 'B']);
        expect(t.get('A')!.transitions.map((h) => [h.to, h.p, h.k, h.n])).toEqual([['B', 0.75, 3, 4], ['C', 0.25, 1, 4]]);
        expect(t.get('B')!.transitions.map((h) => [h.to, h.p])).toEqual([['A', 1]]);
        for (const row of t.values()) expect(row.transitions.reduce((s, h) => s + h.p, 0)).toBeCloseTo(1, 6);
    });

    it('reference rows are not observations: they say so and carry no interval', () => {
        for (const row of t.values()) {
            expect(row.source).toBe('references');
            for (const h of row.transitions) expect('low' in h || 'high' in h).toBe(false);
        }
    });

    it('rows sort p desc, then `to` asc, and shuffled input gives the same table', () => {
        const edges = [W('S', 'z', 1), W('S', 'b', 1), W('S', 'm', 2)];
        const a = transitionsFromWeights(edges);
        expect(a.get('S')!.transitions.map((h) => h.to)).toEqual(['m', 'b', 'z']);
        same([...transitionsFromWeights([...edges].reverse())], [...a]);
    });
});

describe('transitionsFromSequences', () => {
    it('counts first-order hops of the traces; a repeated station is not a hop', () => {
        const t = transitionsFromSequences([['A', 'B', 'C'], ['A', 'A', 'B'], ['A', 'C']]);
        expect(t.get('A')!.transitions.map((h) => [h.to, h.k, h.n, h.p])).toEqual([['B', 2, 3, 0.666667], ['C', 1, 3, 0.333333]]);
        expect(t.get('B')!.transitions.map((h) => [h.to, h.p])).toEqual([['C', 1]]);
        expect(t.get('A')!.source).toBe('traces');
        expect(t.has('C')).toBe(false);
    });

    it('a trace hop carries the 90% Jeffreys interval: pinned at 3 of 4', () => {
        const h = transitionOf(transitionsFromSequences(traces(3, 4)), 'A', 'B')!;
        expect([h.p, h.low, h.high]).toEqual([0.75, 0.349293, 0.953993]);
    });

    it('the interval holds p (upper end 1 when every hop goes one way) and narrows with n', () => {
        const all = transitionOf(transitionsFromSequences(traces(2, 2)), 'A', 'B')!;
        expect([all.p, all.high]).toEqual([1, 1]);
        expect(all.low).toBeLessThan(1);
        const width = (k: number, n: number) => {
            const h = transitionOf(transitionsFromSequences(traces(k, n)), 'A', 'B')!;
            expect(h.low!).toBeLessThanOrEqual(h.p);
            expect(h.high!).toBeGreaterThanOrEqual(h.p);
            return h.high! - h.low!;
        };
        expect(width(30, 40)).toBeLessThan(width(3, 4));
        expect(width(300, 400)).toBeLessThan(width(30, 40));
    });
});

describe('stationTransitions', () => {
    it('takes a station\'s trace row over its reference row', () => {
        const t = stationTransitions({ sequences: [['A', 'C']], weights: [W('A', 'B', 10)] });
        expect(t.get('A')!.source).toBe('traces');
        expect(transitionOf(t, 'A', 'C')!.p).toBe(1);
        expect(transitionOf(t, 'A', 'B')).toBeUndefined();
    });

    it('falls back per row: a station with no trace hop keeps its reference row', () => {
        const t = stationTransitions({ sequences: [['A', 'B']], weights: [W('A', 'C', 5), W('B', 'C', 1), W('B', 'A', 3)] });
        expect([...t].map(([from, row]) => [from, row.source])).toEqual([['A', 'traces'], ['B', 'references']]);
        expect(t.get('B')!.transitions.map((h) => [h.to, h.p])).toEqual([['A', 0.75], ['C', 0.25]]);
        expect(transitionOf(t, 'B', 'A')!.low).toBeUndefined();
        expect(transitionOf(t, 'A', 'B')!.low).toBeDefined();
    });

    it.each([
        ['no sequences', undefined], ['empty sequences', []], ['one-stop traces', [['A']]], ['traces that never move', [['A', 'A']]],
    ])('is all references with %s', (_name, sequences) => {
        const t = stationTransitions({ ...(sequences ? { sequences } : {}), weights: [W('A', 'B', 10)] });
        expect(t.get('A')!.source).toBe('references');
        expect(transitionOf(t, 'A', 'B')!.p).toBe(1);
    });
});

describe('mostLikelyWalk', () => {
    // A → B and A → C tie at 0.5 (B wins on `to`); B → A is likelier but visited, so B goes to D; D has no successor.
    const table = transitionsFromWeights([W('A', 'C', 1), W('A', 'B', 1), W('B', 'A', 9), W('B', 'D', 1), W('C', 'E', 1)]);

    it('follows the per-hop argmax, breaks ties by `to`, never revisits, and stops at a dead end', () => {
        const w = mostLikelyWalk(table, 'A', 10);
        expect(w.stops).toEqual(['A', 'B', 'D']);
        expect(w.hops.map((h) => [h.to, h.p])).toEqual([['B', 0.5], ['D', 0.1]]);
    });

    it('stops after maxHops', () => {
        expect(mostLikelyWalk(table, 'A', 1).stops).toEqual(['A', 'B']);
        expect(mostLikelyWalk(table, 'A', 0)).toEqual({ stops: ['A'], hops: [] });
    });

    it('a start with no row is a walk of one stop', () => {
        expect(mostLikelyWalk(table, 'Z', 5)).toEqual({ stops: ['Z'], hops: [] });
    });

    it('over a mixed table: trace hops carry their interval, reference hops do not', () => {
        // A and B have trace rows (A → B → C); C has none, so its reference row (C → D) continues the walk.
        const mixed = stationTransitions({ sequences: [['A', 'B', 'C']], weights: [W('A', 'Z', 9), W('C', 'D', 1), W('C', 'A', 0)] });
        const w = mostLikelyWalk(mixed, 'A', 10);
        expect(w.stops).toEqual(['A', 'B', 'C', 'D']);
        expect(w.hops.map((h) => [h.to, h.low !== undefined, h.high !== undefined])).toEqual([['B', true, true], ['C', true, true], ['D', false, false]]);
        expect(w.hops[0]).toMatchObject({ p: 1, high: 1 });
        expect('low' in w.hops[2]! || 'high' in w.hops[2]!).toBe(false);
    });

    it('a trace row that leads only to visited stations ends the walk; its reference row is not consulted', () => {
        // B's traces only go back to A (visited); B's reference row would offer C, but the trace row wins for B.
        const t = stationTransitions({ sequences: [['A', 'B', 'A']], weights: [W('B', 'C', 5)] });
        expect(t.get('B')!.source).toBe('traces');
        expect(mostLikelyWalk(t, 'A', 10).stops).toEqual(['A', 'B']);
    });

    it('is deterministic under shuffled input', () => {
        const shuffled = transitionsFromWeights([W('C', 'E', 1), W('B', 'D', 1), W('B', 'A', 9), W('A', 'B', 1), W('A', 'C', 1)]);
        same(mostLikelyWalk(shuffled, 'A', 10), mostLikelyWalk(table, 'A', 10));
    });
});
