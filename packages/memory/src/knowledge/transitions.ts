/**
 * knowledge/transitions.ts — station transition probabilities (PRD 38 P7: Simulate, "Run to end", path-strip hops).
 *
 * A first-order Markov chain over stations: P(j|i) = k / n, where k is the count (or weight) of i → j and n the total out
 * of i. A station's row comes from recorded run traces when the traces have a hop out of it, otherwise from the reference
 * weights between stations; each row says which (`source`). Self-loops are dropped (staying put is not a hop), as are
 * non-positive weights.
 *
 * A trace row is an observation, so every hop carries its credible interval (`rateInterval`, ../stats). A reference row is
 * not one, so its hops carry no interval (`low` / `high` are absent).
 *
 * Stations are a view concept: memory holds the formula, the view applies it to its station grouping.
 * Deterministic: rows in station order, each row p desc then `to` asc; weights summed in ascending order.
 */

import { rateInterval } from '../stats/index.js';
import { compareStr, round6 } from './refs.js';

export type TransitionSource = 'traces' | 'references';

export interface StationTransition {
    to: string;
    /** k / n, round6. */
    p: number;
    /** Count (traces) or weight (references) of this hop. */
    k: number;
    /** Total count or weight out of the row's station. */
    n: number;
    /** Credible interval of p (`rateInterval(k, n)`), round6. Only on a trace row. */
    low?: number;
    high?: number;
}

export interface TransitionRow {
    source: TransitionSource;
    transitions: readonly StationTransition[];
}

export type TransitionTable = ReadonlyMap<string, TransitionRow>;

type Weighted = ReadonlyArray<{ from: string; to: string; weight: number }>;

function tableOf(edges: Weighted, source: TransitionSource): Map<string, TransitionRow> {
    const rows = new Map<string, Map<string, number[]>>();
    for (const e of edges) {
        if (e.from === e.to || !(e.weight > 0) || !Number.isFinite(e.weight)) continue;
        const row = rows.get(e.from) ?? new Map<string, number[]>();
        const ws = row.get(e.to) ?? [];
        ws.push(e.weight);
        row.set(e.to, ws);
        rows.set(e.from, row);
    }
    const sum = (ws: number[]) => [...ws].sort((a, b) => a - b).reduce((s, w) => s + w, 0);
    const table = new Map<string, TransitionRow>();
    for (const from of [...rows.keys()].sort(compareStr)) {
        const hops = [...rows.get(from)!].map(([to, ws]) => ({ to, k: sum(ws) }));
        const n = sum(hops.map((h) => h.k));
        const transitions = hops.map(({ to, k }): StationTransition => {
            const hop = { to, p: round6(k / n), k: round6(k), n: round6(n) };
            if (source === 'references') return hop;
            const { low, high } = rateInterval(k, n);
            return { ...hop, low: round6(low), high: round6(high) };
        }).sort((a, b) => b.p - a.p || compareStr(a.to, b.to));
        table.set(from, { source, transitions });
    }
    return table;
}

/** Directed reference weights, row-normalised; rows are `references` and carry no interval. */
export function transitionsFromWeights(edges: Weighted): TransitionTable {
    return tableOf(edges, 'references');
}

/** First-order Markov counts of observed station sequences (run traces); a repeated station is not a hop. */
export function transitionsFromSequences(sequences: ReadonlyArray<readonly string[]>): TransitionTable {
    const hops: Array<{ from: string; to: string; weight: number }> = [];
    for (const s of sequences) for (let i = 1; i < s.length; i++) hops.push({ from: s[i - 1]!, to: s[i]!, weight: 1 });
    return tableOf(hops, 'traces');
}

/** Per station: its trace row when the traces have a hop out of it, else its reference row. */
export function stationTransitions(input: { sequences?: ReadonlyArray<readonly string[]>; weights: Weighted }): TransitionTable {
    const traced = transitionsFromSequences(input.sequences ?? []);
    const referenced = transitionsFromWeights(input.weights);
    const table = new Map<string, TransitionRow>();
    for (const from of [...new Set([...traced.keys(), ...referenced.keys()])].sort(compareStr)) {
        table.set(from, traced.get(from) ?? referenced.get(from)!);
    }
    return table;
}

/** The hop from → to, if the table has it. */
export function transitionOf(table: TransitionTable, from: string, to: string): StationTransition | undefined {
    return table.get(from)?.transitions.find((t) => t.to === to);
}

/**
 * A greedy walk from `start`: at each hop the likeliest unvisited successor (ties by `to` asc), at most `maxHops`, stopping
 * when no unvisited successor is left. Per-hop argmax, NOT the most likely whole path (no Viterbi search).
 * A station whose trace row leads only to stations already visited ends the walk; it does not fall back to its reference row.
 */
export function mostLikelyWalk(table: TransitionTable, start: string, maxHops: number): { stops: string[]; hops: StationTransition[] } {
    const stops = [start];
    const hops: StationTransition[] = [];
    const seen = new Set(stops);
    for (let cur = start; hops.length < maxHops;) {
        const next = table.get(cur)?.transitions.find((t) => !seen.has(t.to));
        if (!next) break;
        hops.push(next);
        stops.push(next.to);
        seen.add(next.to);
        cur = next.to;
    }
    return { stops, hops };
}
