/**
 * knowledge/trace.ts — "how does this reach that" (PRD 38 §3.4).
 *
 * `traceGraph` walks the chosen edge kinds depth-first from one node: heaviest edge first,
 * then by target ref; each node is entered once; reaching a visited node is a `back` step
 * (also `cycle` when the edge carries the cycle flag). Lanes are the partitions the trace
 * touches — code layers or doc folders — and `toMermaid` draws them as a sequence diagram.
 */

import type { KEdge, KnowledgeGraph, NodeRef, PartitionRegistry, Trace, TraceStep } from './model.js';
import { partitionOrder } from './partitions.js';
import { compareStr } from './refs.js';

export const DEFAULT_TRACE_EDGE_KINDS: readonly string[] = ['references', 'links', 'mentions'];
export const DEFAULT_TRACE_DEPTH = 4;
export const DEFAULT_TRACE_STEPS = 60;
/** The lane of a node with no partition. */
export const UNPARTITIONED_LANE = '(none)';

/** A trace that also knows each ref's lane (what {@link toMermaid} draws participants from). */
export interface LanedTrace extends Trace {
    laneOf: Record<string, string>;
}

export interface TraceOptions {
    edgeKinds?: string[];
    maxDepth?: number;
    maxSteps?: number;
    /** Orders lanes by partition order; without it lanes keep first-visit order. */
    registry?: PartitionRegistry;
}

export function traceGraph(g: KnowledgeGraph, from: NodeRef, opts: TraceOptions = {}): LanedTrace {
    const kinds = new Set(opts.edgeKinds ?? DEFAULT_TRACE_EDGE_KINDS);
    const maxDepth = opts.maxDepth ?? DEFAULT_TRACE_DEPTH;
    const maxSteps = opts.maxSteps ?? DEFAULT_TRACE_STEPS;
    const partition = new Map(g.nodes.map((n) => [n.ref, n.partition] as const));
    const out = new Map<NodeRef, KEdge[]>();
    for (const e of g.edges) {
        if (!kinds.has(e.kind)) continue;
        const list = out.get(e.from) ?? [];
        list.push(e);
        out.set(e.from, list);
    }
    for (const list of out.values()) {
        list.sort((a, b) => b.weight - a.weight || compareStr(a.to, b.to) || compareStr(a.kind, b.kind));
    }

    const steps: TraceStep[] = [];
    const visited = new Set<NodeRef>([from]);
    let truncated = false;
    const walk = (node: NodeRef, depth: number): void => {
        for (const e of out.get(node) ?? []) {
            if (steps.length >= maxSteps) { truncated = true; return; }
            const back = visited.has(e.to);
            steps.push({
                from: e.from, to: e.to, depth: depth + 1, kind: e.kind,
                ...(e.verdict ? { verdict: e.verdict } : {}),
                ...(back && e.cycle ? { cycle: true } : {}),
                ...(back ? { back: true } : {}),
            });
            if (back) continue;
            visited.add(e.to);
            if (depth + 1 < maxDepth) walk(e.to, depth + 1);
            else if ((out.get(e.to) ?? []).length > 0) truncated = true;
        }
    };
    if (partition.has(from)) walk(from, 0);

    const laneOf: Record<string, string> = {};
    const lanes: string[] = [];
    for (const ref of [from, ...steps.flatMap((s) => [s.from, s.to])]) {
        if (ref in laneOf) continue;
        const lane = partition.get(ref) ?? UNPARTITIONED_LANE;
        laneOf[ref] = lane;
        if (!lanes.includes(lane)) lanes.push(lane);
    }
    if (opts.registry) {
        const reg = opts.registry;
        const order = (l: string) => partitionOrder(l, reg) ?? Number.MAX_SAFE_INTEGER;
        lanes.sort((a, b) => order(a) - order(b) || compareStr(a, b));
    }
    return { from, lanes, steps, truncated, laneOf };
}

/** Mermaid text is line- and statement-delimited; keep labels on one safe line. */
function safe(text: string): string {
    return text.replace(/[\r\n]+/g, ' ').replace(/[;#:]/g, ' ').replace(/\s+/g, ' ').trim();
}

/**
 * A `sequenceDiagram`: lanes are participants, steps are messages. A back step is dashed;
 * an upward/skipping step and a cycle are named in the message.
 */
export function toMermaid(t: Trace, label: (r: NodeRef) => string): string {
    const laned = (t as Partial<LanedTrace>).laneOf;
    // Without lanes every node is its own participant.
    const laneOf = (ref: NodeRef): string => laned?.[ref] ?? ref;
    const lanes = laned ? t.lanes : [...new Set([t.from, ...t.steps.flatMap((s) => [s.from, s.to])])];
    const id = new Map(lanes.map((l, i) => [l, `P${i}`] as const));
    const lines = ['sequenceDiagram'];
    for (const lane of lanes) lines.push(`    participant ${id.get(lane)} as ${safe(laned ? lane : label(lane as NodeRef))}`);
    for (const s of t.steps) {
        const arrow = s.back ? '-->>' : '->>';
        const notes = [s.verdict === 'upward' || s.verdict === 'skips' ? s.verdict : '', s.cycle ? 'cycle' : '']
            .filter(Boolean).map((n) => ` [${n}]`).join('');
        const text = safe(`${label(s.from)} ${s.kind} ${label(s.to)}`) + notes;
        lines.push(`    ${id.get(laneOf(s.from))}${arrow}${id.get(laneOf(s.to))}: ${text}`);
    }
    return `${lines.join('\n')}\n`;
}
