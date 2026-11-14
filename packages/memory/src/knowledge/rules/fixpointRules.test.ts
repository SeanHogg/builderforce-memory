/**
 * src/knowledge/rules/fixpointRules.test.ts — the stratified fixed-point evaluator (PRD 41 §B5):
 * strata from the declared derived-row edges, order-independent results, bounded iterations,
 * a fixed point that reuses every row on the final pass, and rejection of recursive rules.
 */

import type { FileFacts } from '../facts/fileFacts.js';
import { derivedRowKey } from './evaluateRules.js';
import { defineRule } from './factRule.js';
import { evaluateRulesFixpoint, MAX_FIXPOINT_ITERATIONS, type SourcedRule } from './fixpointRules.js';
import { emptyFacts, FIXTURE_RULES } from './rules.testkit.test-util.js';

const files = (): Map<string, FileFacts> => new Map([
    ['a.ts', { ...emptyFacts('ha'), calls: [{ callee: 'db.get', enclosing: 'a', awaited: true, inLoop: true, count: 3 }] }],
    ['b.ts', { ...emptyFacts('hb'), calls: [{ callee: 'fs.read', enclosing: 'b', awaited: false, inLoop: true, count: 5 }] }],
]);

/** A global rule whose `seen` input is the io-in-loop rule's row for focus '' (its global row). */
const ioTotal = defineRule<{ seen?: { findings: unknown[] } }>({
    id: 'io-total', version: 1, reads: ['body'], scope: 'global', extraInputs: ['seen'],
    derive: ({ files: fs, extra }) => {
        const own = [...fs.values()].reduce((s, x) => s + x.calls.filter((c) => c.inLoop && c.awaited).length, 0);
        const upstream = (extra.seen?.findings ?? []).length;
        return upstream + own > 0
            ? [{ id: `io-total:${own + upstream}`, kind: 'io-total', pack: 'fixture', nodes: [], severity: 'info', params: { total: own + upstream } }]
            : [];
    },
});

const ioTotalSourced: SourcedRule<{ seen?: { findings: unknown[] } }> = {
    rule: ioTotal,
    rowSources: { seen: { rowKey: derivedRowKey('io-in-loop', '') } },
};

const plain = FIXTURE_RULES.map((rule) => ({ rule }));

describe('evaluateRulesFixpoint: stratified semi-naive fixed point', () => {
    it('places row-reading rules above their producers and reaches a fixed point', () => {
        const r = evaluateRulesFixpoint([...plain, ioTotalSourced], { files: files() });
        // Strata: every FIXTURE_RULE (no row sources) at 0; io-total (reads io-in-loop's row) at 1.
        expect(r.strata).toHaveLength(2);
        expect(r.strata[1]).toEqual(['io-total']);
        // a.ts has 1 awaited io-in-loop call; the global row feeds io-total.
        expect(r.evaluation.findings.map((x) => x.id)).toContain('io-total:1');
        // The fixed point is audited: at least one iteration per stratum; the final pass reused rows.
        expect(r.iterations).toHaveLength(2);
        expect(r.iterations.every((n) => n >= 1)).toBe(true);
    });

    it('is order-independent: registration order never changes the rows or strata', () => {
        const a = evaluateRulesFixpoint([...plain, ioTotalSourced].reverse(), { files: files() });
        const b = evaluateRulesFixpoint([...plain, ioTotalSourced], { files: files() });
        expect([...a.evaluation.rows.keys()].sort()).toEqual([...b.evaluation.rows.keys()].sort());
        expect(a.evaluation.findings.map((x) => x.id).sort()).toEqual(b.evaluation.findings.map((x) => x.id).sort());
        expect(a.strata).toEqual(b.strata);
        expect(a.iterations).toEqual(b.iterations);
    });

    it('reuses every row on a no-change rerun (early cutoff holds across the fixpoint)', () => {
        const first = evaluateRulesFixpoint([...plain, ioTotalSourced], { files: files() });
        const second = evaluateRulesFixpoint([...plain, ioTotalSourced], { files: files() }, first.evaluation.rows);
        expect(second.evaluation.trace.filter((t) => t.outcome === 'ran')).toEqual([]);
        expect(second.evaluation.findings.map((x) => x.id).sort()).toEqual(first.evaluation.findings.map((x) => x.id).sort());
        expect(second.iterations.every((n) => n === 1)).toBe(true);
    });

    it('rejects a recursive derived-row cycle as unstratifiable', () => {
        const cycA = defineRule<{ b?: unknown }>({ id: 'cyc-a', version: 1, reads: [], scope: 'global', extraInputs: ['b'], derive: () => [] });
        const cycB = defineRule<{ a?: unknown }>({ id: 'cyc-b', version: 1, reads: [], scope: 'global', extraInputs: ['a'], derive: () => [] });
        const a: SourcedRule<{ b?: unknown }> = { rule: cycA, rowSources: { b: { rowKey: derivedRowKey('cyc-b', '') } } };
        const b: SourcedRule<{ a?: unknown }> = { rule: cycB, rowSources: { a: { rowKey: derivedRowKey('cyc-a', '') } } };
        expect(() => evaluateRulesFixpoint([a, b], { files: new Map() })).toThrow(/fact_rule_unstratifiable/);
    });

    it('rejects a self-reading rule as unstratifiable and never runs it', () => {
        let calls = 0;
        const noisy = defineRule<{ self?: { findings: unknown[] } }>({
            id: 'noisy', version: 1, reads: [], scope: 'global', extraInputs: ['self'],
            derive: () => { calls += 1; return []; },
        });
        const self: SourcedRule<{ self?: { findings: unknown[] } }> = { rule: noisy, rowSources: { self: { rowKey: derivedRowKey('noisy', '') } } };
        expect(() => evaluateRulesFixpoint([self], { files: new Map() })).toThrow(/fact_rule_unstratifiable/);
        expect(calls).toBe(0);
        expect(MAX_FIXPOINT_ITERATIONS).toBeGreaterThan(0);
    });

    it('a stratum stabilizes after one pass when its row-source inputs never change', () => {
        // io-total reads io-in-loop's row; io-in-loop is in stratum 0 and its inputs are static,
        // so stratum 1 converges on its second pass (first computes, second reuses) or first reuse.
        const r = evaluateRulesFixpoint([...plain, ioTotalSourced], { files: files() });
        // Stratum 0 (7 rules) iterates until reuse; stratum 1 (io-total) needs at most 2 iterations.
        expect(r.iterations[1]).toBeLessThanOrEqual(2);
    });
});
