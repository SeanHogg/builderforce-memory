/**
 * src/knowledge/rules/evaluateRules.test.ts — the trace tests (PRD 41 §8 Part B):
 *   - a body-only edit reruns `body`-reading rules (io-in-loop) and skips `structure`-only rules;
 *   - global rules are skipped when the fact kinds they read did not change anywhere;
 * plus neighbourhood hops, guard chains, extra inputs and rule validation.
 */

import type { FileFacts } from '../facts/fileFacts.js';
import { defineRule } from './factRule.js';
import { evaluateRules, type RuleEvaluation } from './evaluateRules.js';
import { callTotal, contentTag, emptyFacts, exportCount, FIXTURE_RULES, ioInLoop, mutualImport, routeGuard, unresolvedImport } from './rules.testkit.test-util.js';

const base = (): Map<string, FileFacts> => new Map([
    ['a.ts', { ...emptyFacts('ha'), imports: [{ specifier: './b', resolved: 'b.ts', names: [] }], exports: [{ name: 'a', kind: 'function', signatureHash: 's' }],
        calls: [{ callee: 'db.get', enclosing: 'a', awaited: true, inLoop: true, count: 1 }] }],
    ['b.ts', { ...emptyFacts('hb'), imports: [{ specifier: './c', resolved: 'c.ts', names: [] }] }],
    ['c.ts', emptyFacts('hc')],
    ['d.ts', emptyFacts('hd')],
]);

const ran = (e: RuleEvaluation) => e.trace.filter((t) => t.outcome === 'ran').map((t) => `${t.rule}@${t.focus}`).sort();
const outcome = (e: RuleEvaluation, rule: string, focus = '') => e.trace.find((t) => t.rule === rule && t.focus === focus)?.outcome;

describe('evaluateRules: verifying traces and early cutoff', () => {
    it('a first evaluation runs every row; an identical one reuses every row', () => {
        const files = base();
        const first = evaluateRules(FIXTURE_RULES, { files });
        expect(first.trace.every((t) => t.outcome === 'ran')).toBe(true);
        const second = evaluateRules(FIXTURE_RULES, { files: base() }, first.rows);
        expect(ran(second)).toEqual([]);
        expect(second.findings).toEqual(first.findings);
    });

    it('a body-only edit reruns body-reading rules (io-in-loop) and skips structure-only rules', () => {
        const first = evaluateRules(FIXTURE_RULES, { files: base() });
        const files = base();
        files.set('a.ts', { ...files.get('a.ts')!, calls: [] });
        const second = evaluateRules(FIXTURE_RULES, { files }, first.rows);
        expect(ran(second)).toEqual(['call-total@', 'io-in-loop@a.ts']);
        expect(outcome(second, 'export-count', 'a.ts')).toBe('reused');
        expect(outcome(second, 'unresolved-import', 'a.ts')).toBe('reused');
        expect(outcome(second, 'mutual-import')).toBe('reused');
        expect(second.findings.some((x) => x.kind === 'io-in-loop')).toBe(false);
    });

    it('global rules are skipped when their read kinds did not change anywhere (a comment edit)', () => {
        const first = evaluateRules(FIXTURE_RULES, { files: base() });
        const files = base();
        files.set('c.ts', { ...files.get('c.ts')!, contentHash: 'hc2' });
        const second = evaluateRules(FIXTURE_RULES, { files }, first.rows);
        expect(ran(second)).toEqual(['content-tag@c.ts']);
        expect(outcome(second, 'mutual-import')).toBe('reused');
        expect(outcome(second, 'call-total')).toBe('reused');
    });

    it('a structure edit reruns the global structure rule and the neighbourhood of the edited file only', () => {
        const first = evaluateRules(FIXTURE_RULES, { files: base() });
        const files = base();
        files.set('c.ts', { ...files.get('c.ts')!, imports: [{ specifier: './b', resolved: 'b.ts', names: [] }] });
        const second = evaluateRules(FIXTURE_RULES, { files }, first.rows);
        // Graph a–b–c, d alone. unresolved-import (1 hop): b {a,b,c} and c {b,c} see c; a {a,b} and d do not.
        // route-guard (2 hops): a, b and c all reach c; d does not.
        expect(ran(second)).toEqual([
            'export-count@c.ts', 'mutual-import@', 'route-guard@a.ts', 'route-guard@b.ts', 'route-guard@c.ts',
            'unresolved-import@b.ts', 'unresolved-import@c.ts',
        ]);
        expect(second.findings.map((x) => x.id)).toContain('mutual-import:b.ts<->c.ts');
    });

    it('a file added or deleted drops or adds its rows, and its neighbours see the new set', () => {
        const first = evaluateRules(FIXTURE_RULES, { files: base() });
        const files = base();
        files.delete('c.ts');
        const second = evaluateRules(FIXTURE_RULES, { files }, first.rows);
        expect([...second.rows.keys()].some((k) => k.endsWith('\u0000c.ts'))).toBe(false);
        expect(second.findings.map((x) => x.id)).toContain('unresolved-import:b.ts->c.ts');
        expect(outcome(second, 'unresolved-import', 'b.ts')).toBe('ran');
        expect(outcome(second, 'unresolved-import', 'd.ts')).toBe('reused');
    });
});

describe('evaluateRules: derive sees only what its trace covers', () => {
    it('a rule that reads an undeclared field sees it empty, and an undeclared extra input absent', () => {
        const seen: unknown[] = [];
        const sneaky = defineRule<{ secret: string }>({
            id: 'sneaky', version: 1, reads: ['body'], scope: 'file',
            derive: ({ focus, files, extra }) => {
                const f = files.get(focus!)!;
                seen.push({ exports: f.exports, imports: f.imports, contentHash: f.contentHash, calls: f.calls.length, secret: extra.secret });
                return [];
            },
        });
        evaluateRules([sneaky], { files: new Map([['a.ts', base().get('a.ts')!]]), extra: { secret: 's' } });
        expect(seen).toEqual([{ exports: [], imports: [], contentHash: '', calls: 1, secret: undefined }]);
    });

    it('a global rule reading only extra inputs (reads: []) sees no files', () => {
        let count = -1;
        const onlyExtra = defineRule<{ g: string }>({
            id: 'only-extra', version: 1, reads: [], scope: 'global', extraInputs: ['g'],
            derive: ({ files }) => { count = files.size; return []; },
        });
        evaluateRules([onlyExtra], { files: base(), extra: { g: 'x' }, extraHashes: { g: 'h' } });
        expect(count).toBe(0);
    });
});

describe('evaluateRules: guard chains', () => {
    const routed = (): Map<string, FileFacts> => new Map([
        ['routes.ts', { ...emptyFacts('r'), entryPoints: [{ kind: 'http-route', key: 'POST /x', symbol: 'h', guards: ['requireAuth'], inputs: [], chain: ['far/auth.ts'] }] }],
        ['far/auth.ts', { ...emptyFacts('m'), exports: [{ name: 'requireAuth', kind: 'function', signatureHash: 's' }] }],
    ]);

    it('a route reads its guard-chain file however far it is, and reruns when that file loses the guard', () => {
        const first = evaluateRules([routeGuard], { files: routed() });
        expect(first.findings).toEqual([]);
        const files = routed();
        files.set('far/auth.ts', { ...files.get('far/auth.ts')!, exports: [] });
        const second = evaluateRules([routeGuard], { files }, first.rows);
        expect(outcome(second, 'route-guard', 'routes.ts')).toBe('ran');
        expect(second.findings.map((x) => x.id)).toEqual(['route-guard:routes.ts#POST /x']);
    });
});

describe('evaluateRules: extra inputs', () => {
    const readsGraph = defineRule<{ graph: string }>({
        id: 'reads-graph', version: 1, reads: ['structure'], scope: 'global', extraInputs: ['graph'],
        derive: ({ extra }) => [{ id: `g:${extra.graph}`, kind: 'g', pack: 'p', nodes: [], severity: 'info', params: {} }],
    });

    it('reruns when the extra input hash moves, and reuses when it holds', () => {
        const files = base();
        const first = evaluateRules([readsGraph], { files, extra: { graph: 'v1' }, extraHashes: { graph: 'h1' } });
        expect(outcome(evaluateRules([readsGraph], { files, extra: { graph: 'v1' }, extraHashes: { graph: 'h1' } }, first.rows), 'reads-graph')).toBe('reused');
        const moved = evaluateRules([readsGraph], { files, extra: { graph: 'v2' }, extraHashes: { graph: 'h2' } }, first.rows);
        expect(outcome(moved, 'reads-graph')).toBe('ran');
        expect(moved.findings.map((x) => x.id)).toEqual(['g:v2']);
    });

    it('throws when a declared extra input has no hash', () => {
        expect(() => evaluateRules([readsGraph], { files: base(), extra: { graph: 'v1' } })).toThrow('rule_extra_input_unhashed:reads-graph:graph');
    });

    it('a rule version bump reruns the rule', () => {
        const first = evaluateRules([ioInLoop], { files: base() });
        const bumped = defineRule({ ...ioInLoop, version: 2 });
        expect(ran(evaluateRules([bumped], { files: base() }, first.rows))).toEqual(['io-in-loop@a.ts', 'io-in-loop@b.ts', 'io-in-loop@c.ts', 'io-in-loop@d.ts']);
    });
});

describe('defineRule', () => {
    it('rejects invalid declarations', () => {
        const ok = { id: 'r', version: 1, reads: ['body'] as const, scope: 'neighbourhood' as const, derive: () => [] };
        expect(() => defineRule({ ...ok, hops: 3 })).toThrow('fact_rule_invalid:r:hops');
        expect(() => defineRule({ ...ok, scope: 'file', hops: 1 })).toThrow('fact_rule_invalid:r:hops-outside-neighbourhood');
        expect(() => defineRule({ ...ok, reads: [] })).toThrow('fact_rule_invalid:r:reads');
        expect(() => defineRule({ ...ok, scope: 'global', reads: [] })).toThrow('fact_rule_invalid:r:reads');
        expect(defineRule<{ g: number }>({ ...ok, scope: 'global', reads: [], extraInputs: ['g'] }).reads).toEqual([]);
        expect(() => defineRule({ ...ok, version: 0 })).toThrow('fact_rule_invalid:r:version');
        expect(() => defineRule({ ...ok, scope: 'global', readsGuardChain: true })).toThrow('fact_rule_invalid:r:guard-chain-outside-neighbourhood');
        expect(defineRule({ ...ok, reads: ['structure', 'body', 'body'] }).reads).toEqual(['body', 'structure']);
    });

    it('evaluateRules refuses two rules with one id', () => {
        expect(() => evaluateRules([exportCount, { ...contentTag, id: 'export-count' }], { files: base() })).toThrow('fact_rule_duplicate:export-count');
    });

    it('uses every fixture rule', () => {
        expect([ioInLoop, exportCount, contentTag, unresolvedImport, routeGuard, mutualImport, callTotal].map((r) => r.id).sort())
            .toEqual(FIXTURE_RULES.map((r) => r.id).sort());
    });
});
