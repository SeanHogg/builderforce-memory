/**
 * src/knowledge/rules/factGraph.oracle.test.ts — the determinism oracle (PRD 41 §B8.6, §8 Part B).
 *
 * For a fixture tree and 500 seeded sequences of edits, adds, deletes, renames and EOL rewrites, the canonical
 * dump of the graph after the INCREMENTAL updates must equal a FROM-SCRATCH build of the final tree, byte for
 * byte. The incremental side writes some files with CRLF (a Windows `core.autocrlf` checkout); the scratch
 * side reads the same tree LF-only (a Linux checkout) — content identity is EOL-normalized (contentHash.ts),
 * so the dumps still agree. The second-OS leg runs in CI; this is its same-process half.
 */

import { createRng, type Rng } from '../../stats/prng.js';
import { contentHash } from '../contentHash.js';
import type { CallFact, FileFacts } from '../facts/fileFacts.js';
import { buildFactGraph, factGraphDump, updateFactGraph, type FactChange, type FactGraph } from './factGraph.js';
import { emptyFacts, FIXTURE_RULES } from './rules.testkit.test-util.js';

/** 500 by default (the §8 done-check); `FACT_ORACLE_SEQUENCES` overrides it for a CI budget. */
const SEQUENCES = Number(process.env.FACT_ORACLE_SEQUENCES) > 0 ? Math.floor(Number(process.env.FACT_ORACLE_SEQUENCES)) : 500;
const STEPS = 6;
const encoder = new TextEncoder();

/** The fixture extractor: one fact per line, EOL-insensitive (as every real extractor must be). */
function extract(text: string): FileFacts {
    const facts = emptyFacts(contentHash(encoder.encode(text)));
    const calls = new Map<string, CallFact>();
    for (const line of text.split(/\r\n|\r|\n/)) {
        const [op, a, b] = line.trim().split(/\s+/);
        if (op === 'import' && a) facts.imports.push({ specifier: a, resolved: a, names: [] });
        else if (op === 'export' && a) {
            facts.exports.push({ name: a, kind: 'function', signatureHash: a });
            facts.symbols.push({ name: a, kind: 'function' });
        } else if (op === 'call' && a) {
            const loop = b === 'loop';
            const key = `${a}|${loop}`;
            const prior = calls.get(key);
            calls.set(key, prior ? { ...prior, count: prior.count + 1 } : { callee: a, enclosing: 'main', awaited: loop || undefined, inLoop: loop || undefined, count: 1 });
        } else if (op === 'route' && a && b) {
            facts.entryPoints.push({ kind: 'http-route', key: `GET ${a}`, symbol: 'main', guards: [], inputs: [], chain: [b] });
        }
    }
    facts.calls.push(...calls.values());
    return facts;
}

const lf = (text: string) => text.replace(/\r\n|\r/g, '\n');
const crlf = (text: string) => lf(text).replace(/\n/g, '\r\n');

const pick = <T>(rng: Rng, xs: readonly T[]): T => xs[Math.floor(rng() * xs.length)]!;
const int = (rng: Rng, n: number) => Math.floor(rng() * n);

function fixtureTree(): Map<string, string> {
    return new Map([
        ['src/a.ts', 'import src/b.ts\nexport a\ncall db.get loop\n'],
        ['src/b.ts', 'import src/c.ts\nexport b\ncall fetch\n// note\n'],
        ['src/c.ts', 'import src/b.ts\nexport c\n'],
        ['src/auth.ts', 'export requireAuth\n'],
        ['src/routes.ts', 'import src/a.ts\nroute /x src/auth.ts\nroute /y src/a.ts\n'],
        ['src/win.ts', crlf('import src/a.ts\nexport w1\nexport w2\nexport w3\ncall kv.put loop\n')],
        ['lib/u.ts', 'call helper\n'],
    ]);
}

/** One random step over the working tree; returns the paths it touched. */
function step(rng: Rng, tree: Map<string, string>, n: number): string[] {
    const paths = [...tree.keys()].sort();
    const path = pick(rng, paths);
    const text = tree.get(path)!;
    const eol = text.includes('\r\n') ? '\r\n' : '\n';
    const append = (line: string) => { tree.set(path, `${text}${line}${eol}`); return [path]; };
    switch (int(rng, 10)) {
        case 0: return append(`// comment ${n}`);
        case 1: return append(`call fn${int(rng, 4)}${rng() < 0.5 ? ' loop' : ''}`);
        case 2: return append(`export e${int(rng, 5)}`);
        case 3: return append(`import ${pick(rng, paths)}`);
        case 4: return append(`route /r${int(rng, 3)} ${pick(rng, [...paths, 'src/auth.ts'])}`);
        case 5: {
            const lines = text.split(eol).filter((l) => l !== '');
            if (lines.length === 0) return [];
            lines.splice(int(rng, lines.length), 1);
            tree.set(path, lines.map((l) => `${l}${eol}`).join(''));
            return [path];
        }
        case 6:
            if (tree.size <= 3) return [];
            tree.delete(path);
            return [path];
        case 7: {
            const added = `src/new${n}.ts`;
            tree.set(added, `import ${pick(rng, paths)}\nexport n${n}\ncall fn${int(rng, 4)} loop\n`);
            return [added];
        }
        case 8: {
            const to = `moved/${n}.ts`;
            tree.delete(path);
            tree.set(to, text);
            return [path, to];
        }
        default:
            tree.set(path, eol === '\r\n' ? lf(text) : crlf(text));
            return [path];
    }
}

const changesOf = (tree: ReadonlyMap<string, string>, paths: readonly string[]): FactChange[] =>
    paths.map((path) => ({ path, facts: tree.has(path) ? extract(tree.get(path)!) : null }));

/** The Linux checkout of the same tree: LF only. */
const scratchOf = (tree: ReadonlyMap<string, string>) =>
    factGraphDump(buildFactGraph([...tree].map(([p, t]) => [p, extract(lf(t))] as const), FIXTURE_RULES).graph);

describe('determinism oracle (B8.6): incremental dump === scratch dump', () => {
    it(`holds byte for byte over ${SEQUENCES} seeded edit sequences, with CRLF files on the incremental side`, () => {
        let reused = 0;
        for (let seed = 0; seed < SEQUENCES; seed++) {
            const rng = createRng(seed);
            const tree = fixtureTree();
            let graph: FactGraph = buildFactGraph([...tree].map(([p, t]) => [p, extract(t)] as const), FIXTURE_RULES).graph;
            for (let n = 0; n < STEPS; n++) {
                const touched = step(rng, tree, n);
                const update = updateFactGraph(graph, changesOf(tree, touched), FIXTURE_RULES);
                reused += update.trace.filter((t) => t.outcome === 'reused').length;
                graph = update.graph;
            }
            const incremental = factGraphDump(graph);
            const scratch = scratchOf(tree);
            if (incremental !== scratch) throw new Error(`oracle failed at seed ${seed}`);
        }
        // The oracle exercised the incremental path, not a full recompute each step.
        expect(reused).toBeGreaterThan(SEQUENCES * STEPS);
    });

    it('an EOL-only rewrite (autocrlf) is no change at all: every row reused, the dump unmoved', () => {
        const tree = fixtureTree();
        const built = buildFactGraph([...tree].map(([p, t]) => [p, extract(t)] as const), FIXTURE_RULES);
        const update = updateFactGraph(built.graph, [...tree].map(([path, t]) => ({ path, facts: extract(crlf(t)) })), FIXTURE_RULES);
        expect(update.changed).toEqual([]);
        expect(update.trace.every((t) => t.outcome === 'reused')).toBe(true);
        expect(factGraphDump(update.graph)).toBe(factGraphDump(built.graph));
    });

    it('a Windows (CRLF, backslashed paths) checkout and a Linux (LF) checkout build the same dump', () => {
        const tree = fixtureTree();
        const bs = (p: string) => p.replace(/\//g, '\\');
        const windowsFacts = (t: string): FileFacts => {
            const f = extract(crlf(t));
            return {
                ...f,
                imports: f.imports.map((i) => ({ ...i, resolved: bs(i.resolved!) })),
                entryPoints: f.entryPoints.map((e) => ({ ...e, chain: e.chain.map(bs) })),
            };
        };
        const windows = factGraphDump(buildFactGraph([...tree].map(([p, t]) => [bs(p), windowsFacts(t)] as const), FIXTURE_RULES).graph);
        expect(windows).toBe(scratchOf(tree));
    });

    it('`changed` compares against the pre-batch graph: an edit reverted in the same batch is no change', () => {
        const tree = fixtureTree();
        const built = buildFactGraph([...tree].map(([p, t]) => [p, extract(t)] as const), FIXTURE_RULES);
        const a = tree.get('src/a.ts')!;
        const update = updateFactGraph(built.graph, [
            { path: 'src/a.ts', facts: extract(`${a}export z\n`) }, { path: 'src/a.ts', facts: extract(a) },
            { path: 'src/tmp.ts', facts: extract('export t\n') }, { path: 'src/tmp.ts', facts: null },
        ], FIXTURE_RULES);
        expect(update.changed).toEqual([]);
        expect(factGraphDump(update.graph)).toBe(factGraphDump(built.graph));
    });

    it('the dump is independent of the order files arrive in', () => {
        const entries = [...fixtureTree()].map(([p, t]) => [p, extract(t)] as const);
        const forward = factGraphDump(buildFactGraph(entries, FIXTURE_RULES).graph);
        const reverse = factGraphDump(buildFactGraph([...entries].reverse(), FIXTURE_RULES).graph);
        expect(reverse).toBe(forward);
    });
});
