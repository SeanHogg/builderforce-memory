/**
 * knowledge/rules/evaluateRules.ts — the incremental FactRule evaluator (PRD 41 §B1, §B5).
 *
 * Verifying traces with early cutoff (Mokhov, Mitchell, Peyton Jones, *Build Systems à la Carte*): every
 * derived row stores the hash of the inputs it read — the rule's id and version, its focus, and for each
 * file in its scope the hash of each fact kind it `reads`, plus the hash of each declared extra input. On
 * the next evaluation a row whose input hash is unchanged is REUSED without calling `derive`; otherwise
 * the rule reruns. No dirty flags: validity is recomputed from the current facts, so a thrown-away row
 * store rebuilds to the same rows.
 *
 * The trace is ENFORCED, not trusted: `derive` receives each file projected to the kinds the rule reads
 * (`projectFileFacts`, every other field empty) and `extra` cut to its declared `extraInputs`, so an undeclared
 * read sees nothing instead of silently breaking early cutoff.
 *
 * Consequences the trace tests pin: a body-only edit reruns `body`-reading rules for that file and skips
 * `structure`-only rules; a global rule is skipped when no file's read kinds changed anywhere.
 */

import { compareStr } from '../refs.js';
import { sha256Hex } from '../../retrieval/knowledge-index/sha256.js';
import type { Finding } from '../findingModel.js';
import { canonicalFileFacts, projectFileFacts, type FactKind, type FileFacts } from '../facts/fileFacts.js';
import { factKindHashes } from '../facts/factsHash.js';
import { DEFAULT_RULE_HOPS, type FactRule } from './factRule.js';

/** What the evaluator reads: the current facts by path, the import neighbourhood, and the hashed extra inputs. */
export interface RuleWorld<X = Record<never, never>> {
    files: ReadonlyMap<string, FileFacts>;
    /** Undirected one-hop neighbours of a path. Default: from the facts' resolved imports ({@link importNeighbours}). */
    neighbours?: (path: string) => readonly string[];
    extra?: X;
    /** The hash of each extra input. A rule that declares an extra input with no hash here throws `rule_extra_input_unhashed`. */
    extraHashes?: Readonly<Record<string, string>>;
}

/** One derived result and the hash of the inputs it read. `focus` is '' for a global rule. */
export interface DerivedRow {
    rule: string;
    version: number;
    focus: string;
    inputs: string;
    findings: Finding[];
}

export type DerivedRows = ReadonlyMap<string, DerivedRow>;

/** What happened to one row in an evaluation. Sorted by rule, then focus. */
export interface RuleTraceEntry {
    rule: string;
    focus: string;
    outcome: 'ran' | 'reused';
}

export interface RuleEvaluation {
    rows: Map<string, DerivedRow>;
    trace: RuleTraceEntry[];
    /** Every row's findings, deduped by id, sorted by id. */
    findings: Finding[];
}

/** The key of a derived row. */
export const derivedRowKey = (rule: string, focus: string): string => `${rule}\u0000${focus}`;

/** Undirected one-hop import neighbours from each file's resolved imports (only files present), each list sorted. */
export function importNeighbours(files: ReadonlyMap<string, FileFacts>): (path: string) => readonly string[] {
    const adj = new Map<string, Set<string>>();
    const link = (a: string, b: string) => {
        let s = adj.get(a);
        if (!s) adj.set(a, (s = new Set()));
        s.add(b);
    };
    for (const [path, f] of files) {
        for (const i of f.imports) {
            if (i.resolved === undefined || i.resolved === path || !files.has(i.resolved)) continue;
            link(path, i.resolved);
            link(i.resolved, path);
        }
    }
    const sorted = new Map([...adj].map(([k, v]) => [k, [...v].sort(compareStr)] as const));
    return (path) => sorted.get(path) ?? [];
}

/** The paths a neighbourhood evaluation reads: BFS to `hops`, plus the guard-chain files when declared. Sorted. */
function neighbourhoodOf(rule: FactRule<any>, focus: string, world: RuleWorld<any>, neighbours: (p: string) => readonly string[]): string[] {
    const seen = new Set([focus]);
    let frontier = [focus];
    for (let hop = 0; hop < (rule.hops ?? DEFAULT_RULE_HOPS); hop++) {
        const next: string[] = [];
        for (const p of frontier) for (const n of neighbours(p)) if (!seen.has(n)) { seen.add(n); next.push(n); }
        frontier = next;
    }
    if (rule.readsGuardChain) {
        for (const e of world.files.get(focus)?.entryPoints ?? []) for (const c of e.chain) if (world.files.has(c)) seen.add(c);
    }
    return [...seen].sort(compareStr);
}

/** `path\0hash\n` of one kind over every file, hashed once per evaluation and shared by every global rule reading that kind. */
function kindRoots(world: RuleWorld<any>, paths: readonly string[]): (kind: FactKind) => string {
    const roots = new Map<FactKind, string>();
    return (kind) => {
        let root = roots.get(kind);
        if (root === undefined) {
            root = sha256Hex(paths.map((p) => `${p}\u0000${factKindHashes(world.files.get(p)!)[kind]}\n`).join(''));
            roots.set(kind, root);
        }
        return root;
    };
}

/**
 * The trace hash: rule identity, focus, the read kind hashes of each in-scope file (a global rule: the kind roots,
 * which cover every file), each declared extra input's hash.
 */
function inputsHash(rule: FactRule<any>, focus: string, paths: readonly string[], world: RuleWorld<any>, roots: (kind: FactKind) => string): string {
    const parts = [`${rule.id}\u0000${rule.version}\u0000${rule.scope}\u0000${focus}\n`];
    if (rule.scope === 'global') {
        for (const k of rule.reads) parts.push(`#${k}=${roots(k)}\n`);
    } else {
        for (const p of paths) {
            const kinds = factKindHashes(world.files.get(p)!);
            parts.push(p);
            for (const k of rule.reads) parts.push(`\u0000${k}=${kinds[k]}`);
            parts.push('\n');
        }
    }
    for (const name of [...(rule.extraInputs ?? [])].sort(compareStr)) {
        const hash = world.extraHashes?.[name];
        if (hash === undefined) throw new Error(`rule_extra_input_unhashed:${rule.id}:${name}`);
        parts.push(`@${name}=${hash}\n`);
    }
    return sha256Hex(parts.join(''));
}

const byId = (a: Finding, b: Finding) => compareStr(a.id, b.id);

/**
 * Evaluates `rules` against `world`, reusing every row of `previous` whose input hash still holds. Rows for files
 * that no longer exist (and for rules no longer registered) are dropped. Pure: `previous` is not mutated.
 */
export function evaluateRules<X>(rules: readonly FactRule<X>[], world: RuleWorld<X>, previous: DerivedRows = new Map()): RuleEvaluation {
    const ids = new Set<string>();
    for (const r of rules) {
        if (ids.has(r.id)) throw new Error(`fact_rule_duplicate:${r.id}`);
        ids.add(r.id);
    }
    const paths = [...world.files.keys()].sort(compareStr);
    // Canonical (deep-frozen) facts: the hash and projection caches key on them, and a caller's later mutation never reaches a row.
    const w: RuleWorld<X> = { ...world, files: new Map(paths.map((p) => [p, canonicalFileFacts(world.files.get(p)!)] as const)) };
    let neighbours = w.neighbours;
    const rows = new Map<string, DerivedRow>();
    const trace: RuleTraceEntry[] = [];
    const roots = kindRoots(w, paths);

    const evaluate = (rule: FactRule<X>, focus: string, scopePaths: readonly string[]) => {
        const key = derivedRowKey(rule.id, focus);
        // The version is part of the hash, so a bumped rule never matches its old row.
        const inputs = inputsHash(rule, focus, scopePaths, w, roots);
        const prior = previous.get(key);
        if (prior && prior.inputs === inputs) {
            rows.set(key, prior);
            trace.push({ rule: rule.id, focus, outcome: 'reused' });
            return;
        }
        // `derive` sees ONLY what its trace covers: each file projected to the kinds it reads, `extra` cut to its declared inputs.
        // A rule reading no fact kind (extra inputs only) sees no files: not even which paths exist is in its trace.
        const files = new Map(rule.reads.length === 0 ? [] : scopePaths.map((p) => [p, projectFileFacts(w.files.get(p)!, rule.reads)] as const));
        const all = (w.extra ?? {}) as Record<string, unknown>;
        const extra = Object.fromEntries((rule.extraInputs ?? []).filter((n) => n in all).map((n) => [n, all[n]])) as X;
        const findings = rule.derive({ scope: rule.scope, ...(focus ? { focus } : {}), files, extra });
        rows.set(key, { rule: rule.id, version: rule.version, focus, inputs, findings: [...findings].sort(byId) });
        trace.push({ rule: rule.id, focus, outcome: 'ran' });
    };

    for (const rule of [...rules].sort((a, b) => compareStr(a.id, b.id))) {
        if (rule.scope === 'global') { evaluate(rule, '', paths); continue; }
        for (const p of paths) {
            if (rule.scope === 'file') { evaluate(rule, p, [p]); continue; }
            neighbours ??= importNeighbours(w.files);
            evaluate(rule, p, neighbourhoodOf(rule, p, w, neighbours));
        }
    }

    const found = new Map<string, Finding>();
    for (const row of rows.values()) for (const f of row.findings) if (!found.has(f.id)) found.set(f.id, f);
    return { rows, trace, findings: [...found.values()].sort(byId) };
}
