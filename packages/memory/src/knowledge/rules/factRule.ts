/**
 * knowledge/rules/factRule.ts — the FactRule contract (PRD 41 §B5): derived knowledge as rules over facts.
 *
 * A rule declares what it READS (fact kinds, and optionally named non-fact inputs such as the knowledge
 * graph), its SCOPE, and for a neighbourhood rule its HOPS. The evaluator (evaluateRules.ts) uses exactly
 * those declarations as the rule's dependency trace: a rule reruns only when one of its read inputs changed
 * (Build Systems à la Carte: verifying traces + early cutoff). So `derive` must read nothing else — the
 * determinism oracle (factGraph.oracle.test.ts) catches a rule that does.
 *
 *   - `file`: one evaluation per file, reading that file's facts.
 *   - `neighbourhood`: one per file, reading the file plus its import neighbours up to `hops` (default 1,
 *     max 2), plus — with `readsGuardChain` — the guard-chain files of its entry points.
 *   - `global`: one evaluation, reading every file.
 *
 * One rule per file under knowledge/rules/, registered as data (findings.ts is the generic registry).
 */

import type { Finding } from '../findingModel.js';
import type { FactKind, FileFacts } from '../facts/fileFacts.js';
import { FACT_KINDS } from '../facts/fileFacts.js';

export type RuleScope = 'file' | 'neighbourhood' | 'global';

export const DEFAULT_RULE_HOPS = 1;
/** The C2 impact-set bound: a neighbourhood never reaches further than this. */
export const MAX_RULE_HOPS = 2;

/** What one evaluation of a rule sees. */
export interface RuleInput<X = Record<never, never>> {
    scope: RuleScope;
    /** `file` / `neighbourhood`: the file the evaluation is for. Absent for `global`. */
    focus?: string;
    /** The facts in scope, by root-relative path (iterate in sorted order): the focus; the focus plus its neighbourhood; every file. */
    files: ReadonlyMap<string, FileFacts>;
    /** The named non-fact inputs the host supplies (only those the rule declares in `extraInputs` are part of its trace). */
    extra: X;
}

export interface FactRule<X = Record<never, never>> {
    /** 'cycle', 'partition-violation', 'unused-export', 'io-in-loop', … */
    id: string;
    /** Bump whenever `derive` can return different findings for the same inputs: it is part of the cache key. */
    version: number;
    /** The fact kinds it consumes — its dependency trace. Empty only for a global rule reading extra inputs alone. */
    reads: readonly FactKind[];
    scope: RuleScope;
    /** Neighbourhood rules: import hops to read around the focus (1 or 2). */
    hops?: number;
    /** Neighbourhood rules: also read the facts of the guard-chain files of the focus's entry points. */
    readsGuardChain?: boolean;
    /** Names of `extra` inputs `derive` reads; each needs a hash from the host (`RuleWorld.extraHashes`). */
    extraInputs?: readonly (keyof X & string)[];
    /** Pure. Findings sorted by id (the evaluator sorts again). */
    derive(input: RuleInput<X>): Finding[];
}

/**
 * Validates and freezes a rule: a non-empty id, an integer version ≥ 1, known `reads` — empty only for a global rule
 * that reads extra inputs alone (it then sees no files) — and `hops` only on a neighbourhood rule and within
 * 1..{@link MAX_RULE_HOPS}. Throws `fact_rule_invalid:<id>:<why>`.
 */
export function defineRule<X = Record<never, never>>(rule: FactRule<X>): FactRule<X> {
    const fail = (why: string): never => { throw new Error(`fact_rule_invalid:${rule.id}:${why}`); };
    if (!rule.id) fail('id');
    if (!Number.isInteger(rule.version) || rule.version < 1) fail('version');
    if (rule.reads.some((k) => !FACT_KINDS.includes(k))) fail('reads');
    if (rule.reads.length === 0 && (rule.scope !== 'global' || (rule.extraInputs ?? []).length === 0)) fail('reads');
    if (rule.hops !== undefined) {
        if (rule.scope !== 'neighbourhood') fail('hops-outside-neighbourhood');
        if (!Number.isInteger(rule.hops) || rule.hops < 1 || rule.hops > MAX_RULE_HOPS) fail('hops');
    }
    if (rule.readsGuardChain && rule.scope !== 'neighbourhood') fail('guard-chain-outside-neighbourhood');
    return Object.freeze({ ...rule, reads: Object.freeze([...new Set(rule.reads)].sort()) });
}
