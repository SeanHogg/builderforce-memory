/**
 * knowledge/rules/factGraph.ts — the content-addressed fact graph in memory (PRD 41 §B1, §B8.6): the
 * manifest (path → content hash, facts hash, kind hashes), the facts objects by facts hash, and the
 * derived rule rows with their traces. Pure and immutable: an update returns a new graph.
 *
 * {@link factGraphDump} is the canonical dump the determinism oracle compares: sorted canonical JSON of
 * the manifest, the referenced objects and the derived rows — never a trace outcome, a time or an order of
 * arrival. An incremental graph and a from-scratch build of the same files dump to the same bytes.
 * A durable host persists the same three tables behind its store; this module does no I/O.
 */

import { canonicalJson, compareStr, isRootFilePath, normalizePath } from '../refs.js';
import type { Finding } from '../findingModel.js';
import { canonicalFileFacts, factKindHashes, factsHash, type FactKind, type FileFacts } from '../facts/fileFacts.js';
import type { FactRule } from './factRule.js';
import { evaluateRules, type DerivedRows, type RuleTraceEntry, type RuleWorld } from './evaluateRules.js';

export interface FactManifestEntry {
    contentHash: string;
    factsHash: string;
    kinds: Readonly<Record<FactKind, string>>;
}

export interface FactGraph {
    manifest: ReadonlyMap<string, FactManifestEntry>;
    /** Facts by facts hash — only those the manifest references. */
    objects: ReadonlyMap<string, FileFacts>;
    derived: DerivedRows;
}

/** One file's new facts, or `null` when it was deleted. A rename is a delete plus an add. */
export interface FactChange {
    path: string;
    facts: FileFacts | null;
}

export type FactGraphOptions<X> = Pick<RuleWorld<X>, 'neighbours' | 'extra' | 'extraHashes'>;

export interface FactGraphUpdate {
    graph: FactGraph;
    trace: RuleTraceEntry[];
    findings: Finding[];
    /** Paths whose facts hash changed (added, edited or deleted), sorted. Empty ⇒ every rule row was reused (early cutoff). */
    changed: string[];
}

/** The current dump format. */
export const FACT_GRAPH_DUMP_VERSION = 1;

export const emptyFactGraph = (): FactGraph => ({ manifest: new Map(), objects: new Map(), derived: new Map() });

/** The ONE path form of the graph (§B8.1); throws `fact_graph_bad_path` for a path outside the root. */
function pathOf(raw: string): string {
    const p = normalizePath(raw);
    if (!isRootFilePath(p)) throw new Error(`fact_graph_bad_path:${raw}`);
    return p;
}

/**
 * Applies `changes` (the last change of a path wins) and re-evaluates `rules`, reusing every derived row whose
 * trace still holds. Unreferenced objects are dropped, so the graph never depends on its history.
 */
export function updateFactGraph<X>(graph: FactGraph, changes: readonly FactChange[], rules: readonly FactRule<X>[], opts: FactGraphOptions<X> = {}): FactGraphUpdate {
    const manifest = new Map(graph.manifest);
    const objects = new Map(graph.objects);
    const touched = new Set<string>();
    for (const c of changes) {
        const path = pathOf(c.path);
        touched.add(path);
        if (c.facts === null) { manifest.delete(path); continue; }
        // Canonical facts are deep-frozen: the stored object can never be mutated under its hash.
        const facts = canonicalFileFacts(c.facts);
        const hash = factsHash(facts);
        if (manifest.get(path)?.factsHash === hash) continue;
        if (!objects.has(hash)) objects.set(hash, facts);
        manifest.set(path, { contentHash: facts.contentHash, factsHash: hash, kinds: factKindHashes(objects.get(hash)!) });
    }
    // Against the PRE-batch manifest, so a path edited and reverted (or added and deleted) in one batch is no change.
    const changed = [...touched].filter((p) => graph.manifest.get(p)?.factsHash !== manifest.get(p)?.factsHash);
    const live = new Set([...manifest.values()].map((e) => e.factsHash));
    for (const h of [...objects.keys()]) if (!live.has(h)) objects.delete(h);

    const files = new Map([...manifest].map(([p, e]) => [p, objects.get(e.factsHash)!] as const));
    const result = evaluateRules(rules, { files, ...opts }, graph.derived);
    return {
        graph: { manifest, objects, derived: result.rows },
        trace: result.trace,
        findings: result.findings,
        changed: changed.sort(compareStr),
    };
}

/** A from-scratch build: every file added to an empty graph. */
export function buildFactGraph<X>(files: Iterable<readonly [string, FileFacts]>, rules: readonly FactRule<X>[], opts: FactGraphOptions<X> = {}): FactGraphUpdate {
    return updateFactGraph(emptyFactGraph(), [...files].map(([path, facts]) => ({ path, facts })), rules, opts);
}

/** The canonical dump: sorted canonical JSON of the manifest, the referenced objects and the derived rows. */
export function factGraphDump(graph: FactGraph): string {
    const sortedEntries = <V>(m: ReadonlyMap<string, V>) => [...m].sort((a, b) => compareStr(a[0], b[0]));
    return canonicalJson({
        version: FACT_GRAPH_DUMP_VERSION,
        manifest: sortedEntries(graph.manifest).map(([path, e]) => ({ path, ...e })),
        objects: sortedEntries(graph.objects).map(([hash, facts]) => ({ hash, facts })),
        derived: sortedEntries(graph.derived).map(([, row]) => row),
    });
}
