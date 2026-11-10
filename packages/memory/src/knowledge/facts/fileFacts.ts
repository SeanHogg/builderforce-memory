/**
 * knowledge/facts/fileFacts.ts — the per-file fact contract, `FileFacts` v3 (PRD 41 §B1, §B3, §4).
 *
 * What ONE extractor (tree-sitter in Node hosts, regex in the Worker) emits for one file. Pure data: no
 * positions (a line number would move every fact on an edit above it and defeat early cutoff), every
 * array sorted by a declared key ({@link canonicalFileFacts}), canonical-serialized for every hash.
 *
 * The fields split into three FACT KINDS ({@link FACT_KIND_FIELDS}), each with its own hash, so a rule
 * that `reads` only `structure` never reruns for a body or comment edit (early cutoff, §B1):
 *   - `structure`: symbols, imports, exports, entry points (the declared surface; a guard added to or
 *     removed from a route registration moves it);
 *   - `body`: calls, complexity, capability imports, test facts;
 *   - `content`: the EOL-normalized bytes themselves, i.e. `contentHash` (contentHash.ts).
 *
 * The header (`factsVersion`, `extractor`, `language`) enters every kind's hash: facts of another
 * extractor are never "the same structure". The extractors live in the hosts (Builderforce.ai
 * agent-tools); memory owns only this shape and its hashing.
 */

import { canonicalJson, compareStr, normalizePath, sortBy } from '../refs.js';
import { sha256Hex } from '../../retrieval/knowledge-index/sha256.js';
import { FACTS_EXTRACTOR_VERSION } from './extractorVersion.js';

/** The three fact kinds a rule may read (§B1). Sorted. */
export type FactKind = 'body' | 'content' | 'structure';
export const FACT_KINDS: readonly FactKind[] = Object.freeze(['body', 'content', 'structure'] as const);

/** A definition. `name` is the symbol path (`Class.method`); `signatureHash` hashes its normalized signature text. */
export interface SymbolFact {
    name: string;
    /** 'function' | 'class' | 'method' | 'interface' | 'type' | 'enum' | 'variable' | 'constant' | 'module' … */
    kind: string;
    signatureHash?: string;
}

/** One import statement. `resolved` is the root-relative target file when the host resolved it. */
export interface ImportFact {
    specifier: string;
    resolved?: string;
    /** Imported binding names (sorted); `*` for a namespace import, `default` for the default binding. */
    names: string[];
    typeOnly?: boolean;
    dynamic?: boolean;
}

/** One exported name (§B3): its kind and the hash of its signature, so a changed signature is a changed export (C7 `api-breaking`). */
export interface ExportFact {
    name: string;
    kind: string;
    signatureHash: string;
    /** A re-export's source specifier. */
    from?: string;
}

/** Calls of one callee from one enclosing symbol, deduped by every field but `count` (no positions). */
export interface CallFact {
    /** The callee as written (`db.select`, `fetch`, `fs.readFileSync`). */
    callee: string;
    /** The enclosing symbol path; '' at module top level. */
    enclosing: string;
    awaited?: boolean;
    /** Inside `for` / `while` / `.map` / `.forEach` / `.reduce` (C6.1 `io-in-loop`). */
    inLoop?: boolean;
    count: number;
}

/** McCabe cyclomatic complexity of one function-like symbol. */
export interface ComplexityFact {
    symbol: string;
    cyclomatic: number;
}

export type EntryPointKind = 'http-route' | 'ws-handler' | 'webview-message' | 'vscode-command' | 'cli-command'
    | 'mcp-tool' | 'queue-consumer' | 'cron' | 'exported-api';

/** One entry point (§C5.1, Manadhata–Wing attack surface). */
export interface EntryPointFact {
    kind: EntryPointKind;
    /** e.g. 'POST /api/projects/:id/knowledge/write'. */
    key: string;
    /** The handler symbol path. */
    symbol: string;
    /** Guard names found on its chain (middleware, wrappers), sorted. Which names are guards is registry data (run-delta). */
    guards: string[];
    /** Untrusted input accessors the handler uses (`req.body`, `c.req.json`, `msg.data`, `argv`), sorted. */
    inputs: string[];
    /**
     * The GUARD CHAIN (§B5): root-relative files of the router, middleware and wrappers resolved from the route
     * registration, sorted. A neighbourhood rule with `readsGuardChain` reads these files' facts whatever their import distance.
     */
    chain: string[];
}

export type CapabilityKind = 'process-exec' | 'fs-write' | 'network-client' | 'network-server' | 'dynamic-code' | 'native-addon'
    | 'crypto' | 'env-read' | 'secret-read';

/** One capability the file reaches (§C5.3): the module (and member) that grants it, and the algorithm for crypto. */
export interface CapabilityImportFact {
    capability: CapabilityKind;
    module: string;
    symbol?: string;
    /** Crypto only: the algorithm named at the call (`md5`, `sha1`, `aes-128-ecb`); weak ones are registry data. */
    algorithm?: string;
}

/** Matcher strength class (§C8.2), strongest first: `exact` (toEqual, toBe) > `partial` (toMatchObject, toContain) > `presence` (toBeTruthy, toBeDefined). */
export type MatcherStrength = 'exact' | 'partial' | 'presence';
export const MATCHER_STRENGTH_ORDER: readonly MatcherStrength[] = Object.freeze(['exact', 'partial', 'presence'] as const);

export interface MatcherUse {
    matcher: string;
    strength: MatcherStrength;
    count: number;
}

export type TestModifier = 'skip' | 'only' | 'todo';

/** One test case (§C8.2). */
export interface TestCaseFact {
    /** The full title path, `describe › it`. */
    name: string;
    assertions: number;
    matchers: MatcherUse[];
    /** `.skip` / `.only` / `.todo` / `xit`. */
    modifier?: TestModifier;
    timeoutMs?: number;
    /** Hash of the test's normalized expected-value texts, so an edited expectation is detectable (`assertion-edited-with-impl`). */
    expectedHash?: string;
}

/** A test file's facts (§C8.2). Absent on a file that is not a test file. */
export interface TestFacts {
    framework?: string;
    tests: TestCaseFact[];
    /** Snapshot files the tests read, root-relative, sorted. */
    snapshotFiles: string[];
}

/** FileFacts v3. Every array is sorted by its declared key ({@link canonicalFileFacts}). */
export interface FileFacts {
    /** {@link FACTS_EXTRACTOR_VERSION} for v3; 2 for facts read through {@link readFileFacts} from a v2 producer. */
    factsVersion: number;
    /** The composite extractor tag for this file's language (`factsExtractorTag`, e.g. `tree-sitter@3.0/typescript=0.23.2/eol-lf@1`). */
    extractor: string;
    language: string;
    /** `contentHash()` of the file's bytes — the `content` kind. */
    contentHash: string;
    // structure
    symbols: SymbolFact[];
    imports: ImportFact[];
    exports: ExportFact[];
    entryPoints: EntryPointFact[];
    // body
    calls: CallFact[];
    complexity: ComplexityFact[];
    capabilityImports: CapabilityImportFact[];
    testFacts?: TestFacts;
}

/** Which fields each kind hashes (data: the ONE place the split is declared). */
export const FACT_KIND_FIELDS: Readonly<Record<FactKind, readonly (keyof FileFacts)[]>> = Object.freeze({
    structure: ['symbols', 'imports', 'exports', 'entryPoints'],
    body: ['calls', 'complexity', 'capabilityImports', 'testFacts'],
    content: ['contentHash'],
});

const str = (v: unknown) => (v === undefined ? '' : String(v));
const sorted = (xs: readonly string[] | undefined) => [...(xs ?? [])].sort(compareStr);
const paths = (xs: readonly string[] | undefined) => sorted((xs ?? []).map(normalizePath));
/** The final sort key of every list: the whole item, so items equal on the declared keys still sort one way. */
const whole = (x: unknown) => canonicalJson(x);

/** Facts objects produced by {@link canonicalFileFacts}: deep-frozen, so every cache keyed by object identity holds. */
const canonicalSet = new WeakSet<FileFacts>();

function deepFreeze<T>(value: T): T {
    if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
        for (const v of Object.values(value as Record<string, unknown>)) deepFreeze(v);
        Object.freeze(value);
    }
    return value;
}

/**
 * The canonical form: a deep copy with every array sorted by its declared keys then by the whole item, nested string
 * lists sorted, and every path field (`imports[].resolved`, `entryPoints[].chain`, `testFacts.snapshotFiles`) in the
 * ONE path form (`normalizePath`, §B8.1). The result is DEEP-FROZEN: facts are immutable values, and the kind-hash and
 * projection caches key on the object. Idempotent; a canonical object is returned as is.
 */
export function canonicalFileFacts(f: FileFacts): FileFacts {
    if (canonicalSet.has(f)) return f;
    const out: FileFacts = {
        ...f,
        symbols: sortBy(f.symbols.map((s) => ({ ...s })), (s) => s.name, (s) => s.kind, (s) => str(s.signatureHash), whole),
        imports: sortBy(
            f.imports.map((i) => ({ ...i, names: sorted(i.names), ...(i.resolved !== undefined ? { resolved: normalizePath(i.resolved) } : {}) })),
            (i) => i.specifier, (i) => str(i.resolved), (i) => i.names.join('\u0000'), whole,
        ),
        exports: sortBy(f.exports.map((e) => ({ ...e })), (e) => e.name, (e) => e.kind, (e) => e.signatureHash, (e) => str(e.from), whole),
        entryPoints: sortBy(
            f.entryPoints.map((e) => ({ ...e, guards: sorted(e.guards), inputs: sorted(e.inputs), chain: paths(e.chain) })),
            (e) => e.kind, (e) => e.key, (e) => e.symbol, whole,
        ),
        calls: sortBy(f.calls.map((c) => ({ ...c })), (c) => c.enclosing, (c) => c.callee, (c) => str(c.awaited), (c) => str(c.inLoop), whole),
        complexity: sortBy(f.complexity.map((c) => ({ ...c })), (c) => c.symbol, whole),
        capabilityImports: sortBy(f.capabilityImports.map((c) => ({ ...c })), (c) => c.capability, (c) => c.module, (c) => str(c.symbol), (c) => str(c.algorithm), whole),
    };
    if (f.testFacts) {
        out.testFacts = {
            ...f.testFacts,
            tests: sortBy(
                f.testFacts.tests.map((t) => ({ ...t, matchers: sortBy(t.matchers.map((m) => ({ ...m })), (m) => m.matcher, (m) => m.strength, whole) })),
                (t) => t.name, whole,
            ),
            snapshotFiles: paths(f.testFacts.snapshotFiles),
        };
    }
    deepFreeze(out);
    canonicalSet.add(out);
    return out;
}

/** The header every kind hash includes. */
const headerOf = (f: FileFacts) => ({ factsVersion: f.factsVersion, extractor: f.extractor, language: f.language });

const kindCache = new WeakMap<FileFacts, Readonly<Record<FactKind, string>>>();

/**
 * The hash of each fact kind of `f` (canonicalized first). `content` is the content hash itself; `structure` and `body`
 * are sha256 over the canonical JSON of the header plus that kind's fields. Cached per CANONICAL (deep-frozen) object;
 * a non-canonical input is canonicalized and hashed again on every call, so mutating it can never serve a stale hash.
 */
export function factKindHashes(f: FileFacts): Readonly<Record<FactKind, string>> {
    const c = canonicalFileFacts(f);
    const hit = kindCache.get(c);
    if (hit) return hit;
    const of = (kind: FactKind) => {
        const body: Record<string, unknown> = { header: headerOf(c) };
        for (const field of FACT_KIND_FIELDS[kind]) body[field] = c[field];
        return sha256Hex(canonicalJson(body));
    };
    const hashes = Object.freeze({ body: of('body'), content: c.contentHash, structure: of('structure') });
    kindCache.set(c, hashes);
    return hashes;
}

/** The empty value of every kind's fields: what a rule that does not read that kind sees. */
const EMPTY_FIELDS: Readonly<Record<FactKind, Partial<FileFacts>>> = Object.freeze({
    structure: { symbols: [], imports: [], exports: [], entryPoints: [] },
    body: { calls: [], complexity: [], capabilityImports: [], testFacts: undefined },
    content: { contentHash: '' },
});

const projectionCache = new WeakMap<FileFacts, Map<string, FileFacts>>();

/**
 * `f` (canonicalized) as a rule that reads `reads` sees it: the header plus the fields of the read kinds; every other
 * kind's fields empty. So an undeclared read sees nothing, and can never hide a dependency from the trace (§B5).
 * Deep-frozen; cached per (canonical object, reads).
 */
export function projectFileFacts(f: FileFacts, reads: readonly FactKind[]): FileFacts {
    const c = canonicalFileFacts(f);
    const key = [...reads].sort().join(',');
    let byReads = projectionCache.get(c);
    if (!byReads) projectionCache.set(c, (byReads = new Map()));
    let hit = byReads.get(key);
    if (!hit) {
        const out: FileFacts = { ...c };
        for (const kind of FACT_KINDS) if (!reads.includes(kind)) Object.assign(out, EMPTY_FIELDS[kind]);
        if (out.testFacts === undefined) delete out.testFacts;
        hit = deepFreeze(out);
        byReads.set(key, hit);
    }
    return hit;
}

/** The content address of a facts object: sha256 of its canonical JSON. */
export function factsHash(f: FileFacts): string {
    return sha256Hex(canonicalJson(canonicalFileFacts(f)));
}

/**
 * Reads facts from any producer: a v3 object as is (canonicalized), a v2 one (repo-map's symbols + imports, no
 * `factsVersion`) with every v3 list empty and `factsVersion: 2`. A v2 fact set is never comparable with a v3 one
 * (its `factsVersion` and extractor tag differ, C9), so the empty lists never read as "everything was removed".
 */
export function readFileFacts(raw: Partial<FileFacts> & Pick<FileFacts, 'contentHash'>): FileFacts {
    return canonicalFileFacts({
        factsVersion: raw.factsVersion ?? 2,
        extractor: raw.extractor ?? '',
        language: raw.language ?? '',
        contentHash: raw.contentHash,
        symbols: raw.symbols ?? [],
        imports: raw.imports ?? [],
        exports: raw.exports ?? [],
        entryPoints: raw.entryPoints ?? [],
        calls: raw.calls ?? [],
        complexity: raw.complexity ?? [],
        capabilityImports: raw.capabilityImports ?? [],
        ...(raw.testFacts ? { testFacts: raw.testFacts } : {}),
    });
}

/** Whether a facts object is v3 (current shape). */
export const isFileFactsV3 = (f: Pick<FileFacts, 'factsVersion'>): boolean => f.factsVersion === FACTS_EXTRACTOR_VERSION;
