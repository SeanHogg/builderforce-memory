/**
 * knowledge/facts/fileFacts.ts — the per-file fact contract, `FileFacts` v3 (PRD 41 §B1, §B3, §4).
 *
 * What ONE extractor (tree-sitter in Node hosts, regex in the Worker) emits for one file. Pure data: no
 * positions (a line number would move every fact on an edit above it and defeat early cutoff), every
 * array sorted by a declared key ({@link canonicalFileFacts}), canonical-serialized for every hash.
 *
 * The fields split into three FACT KINDS ({@link FACT_KIND_FIELDS}), each with its own hash, so a rule
 * that `reads` only `structure` never reruns for a body or comment edit (early cutoff, §B1):
 *   - `structure`: symbols, imports, exports, entry points, router mounts and `use` registrations (the
 *     declared surface; a guard added to or removed from a route registration moves it);
 *   - `body`: calls, complexity, capability imports, test facts;
 *   - `content`: the EOL-normalized bytes themselves, i.e. `contentHash` (contentHash.ts).
 *
 * The header (`factsVersion`, `extractor`, `language`) enters every kind's hash: facts of another
 * extractor are never "the same structure". The extractors live in the hosts (Builderforce.ai
 * agent-tools); memory owns only this shape and its hashing (the hashes and the wire form: factsHash.ts).
 */

import { canonicalJson, compareStr, normalizePath, sortBy } from '../refs.js';
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
    /**
     * Guard names on the REGISTRATION itself (inline middleware arguments, wrappers around the handler), sorted. Which names
     * are guards is registry data (run-delta). File-level `use(path?, guard)` registrations are recorded in
     * {@link FileFacts.uses} and composed by `effectiveGuards` (guards.ts): a producer that records `uses` does not also fold
     * them in here. A producer with no `uses` may fold them, by the same rule (a `use` guards only routes registered AFTER it
     * whose path its pattern covers).
     */
    guards: string[];
    /** Untrusted input accessors the handler uses (`req.body`, `c.req.json`, `msg.data`, `argv`), sorted. */
    inputs: string[];
    /**
     * The GUARD CHAIN (§B5): root-relative files of the router, middleware and wrappers resolved from the route
     * registration, sorted. A neighbourhood rule with `readsGuardChain` reads these files' facts whatever their import distance.
     */
    chain: string[];
    /**
     * The registration ORDER within the file: the 0-based index of this registration among the file's route, `use` and
     * mount registrations (ONE counter shared with {@link UseFact.order} and {@link MountFact.order}). A registration index,
     * not a line number and not a byte offset, so an edit that adds no registration never moves it. A non-negative integer.
     * A producer that emits `uses` MUST set `order` on every entry point; absent ⇒ unknown, and `effectiveGuards` applies
     * every covering `use` but reports those guards as inferred (`guardsInferred`).
     */
    order?: number;
    /** The local router binding the route is registered on (`app`, `adminRouter`). Absent ⇒ the file's one default router. */
    router?: string;
}

/**
 * One file-level middleware registration (§C5.1): `app.use(path?, guard…)`. It guards the routes of its router registered
 * AFTER it (by `order`) whose path `path` covers, and every sub-router that router mounts after it ({@link MountFact}).
 */
export interface UseFact {
    /**
     * The use-path pattern: '' (or '*', '/*') for every path; `:param` (also `:param?`, `:param{regex}`) matches one segment;
     * a trailing `*` (`/api/*`, `/api*`) matches the rest.
     */
    path: string;
    /**
     * `prefix` (Express `use`, the default; canonical form omits it): the pattern covers every path under it.
     * `exact` (Hono `use('/api')`): that path only.
     */
    match?: 'prefix' | 'exact';
    /** Guard names it registers, sorted. */
    guards: string[];
    /** Registration order (see {@link EntryPointFact.order}). */
    order: number;
    /** The router binding it is registered on (see {@link EntryPointFact.router}). */
    router?: string;
}

/**
 * One sub-router mount (§C5.1): `app.route('/api', sub)` (Hono) / `app.use('/api', subRouter)` (Express). The sub-router's
 * routes answer under `prefix`, and the mounting router's `use` guards registered before the mount apply to them.
 */
export interface MountFact {
    /** The mount prefix ('/api'); '/' for the root. Canonical form: leading `/`, no trailing `/`. */
    prefix: string;
    /**
     * The sub-router: an import specifier resolved to a file by the host (`ResolvedMount`), or the name of a LOCAL router
     * binding of this file (one that registers routes, uses or mounts here), which `effectiveGuards` resolves itself.
     */
    target: string;
    /** Registration order (see {@link EntryPointFact.order}). */
    order: number;
    /** The router binding that registers the mount (see {@link EntryPointFact.router}). */
    router?: string;
    /** Guards given inline at the mount (`router.use('/api', requireAuth, apiRouter)`), sorted; applied to every mounted route. */
    guards?: string[];
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
    /** Sub-router mounts (§C5.1). Optional on input; a canonical object always carries it (`[]` when none). */
    mounts?: MountFact[];
    /** File-level `use()` middleware registrations (§C5.1). Optional on input; a canonical object always carries it. */
    uses?: UseFact[];
    // body
    calls: CallFact[];
    complexity: ComplexityFact[];
    capabilityImports: CapabilityImportFact[];
    testFacts?: TestFacts;
}

/** Which fields each kind hashes (data: the ONE place the split is declared). */
export const FACT_KIND_FIELDS: Readonly<Record<FactKind, readonly (keyof FileFacts)[]>> = Object.freeze({
    structure: ['symbols', 'imports', 'exports', 'entryPoints', 'mounts', 'uses'],
    body: ['calls', 'complexity', 'capabilityImports', 'testFacts'],
    content: ['contentHash'],
});

const str = (v: unknown) => (v === undefined ? '' : String(v));
const sorted = (xs: readonly string[] | undefined) => [...(xs ?? [])].sort(compareStr);
const paths = (xs: readonly string[] | undefined) => sorted((xs ?? []).map(normalizePath));
/** The final sort key of every list: the whole item, so items equal on the declared keys still sort one way. */
const whole = (x: unknown) => canonicalJson(x);
/** A registration order as a string that sorts numerically. */
const ord = (n: number) => String(n).padStart(16, '0');

/** Refuses a registration order that is not a non-negative safe integer: `file_facts_invalid_order:<what>:<value>`. */
function checkOrder(what: string, order: number | undefined): void {
    if (order === undefined) return;
    if (!Number.isSafeInteger(order) || order < 0) throw new Error(`file_facts_invalid_order:${what}:${String(order)}`);
}

/** The ONE mount-prefix form: leading `/`, no trailing or doubled `/` ('' and '/' → '/', 'api/' → '/api'). */
export function normalizeMountPrefix(prefix: string): string {
    return `/${prefix.split('/').filter((s) => s !== '').join('/')}`;
}

/**
 * The ONE `use`-path form: '' stays '' (every path); anything else as {@link normalizeMountPrefix} ('/admin/' → '/admin',
 * '*' → '/*'), so a trailing or doubled `/` never moves a hash.
 */
const normalizeUsePath = (path: string): string => (path === '' ? '' : normalizeMountPrefix(path));

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
 *
 * Throws `file_facts_invalid_order:<what>:<value>` when a registration `order` is not a non-negative integer; producers must
 * catch it per file at extraction time and treat it as an extract failure of that file (never let it abort a whole build).
 */
export function canonicalFileFacts(f: FileFacts): FileFacts {
    if (canonicalSet.has(f)) return f;
    for (const e of f.entryPoints) checkOrder(`entryPoint:${e.key}`, e.order);
    for (const u of f.uses ?? []) checkOrder(`use:${u.path}`, u.order);
    for (const m of f.mounts ?? []) checkOrder(`mount:${m.prefix}`, m.order);
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
        mounts: sortBy((f.mounts ?? []).map((m) => {
            const { guards, ...rest } = m;
            return { ...rest, prefix: normalizeMountPrefix(m.prefix), ...((guards?.length ?? 0) > 0 ? { guards: sorted(guards) } : {}) };
        }), (m) => ord(m.order), (m) => m.prefix, (m) => m.target, whole),
        uses: sortBy((f.uses ?? []).map((u) => {
            const { match, ...rest } = u;
            return { ...rest, path: normalizeUsePath(u.path), guards: sorted(u.guards), ...(match === 'exact' ? { match } : {}) };
        }), (u) => ord(u.order), (u) => u.path, whole),
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

/** The empty value of every kind's fields: what a rule that does not read that kind sees. */
const EMPTY_FIELDS: Readonly<Record<FactKind, Partial<FileFacts>>> = Object.freeze({
    structure: { symbols: [], imports: [], exports: [], entryPoints: [], mounts: [], uses: [] },
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
        mounts: raw.mounts ?? [],
        uses: raw.uses ?? [],
        calls: raw.calls ?? [],
        complexity: raw.complexity ?? [],
        capabilityImports: raw.capabilityImports ?? [],
        ...(raw.testFacts ? { testFacts: raw.testFacts } : {}),
    });
}

/** Whether a facts object is v3 (current shape). */
export const isFileFactsV3 = (f: Pick<FileFacts, 'factsVersion'>): boolean => f.factsVersion === FACTS_EXTRACTOR_VERSION;
