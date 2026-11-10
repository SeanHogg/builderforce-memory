/**
 * knowledge/facts/guards.ts — the guard-chain semantics of entry points across `use()` registrations and router mounts
 * (PRD 41 §C5.1). Pure: the host resolves each imported mount's target to a file and its facts; this module composes.
 *
 * The rules, explicit:
 *   1. ROUTER IDENTITY. Routes, uses and mounts belong to the router binding that registers them (`router`; absent = the
 *      file's one default router). A `use` on `adminRouter` never guards a route on `publicRouter`, even in one file.
 *   2. ORDER. A `use(path?, guard)` ({@link UseFact}) guards a route of its router only when it was registered BEFORE the
 *      route (`order`) and its path pattern covers the route's path. A path-scoped `use` registered after a route never
 *      guards it. A route with no `order` takes every covering `use`, and those guards are reported in `guardsInferred`
 *      (a C5.1 rule downgrades them to heuristic). A producer that emits `uses` MUST set `order` on every entry point.
 *   3. MOUNTS. A mounted sub-router (`app.route('/api', sub)`, `app.use('/api', subRouter)`, {@link MountFact}) answers its
 *      routes under the prefix-joined path, and inherits every `use` of the mounting router registered before the mount
 *      that covers the joined path, plus the mount's own inline `guards` — transitively, up to {@link MAX_MOUNT_DEPTH}
 *      levels. A target naming a local router binding of the same file is resolved here; other targets come from the host
 *      ({@link ResolvedMount}). Cycles are cut by the (file, router) pairs on the stack.
 *   4. A route's own `guards` (inline middleware, wrappers) always apply.
 *
 * CALL SITES. `effectiveGuards` is called on ROOT routers only: files no other file mounts ({@link rootRouters}), and
 * within a file the routers no local mount targets. A sub-router's routes are reported through their root (prefix-joined,
 * with the inherited guards); calling it on a mounted file as well would report a bare, under-guarded copy of each route.
 * Consumers that read `entryPoints[].guards` directly must switch to this function in the same change in which a producer
 * starts emitting `uses`.
 *
 * Route paths come from the entry key: `'<METHOD> /path'` or `'/path'`. An entry point without a path (a VS Code command,
 * an exported API) gets only path-less (`''`, `'*'`, `'/*'`) uses of its router, and is never mounted.
 */

import { compareStr, sortBy } from '../refs.js';
import { canonicalFileFacts, type EntryPointFact, type FileFacts, type MountFact, type UseFact } from './fileFacts.js';

/** How many mount levels `effectiveGuards` descends at most (mount nesting, not import hops). */
export const MAX_MOUNT_DEPTH = 8;

/** A mount the host resolved: the target (as written in {@link MountFact.target}) → the sub-router's file and facts. */
export interface ResolvedMount {
    /** The {@link MountFact.target} this resolves (a file may mount one target at several prefixes). */
    target: string;
    /** The sub-router's root-relative file. */
    path: string;
    facts: FileFacts;
    /** The router binding in that file that is mounted. Absent ⇒ that file's root routers. */
    router?: string;
    /** The sub-router file's own imported mounts, resolved the same way (nested routers). */
    mounts?: readonly ResolvedMount[];
}

/** An entry point with its effective guards: prefix-joined key, composed guards, the mounting files on its chain. */
export interface EffectiveEntryPoint extends EntryPointFact {
    /** The file that registers the handler: `opts.path` (default '') for the given file, the sub-router's path otherwise. */
    file: string;
    /** The mount prefixes from the root router down to the registering router, outermost first; empty for root routes. */
    via: string[];
    /** Guards in `guards` that apply only because the route has no `order` (rule 2), sorted. Empty when the order is known. */
    guardsInferred: string[];
}

export interface EffectiveGuardsOptions {
    /** The given file's root-relative path: reported as `file`, and added to a mounted route's `chain`. */
    path?: string;
    /** The root router binding to start from ('' = the file's default router). Absent ⇒ every router of the file that no local mount targets. */
    router?: string;
    /** Mount levels to descend (0..{@link MAX_MOUNT_DEPTH}). Default {@link MAX_MOUNT_DEPTH}. */
    maxDepth?: number;
}

export interface EffectiveGuardsResult {
    /** Sorted by file, kind, key, symbol, via. */
    entryPoints: EffectiveEntryPoint[];
    /** Mounts not followed because of the depth limit or a cycle. */
    truncated: number;
    /** Mounts whose target was neither a local router nor in `resolvedMounts`. */
    unresolved: number;
}

const WILDCARD_PATHS = new Set(['', '*', '/*']);
const segments = (path: string) => path.split('/').filter((s) => s !== '');

/** Joins a mount prefix and a route path into one `/`-rooted path ('/api' + '/x' → '/api/x'; '/api' + '/' → '/api'). */
export function joinRoutePath(prefix: string, path: string): string {
    return `/${[...segments(prefix), ...segments(path)].join('/')}`;
}

const isParam = (seg: string) => seg.startsWith(':');

/**
 * Whether a `use` path pattern covers `path` (rule 2's path half). Segments: a literal matches itself; `:x`, `:x{regex}`
 * match one segment; `:x?` matches one segment or none; `*` as the last segment matches the rest (zero or more segments);
 * a last segment ending in `*` (`api*`) matches any segment starting with its stem, and the rest. `prefix` (default)
 * covers every path under the pattern; `exact` that path only.
 */
export function routePathCovers(pattern: string, path: string, match: UseFact['match'] = 'prefix'): boolean {
    if (WILDCARD_PATHS.has(pattern)) return true;
    const pat = segments(pattern);
    const segs = segments(path);
    const last = pat.length - 1;
    const at = (pi: number, si: number): boolean => {
        if (pi === pat.length) return match === 'exact' ? si === segs.length : true;
        const p = pat[pi]!;
        if (pi === last && p === '*') return true;
        if (pi === last && p.endsWith('*') && !isParam(p)) return si < segs.length && segs[si]!.startsWith(p.slice(0, -1));
        if (isParam(p)) {
            if (si < segs.length && at(pi + 1, si + 1)) return true;
            return p.endsWith('?') && at(pi + 1, si);
        }
        if (p === '*') return si < segs.length && at(pi + 1, si + 1);
        return si < segs.length && segs[si] === p && at(pi + 1, si + 1);
    };
    return at(0, 0);
}

/** `'<METHOD> /path'` → method + path; `'/path'` → path; anything else → null (not a routed entry point). */
function parseRouteKey(key: string): { method: string; path: string } | null {
    const m = /^(?:(\S+)\s+)?(\/\S*)$/.exec(key.trim());
    return m ? { method: m[1] ?? '', path: m[2]! } : null;
}

/** A use lifted to absolute paths: its pattern joined with the prefix of the router that registered it. */
interface ScopedUse { pattern: string; match: UseFact['match']; guards: readonly string[]; wildcard: boolean }

const everything = (prefix: string, guards: readonly string[], wildcard: boolean): ScopedUse =>
    ({ pattern: `${joinRoutePath(prefix, '')}/*`, match: 'prefix', guards, wildcard });

const scoped = (u: UseFact, prefix: string): ScopedUse => (WILDCARD_PATHS.has(u.path)
    ? everything(prefix, u.guards, true)
    : { pattern: joinRoutePath(prefix, u.path), match: u.match, guards: u.guards, wildcard: false });

/** The router key of a registration ('' = the file's default router). */
const routerOf = (x: { router?: string }) => x.router ?? '';

/** Every router binding that registers something in `f`. */
function routersOf(f: FileFacts): Set<string> {
    return new Set([...f.entryPoints.map(routerOf), ...(f.uses ?? []).map(routerOf), ...(f.mounts ?? []).map(routerOf)]);
}

/** The routers of `f` no local mount targets (rule 3), sorted. */
function rootRoutersOf(f: FileFacts): string[] {
    const routers = routersOf(f);
    for (const m of f.mounts ?? []) if (routers.has(m.target) && m.target !== routerOf(m)) routers.delete(m.target);
    return [...routers].sort(compareStr);
}

/**
 * Every entry point of the root router(s) of `fileFacts` and of the sub-routers they mount, with its EFFECTIVE guards
 * (rules 1–4 of the header). Mounted routes get prefix-joined keys and the mounting files added to their `chain`.
 */
export function effectiveGuards(fileFacts: FileFacts, resolvedMounts: readonly ResolvedMount[], opts: EffectiveGuardsOptions = {}): EffectiveGuardsResult {
    const requested = opts.maxDepth ?? MAX_MOUNT_DEPTH;
    const maxDepth = Number.isFinite(requested) ? Math.max(0, Math.min(MAX_MOUNT_DEPTH, Math.floor(requested))) : MAX_MOUNT_DEPTH;
    const out: EffectiveEntryPoint[] = [];
    let truncated = 0;
    let unresolved = 0;

    interface Frame {
        facts: FileFacts; file: string; router: string; prefix: string; via: readonly string[];
        inherited: readonly ScopedUse[]; chainFiles: readonly string[]; resolved: readonly ResolvedMount[]; depth: number; stack: ReadonlySet<string>;
    }

    const visit = (fr: Frame) => {
        const f = fr.facts;
        const uses = (f.uses ?? []).filter((u) => routerOf(u) === fr.router);
        for (const ep of f.entryPoints) {
            if (routerOf(ep) !== fr.router) continue;
            const route = parseRouteKey(ep.key);
            if (fr.depth > 0 && !route) continue; // only routes answer under a mount
            const absPath = route ? joinRoutePath(fr.prefix, route.path) : null;
            const applies = (u: ScopedUse) => (absPath === null ? u.wildcard : routePathCovers(u.pattern, absPath, u.match));
            const known = new Set(ep.guards);
            for (const u of fr.inherited) if (applies(u)) for (const g of u.guards) known.add(g);
            const inferred = new Set<string>();
            for (const u of uses) {
                if (ep.order !== undefined && u.order >= ep.order) continue;
                if (!applies(scoped(u, fr.prefix))) continue;
                for (const g of u.guards) (ep.order === undefined && !known.has(g) ? inferred : known).add(g);
            }
            const key = route && fr.depth > 0 ? (route.method ? `${route.method} ${absPath}` : absPath!) : ep.key;
            out.push({
                ...ep,
                key,
                guards: [...new Set([...known, ...inferred])].sort(compareStr),
                guardsInferred: [...inferred].sort(compareStr),
                chain: [...new Set([...ep.chain, ...fr.chainFiles.filter((c) => c !== fr.file)])].sort(compareStr),
                file: fr.file,
                via: [...fr.via],
            });
        }
        const local = routersOf(f);
        for (const m of f.mounts ?? []) {
            if (routerOf(m) !== fr.router) continue;
            const prefix = joinRoutePath(fr.prefix, m.prefix);
            const inherited = [
                ...fr.inherited,
                ...uses.filter((u) => u.order < m.order).map((u) => scoped(u, fr.prefix)),
                ...((m.guards?.length ?? 0) > 0 ? [everything(prefix, m.guards!, false)] : []),
            ];
            let targets = fr.resolved.filter((r) => r.target === m.target);
            if (targets.length === 0 && local.has(m.target)) {
                targets = [{ target: m.target, path: fr.file, facts: f, router: m.target, mounts: fr.resolved }];
            }
            if (targets.length === 0) { unresolved++; continue; }
            if (fr.depth >= maxDepth) { truncated += targets.length; continue; }
            const chainFiles = fr.file === '' || fr.chainFiles.includes(fr.file) ? fr.chainFiles : [...fr.chainFiles, fr.file];
            for (const r of targets) {
                const sub = canonicalFileFacts(r.facts);
                for (const router of r.router !== undefined ? [r.router] : rootRoutersOf(sub)) {
                    const id = `${r.path}\u0000${router}`;
                    if (fr.stack.has(id)) { truncated++; continue; }
                    visit({
                        facts: sub, file: r.path, router, prefix, via: [...fr.via, m.prefix], inherited, chainFiles,
                        resolved: r.mounts ?? [], depth: fr.depth + 1, stack: new Set([...fr.stack, id]),
                    });
                }
            }
        }
    };

    const root = canonicalFileFacts(fileFacts);
    const file = opts.path ?? '';
    for (const router of opts.router !== undefined ? [opts.router] : rootRoutersOf(root)) {
        visit({
            facts: root, file, router, prefix: '', via: [], inherited: [], chainFiles: [], resolved: resolvedMounts, depth: 0,
            stack: new Set([`${file}\u0000${router}`]),
        });
    }
    return {
        entryPoints: sortBy(out, (e) => e.file, (e) => e.kind, (e) => e.key, (e) => e.symbol, (e) => e.via.join('\u0000')),
        truncated,
        unresolved,
    };
}

/** One root router: a file and the router binding in it ('' = the file's default router). */
export interface RootRouter {
    path: string;
    router: string;
}

/** Where an imported mount lands: the mounted file, and the router binding in it (absent ⇒ that file's root routers). */
export interface MountTarget {
    path: string;
    router?: string;
}

/**
 * The ROOT routers of a set of files, PER ROUTER: every (file, router) that registers something and that no other router's
 * mount reaches — so a mounted file's second, unmounted router is still a root. Routers mounted only inside a cycle (files
 * mounting each other) are not dropped: the lowest unreached (path, router) becomes a root, then everything it reaches is
 * covered, repeated until every router is covered (deterministic). `resolveMount` maps a mount of the file at `from` to
 * its target (undefined when unresolved); local-binding mounts stay within their file. Sorted by path, then router. Call
 * `effectiveGuards(facts, mounts, { path, router })` on each (see CALL SITES in the header).
 */
export function rootRouters(files: ReadonlyMap<string, FileFacts>, resolveMount: (from: string, mount: MountFact) => MountTarget | undefined): RootRouter[] {
    const id = (path: string, router: string) => `${path}\u0000${router}`;
    const nodes = new Map<string, RootRouter>();
    const canon = new Map<string, FileFacts>();
    for (const [path, raw] of files) {
        const f = canonicalFileFacts(raw);
        canon.set(path, f);
        for (const router of routersOf(f)) nodes.set(id(path, router), { path, router });
    }
    const edges = new Map<string, Set<string>>();
    const incoming = new Set<string>();
    const link = (from: string, to: string) => {
        if (to === from || !nodes.has(to)) return;
        let s = edges.get(from);
        if (!s) edges.set(from, (s = new Set()));
        s.add(to);
        incoming.add(to);
    };
    for (const [path, f] of canon) {
        const local = routersOf(f);
        for (const m of f.mounts ?? []) {
            const from = id(path, routerOf(m));
            if (local.has(m.target)) { link(from, id(path, m.target)); continue; }
            const target = resolveMount(path, m);
            const sub = target ? canon.get(target.path) : undefined;
            if (!target || !sub) continue;
            for (const router of target.router !== undefined ? [target.router] : rootRoutersOf(sub)) link(from, id(target.path, router));
        }
    }
    const order = [...nodes.keys()].sort(compareStr);
    const covered = new Set<string>();
    const cover = (start: string) => {
        const stack = [start];
        while (stack.length > 0) {
            const n = stack.pop()!;
            if (covered.has(n)) continue;
            covered.add(n);
            for (const next of edges.get(n) ?? []) stack.push(next);
        }
    };
    const roots: string[] = order.filter((n) => !incoming.has(n));
    for (const r of roots) cover(r);
    for (const n of order) if (!covered.has(n)) { roots.push(n); cover(n); }
    return roots.sort(compareStr).map((n) => nodes.get(n)!);
}
