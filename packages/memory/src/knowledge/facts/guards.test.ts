/**
 * src/knowledge/facts/guards.test.ts — golden tests for the guard-chain semantics (PRD 41 §C5.1): a path-scoped `use`
 * guards only routes registered after it; router identity; a mounted sub-router inherits its mounting router's guards;
 * mount guards; unknown order; depth and cycles; path patterns; the additive `mounts` / `uses` fields.
 */

import { FACTS_EXTRACTOR_VERSION, factsExtractorTag } from './extractorVersion.js';
import { canonicalFileFacts, readFileFacts, type EntryPointFact, type FileFacts } from './fileFacts.js';
import { factKindHashes, factsHash } from './factsHash.js';
import { effectiveGuards, joinRoutePath, MAX_MOUNT_DEPTH, rootRouters, routePathCovers, type EffectiveGuardsResult, type ResolvedMount } from './guards.js';

const base = (over: Partial<FileFacts> = {}): FileFacts => ({
    factsVersion: FACTS_EXTRACTOR_VERSION, extractor: factsExtractorTag({ family: 'tree-sitter' }, 'typescript'), language: 'typescript',
    contentHash: 'c', symbols: [], imports: [], exports: [], entryPoints: [], calls: [], complexity: [], capabilityImports: [], ...over,
});

const route = (key: string, order: number | undefined, guards: string[] = [], router?: string): EntryPointFact =>
    ({ kind: 'http-route', key, symbol: `h${order ?? 'x'}`, guards, inputs: [], chain: [], ...(order !== undefined ? { order } : {}), ...(router ? { router } : {}) });

const guardsOf = (r: EffectiveGuardsResult) => Object.fromEntries(r.entryPoints.map((e) => [e.key, e.guards]));

describe('effectiveGuards: a use guards only routes registered after it', () => {
    it('golden: a path-scoped use registered AFTER a route does not guard it', () => {
        // app.get('/admin/early', h0); app.use('/admin', requireAuth); app.get('/admin/late', h2); app.get('/public', h3)
        const app = base({
            entryPoints: [route('GET /admin/early', 0), route('GET /admin/late', 2), route('GET /public', 3)],
            uses: [{ path: '/admin', guards: ['requireAuth'], order: 1 }],
        });
        expect(guardsOf(effectiveGuards(app, []))).toEqual({
            'GET /admin/early': [],
            'GET /admin/late': ['requireAuth'],
            'GET /public': [],
        });
    });

    it('a path-less use covers every later route; a route keeps its own inline guards', () => {
        const app = base({
            entryPoints: [route('POST /a', 0, ['csrf']), route('POST /b', 2, ['csrf'])],
            uses: [{ path: '', guards: ['requireAuth'], order: 1 }],
        });
        expect(guardsOf(effectiveGuards(app, []))).toEqual({ 'POST /a': ['csrf'], 'POST /b': ['csrf', 'requireAuth'] });
    });
});

describe('effectiveGuards: unknown order', () => {
    it('a route with no order takes every covering use, reported as inferred', () => {
        const app = base({ entryPoints: [route('GET /x', undefined, ['own'])], uses: [{ path: '/', guards: ['g', 'own'], order: 5 }] });
        const [ep] = effectiveGuards(app, []).entryPoints;
        expect(ep!.guards).toEqual(['g', 'own']);
        expect(ep!.guardsInferred).toEqual(['g']);
    });

    it('a route with a known order infers nothing', () => {
        const app = base({ entryPoints: [route('GET /x', 6)], uses: [{ path: '/', guards: ['g'], order: 5 }] });
        const [ep] = effectiveGuards(app, []).entryPoints;
        expect(ep!.guards).toEqual(['g']);
        expect(ep!.guardsInferred).toEqual([]);
    });
});

describe('effectiveGuards: router identity', () => {
    it('golden: two routers in one file — an adminRouter use never guards publicRouter routes', () => {
        const file = base({
            entryPoints: [route('GET /dash', 1, [], 'adminRouter'), route('GET /home', 2, [], 'publicRouter')],
            uses: [{ path: '', guards: ['requireSuperadmin'], order: 0, router: 'adminRouter' }],
        });
        expect(guardsOf(effectiveGuards(file, []))).toEqual({ 'GET /dash': ['requireSuperadmin'], 'GET /home': [] });
    });

    it('golden: a same-file local mount does not re-emit the parent router\'s own routes', () => {
        // app.use(requireAuth); app.get('/health', h1); adminRouter.get('/users', h2); app.route('/admin', adminRouter)
        const file = base({
            entryPoints: [route('GET /health', 1, [], 'app'), route('GET /users', 2, [], 'adminRouter')],
            uses: [{ path: '', guards: ['requireAuth'], order: 0, router: 'app' }],
            mounts: [{ prefix: '/admin', target: 'adminRouter', order: 3, router: 'app' }],
        });
        const r = effectiveGuards(file, [], { path: 'src/app.ts' });
        expect(r.entryPoints.map((e) => [e.key, e.guards, e.via])).toEqual([
            ['GET /admin/users', ['requireAuth'], ['/admin']],
            ['GET /health', ['requireAuth'], []],
        ]);
        expect(r.entryPoints.every((e) => e.file === 'src/app.ts' && !e.chain.includes('src/app.ts'))).toBe(true);
        expect([r.truncated, r.unresolved]).toEqual([0, 0]);
    });
});

describe('effectiveGuards: a mounted sub-router inherits the mounting app guards', () => {
    const sub = base({
        contentHash: 'sub',
        entryPoints: [route('GET /projects/:id', 1, ['RoleGate']), route('POST /admin/reset', 2)],
        uses: [{ path: '/admin/*', guards: ['requireSuperadmin'], order: 0 }],
    });

    it('golden: guards registered before the mount, covering the prefix, apply to every sub-router route (prefix-joined)', () => {
        // app.use('/api/*', requireAuth); app.route('/api', sub); app.use('/api/*', lateAudit)
        const app = base({
            uses: [{ path: '/api/*', guards: ['requireAuth'], order: 0 }, { path: '/api/*', guards: ['lateAudit'], order: 2 }],
            mounts: [{ prefix: '/api', target: './sub', order: 1 }],
        });
        const r = effectiveGuards(app, [{ target: './sub', path: 'src/sub.ts', facts: sub }], { path: 'src/app.ts' });
        expect(guardsOf(r)).toEqual({
            'GET /api/projects/:id': ['RoleGate', 'requireAuth'],
            'POST /api/admin/reset': ['requireAuth', 'requireSuperadmin'],
        });
        expect(r.entryPoints.every((e) => e.file === 'src/sub.ts' && e.via.join() === '/api' && e.chain.includes('src/app.ts'))).toBe(true);
    });

    it('golden: inline mount guards — router.use(\'/api\', requireAuth, apiRouter) — apply to every mounted route', () => {
        const app = base({ mounts: [{ prefix: '/api', target: './sub', order: 0, guards: ['requireAuth'] }] });
        expect(guardsOf(effectiveGuards(app, [{ target: './sub', path: 'src/sub.ts', facts: sub }]))).toEqual({
            'GET /api/projects/:id': ['RoleGate', 'requireAuth'],
            'POST /api/admin/reset': ['requireAuth', 'requireSuperadmin'],
        });
    });

    it('a mounting use scoped below the prefix guards only the sub-routes under it', () => {
        const app = base({
            uses: [{ path: '/api/admin', guards: ['requireSuperadmin2'], order: 0 }],
            mounts: [{ prefix: '/api', target: './sub', order: 1 }],
        });
        expect(guardsOf(effectiveGuards(app, [{ target: './sub', path: 'src/sub.ts', facts: sub }]))).toEqual({
            'GET /api/projects/:id': ['RoleGate'],
            'POST /api/admin/reset': ['requireSuperadmin', 'requireSuperadmin2'],
        });
    });

    it('nested mounts compose transitively; maxDepth truncates and says so', () => {
        const leaf = base({ contentHash: 'leaf', entryPoints: [route('GET /ping', 0)] });
        const mid = base({ contentHash: 'mid', uses: [{ path: '', guards: ['mid'], order: 0 }], mounts: [{ prefix: '/v1', target: './leaf', order: 1 }] });
        const app = base({ uses: [{ path: '', guards: ['top'], order: 0 }], mounts: [{ prefix: '/api', target: './mid', order: 1 }] });
        const resolved: ResolvedMount[] = [{ target: './mid', path: 'mid.ts', facts: mid, mounts: [{ target: './leaf', path: 'leaf.ts', facts: leaf }] }];
        const r = effectiveGuards(app, resolved, { path: 'app.ts' });
        expect(r.entryPoints.map((e) => [e.key, e.guards, e.via, e.chain])).toEqual([['GET /api/v1/ping', ['mid', 'top'], ['/api', '/v1'], ['app.ts', 'mid.ts']]]);
        expect(r.truncated).toBe(0);
        const shallow = effectiveGuards(app, resolved, { maxDepth: 1 });
        expect([shallow.entryPoints, shallow.truncated]).toEqual([[], 1]);
        expect(effectiveGuards(app, resolved, { maxDepth: Number.NaN }).entryPoints).toHaveLength(1);
    });

    it('a mount cycle is cut and counted, never looped', () => {
        const a = base({ contentHash: 'a', entryPoints: [route('GET /a', 0)], mounts: [{ prefix: '/b', target: './b', order: 1 }] });
        const b = base({ contentHash: 'b', entryPoints: [route('GET /b', 0)], mounts: [{ prefix: '/a', target: './a', order: 1 }] });
        const resolvedB: ResolvedMount = { target: './b', path: 'b.ts', facts: b };
        const resolvedA: ResolvedMount = { target: './a', path: 'a.ts', facts: a, mounts: [resolvedB] };
        (resolvedB as { mounts?: readonly ResolvedMount[] }).mounts = [resolvedA];
        const r = effectiveGuards(a, [resolvedB], { path: 'a.ts' });
        expect(r.entryPoints.map((e) => e.key)).toEqual(['GET /a', 'GET /b/b']);
        expect(r.truncated).toBe(1);
        expect(MAX_MOUNT_DEPTH).toBe(8);
    });

    it('an unresolved mount is counted; non-route entries of a sub-router are not mounted', () => {
        const withCmd = base({ entryPoints: [{ kind: 'vscode-command', key: 'bf.run', symbol: 'run', guards: [], inputs: [], chain: [] }] });
        const app = base({ mounts: [{ prefix: '/api', target: './x', order: 0 }] });
        expect(effectiveGuards(app, [])).toEqual({ entryPoints: [], truncated: 0, unresolved: 1 });
        expect(effectiveGuards(app, [{ target: './x', path: 'x.ts', facts: withCmd }]).entryPoints).toEqual([]);
    });
});

describe('rootRouters (per router)', () => {
    it('drops routers another file mounts, keeps local mounts within their file', () => {
        const files = new Map<string, FileFacts>([
            ['app.ts', base({ entryPoints: [route('GET /', 0)], mounts: [{ prefix: '/api', target: './sub', order: 1 }, { prefix: '/x', target: 'local', order: 2 }] })],
            ['sub.ts', base({ contentHash: 's', entryPoints: [route('GET /p', 0)] })],
            ['other.ts', base({ contentHash: 'o', entryPoints: [route('GET /o', 0, [], 'local')] })],
            ['lib.ts', base({ contentHash: 'l' })],
        ]);
        expect(rootRouters(files, (_from, m) => (m.target === './sub' ? { path: 'sub.ts' } : undefined)))
            .toEqual([{ path: 'app.ts', router: '' }, { path: 'other.ts', router: 'local' }]);
    });

    it('a mounted file\'s second, unmounted router is still a root; its locally mounted router is not', () => {
        const files = new Map<string, FileFacts>([
            ['app.ts', base({ entryPoints: [route('GET /', 0)], mounts: [{ prefix: '/api', target: './routers', order: 1 }] })],
            ['routers.ts', base({
                contentHash: 'r',
                entryPoints: [route('GET /a', 0, [], 'apiRouter'), route('GET /w', 1, [], 'webhookRouter'), route('GET /n', 2, [], 'nested')],
                mounts: [{ prefix: '/n', target: 'nested', order: 3, router: 'apiRouter' }],
            })],
        ]);
        const roots = rootRouters(files, (_from, m) => (m.target === './routers' ? { path: 'routers.ts', router: 'apiRouter' } : undefined));
        expect(roots).toEqual([{ path: 'app.ts', router: '' }, { path: 'routers.ts', router: 'webhookRouter' }]);
        const webhook = effectiveGuards(files.get('routers.ts')!, [], roots[1]!);
        expect(webhook.entryPoints.map((e) => e.key)).toEqual(['GET /w']);
    });

    it('files that mount each other: the lowest path becomes the root, deterministically', () => {
        const a = base({ contentHash: 'a', entryPoints: [route('GET /a', 0)], mounts: [{ prefix: '/b', target: './b', order: 1 }] });
        const b = base({ contentHash: 'b', entryPoints: [route('GET /b', 0)], mounts: [{ prefix: '/a', target: './a', order: 1 }] });
        const resolve = (_from: string, m: { target: string }) => ({ path: m.target === './a' ? 'a.ts' : 'b.ts' });
        expect(rootRouters(new Map([['b.ts', b], ['a.ts', a]]), resolve)).toEqual([{ path: 'a.ts', router: '' }]);
        expect(rootRouters(new Map([['a.ts', a], ['b.ts', b]]), resolve)).toEqual([{ path: 'a.ts', router: '' }]);
    });
});

describe('route path patterns', () => {
    it('prefix, exact, params and wildcards', () => {
        expect(routePathCovers('/api', '/api/x')).toBe(true);
        expect(routePathCovers('/api', '/api/x', 'exact')).toBe(false);
        expect(routePathCovers('/api', '/apix')).toBe(false);
        expect(routePathCovers('/api/*', '/api')).toBe(true);
        expect(routePathCovers('/p/:id/k', '/p/7/k/w')).toBe(true);
        expect(routePathCovers('*', '/anything')).toBe(true);
        expect(joinRoutePath('/api/', '/')).toBe('/api');
        expect(joinRoutePath('', '/x')).toBe('/x');
    });

    it('a trailing * on the last segment (/api*) is a prefix of that segment and the rest', () => {
        expect(routePathCovers('/api*', '/api')).toBe(true);
        expect(routePathCovers('/api*', '/apiv2/x')).toBe(true);
        expect(routePathCovers('/api*', '/ap')).toBe(false);
        expect(routePathCovers('/api*', '/other')).toBe(false);
    });

    it('an optional :x? matches one segment or none', () => {
        expect(routePathCovers('/p/:id?', '/p', 'exact')).toBe(true);
        expect(routePathCovers('/p/:id?', '/p/7', 'exact')).toBe(true);
        expect(routePathCovers('/p/:id?', '/p/7/8', 'exact')).toBe(false);
        expect(routePathCovers('/p/:id', '/p', 'exact')).toBe(false);
    });

    it('a Hono :id{regex} param matches one segment', () => {
        expect(routePathCovers('/p/:id{[0-9]+}', '/p/42', 'exact')).toBe(true);
        expect(routePathCovers('/p/:id{[0-9]+}/k', '/p/42/k')).toBe(true);
        expect(routePathCovers('/p/:id{[0-9]+}', '/p', 'exact')).toBe(false);
    });
});

describe('FileFacts mounts / uses: additive, non-breaking', () => {
    it('readFileFacts of an older object gives mounts: [] and uses: []; its hashes do not move', () => {
        const old = base();
        const read = readFileFacts(old);
        expect(read.mounts).toEqual([]);
        expect(read.uses).toEqual([]);
        expect(factsHash(read)).toBe(factsHash({ ...old, mounts: [], uses: [] }));
        expect(factKindHashes(read)).toEqual(factKindHashes({ ...old, mounts: [], uses: [] }));
    });

    it('mounts and uses are structure facts, sorted canonically', () => {
        const a = base({ mounts: [{ prefix: '/b', target: './b', order: 10 }, { prefix: '/a', target: './a', order: 2 }], uses: [{ path: '/', guards: ['y', 'x'], order: 1 }] });
        const c = canonicalFileFacts(a);
        expect(c.mounts!.map((m) => m.order)).toEqual([2, 10]);
        expect(c.uses![0]!.guards).toEqual(['x', 'y']);
        const reversed = base({ mounts: [...a.mounts!].reverse(), uses: a.uses });
        expect(factsHash(reversed)).toBe(factsHash(a));
        const before = factKindHashes(base());
        const after = factKindHashes(a);
        expect(after.structure).not.toBe(before.structure);
        expect(after.body).toBe(before.body);
    });

    it('canonical form: match "prefix" dropped, mount prefix normalized, empty mount guards and absent router omitted', () => {
        const c = canonicalFileFacts(base({
            mounts: [{ prefix: 'api/', target: './a', order: 0, guards: [] }],
            uses: [{ path: '/x', match: 'prefix', guards: [], order: 1 }, { path: '/y', match: 'exact', guards: [], order: 2 }],
        }));
        expect(c.mounts).toEqual([{ prefix: '/api', target: './a', order: 0 }]);
        expect(c.uses).toEqual([{ path: '/x', guards: [], order: 1 }, { path: '/y', match: 'exact', guards: [], order: 2 }]);
        expect(factsHash(base({ mounts: [{ prefix: '/api', target: './a', order: 0 }] })))
            .toBe(factsHash(base({ mounts: [{ prefix: '/api/', target: './a', order: 0, guards: [] }] })));
    });

    it('a use path is normalized: \'/admin/\' and \'/admin\' hash the same; \'\' stays the every-path form', () => {
        const use = (path: string) => base({ uses: [{ path, guards: ['g'], order: 0 }] });
        expect(factsHash(use('/admin/'))).toBe(factsHash(use('/admin')));
        expect(factsHash(use('admin//'))).toBe(factsHash(use('/admin')));
        expect(canonicalFileFacts(use('')).uses![0]!.path).toBe('');
        expect(canonicalFileFacts(use('/admin/*/')).uses![0]!.path).toBe('/admin/*');
    });

    it('an order that is not a non-negative integer is refused', () => {
        expect(() => canonicalFileFacts(base({ uses: [{ path: '', guards: [], order: -1 }] }))).toThrow(/file_facts_invalid_order:use/);
        expect(() => canonicalFileFacts(base({ mounts: [{ prefix: '/', target: 'x', order: 1.5 }] }))).toThrow(/file_facts_invalid_order:mount/);
        expect(() => canonicalFileFacts(base({ entryPoints: [route('GET /x', Number.NaN)] }))).toThrow(/file_facts_invalid_order:entryPoint/);
    });
});
