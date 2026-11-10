/**
 * src/knowledge/facts/fileFacts.test.ts — FileFacts v3 (PRD 41 §B1, §B3): canonical order, per-kind hashes
 * (a body edit never moves the structure hash), v2 reads, and the composite extractor version.
 */

import { comparable } from '../comparability.js';
import { canonicalFileFacts, readFileFacts, isFileFactsV3, projectFileFacts, type FileFacts } from './fileFacts.js';
import { factKindHashes, factsHash } from './factsHash.js';
import { extractorStampMeta, factsExtractorTag, FACTS_EXTRACTOR_VERSION, isFactsCurrent, type ExtractorStamp } from './extractorVersion.js';

const STAMP: ExtractorStamp = { family: 'tree-sitter', grammars: { typescript: '0.23.2', python: '0.23.6' } };

const facts = (over: Partial<FileFacts> = {}): FileFacts => ({
    factsVersion: FACTS_EXTRACTOR_VERSION,
    extractor: factsExtractorTag(STAMP, 'typescript'),
    language: 'typescript',
    contentHash: 'c1',
    symbols: [{ name: 'b', kind: 'function' }, { name: 'a', kind: 'function' }],
    imports: [{ specifier: './z', resolved: 'z.ts', names: ['y', 'x'] }, { specifier: './a', resolved: 'a.ts', names: [] }],
    exports: [{ name: 'b', kind: 'function', signatureHash: 's2' }, { name: 'a', kind: 'function', signatureHash: 's1' }],
    entryPoints: [{ kind: 'http-route', key: 'POST /x', symbol: 'h', guards: ['requireAuth', 'audit'], inputs: ['req.body'], chain: ['mw.ts', 'app.ts'] }],
    calls: [{ callee: 'fetch', enclosing: 'b', awaited: true, inLoop: true, count: 1 }, { callee: 'db.select', enclosing: 'a', count: 2 }],
    complexity: [{ symbol: 'b', cyclomatic: 3 }, { symbol: 'a', cyclomatic: 1 }],
    capabilityImports: [{ capability: 'network-client', module: 'node:http' }, { capability: 'crypto', module: 'node:crypto', algorithm: 'md5' }],
    ...over,
});

describe('canonicalFileFacts', () => {
    it('sorts every array by its declared key, so producer order never reaches a hash', () => {
        const a = facts();
        const b = facts({ symbols: [...a.symbols].reverse(), imports: [...a.imports].reverse(), calls: [...a.calls].reverse() });
        expect(factsHash(a)).toBe(factsHash(b));
        const c = canonicalFileFacts(a);
        expect(c.symbols.map((s) => s.name)).toEqual(['a', 'b']);
        expect(c.imports[1]!.names).toEqual(['x', 'y']);
        expect(c.entryPoints[0]!.guards).toEqual(['audit', 'requireAuth']);
        expect(c.entryPoints[0]!.chain).toEqual(['app.ts', 'mw.ts']);
        expect(canonicalFileFacts(c)).toEqual(c);
    });
});

describe('canonicalFileFacts: paths, tie-breaks, immutability', () => {
    it('normalizes every path field (resolved imports, guard chains, snapshot files)', () => {
        const ep = facts().entryPoints[0]!;
        const c = canonicalFileFacts(facts({
            imports: [{ specifier: '..\\b', resolved: 'src\\b.ts', names: [] }],
            entryPoints: [{ ...ep, chain: ['.\\src\\mw.ts', 'src/app.ts'] }],
            testFacts: { tests: [], snapshotFiles: ['__snapshots__\\a.snap'] },
        }));
        expect(c.imports[0]!.resolved).toBe('src/b.ts');
        expect(c.entryPoints[0]!.chain).toEqual(['src/app.ts', 'src/mw.ts']);
        expect(c.testFacts!.snapshotFiles).toEqual(['__snapshots__/a.snap']);
    });

    it('items equal on the declared keys still sort one way (the whole item breaks the tie)', () => {
        const i1 = { specifier: './a', resolved: 'a.ts', names: [], typeOnly: true };
        const i2 = { specifier: './a', resolved: 'a.ts', names: [], dynamic: true };
        expect(factsHash(facts({ imports: [i1, i2] }))).toBe(factsHash(facts({ imports: [i2, i1] })));
        const t1 = { name: 't', assertions: 1, matchers: [{ matcher: 'toBe', strength: 'exact' as const, count: 1 }] };
        const t2 = { ...t1, assertions: 2 };
        expect(factsHash(facts({ testFacts: { tests: [t1, t2], snapshotFiles: [] } }))).toBe(factsHash(facts({ testFacts: { tests: [t2, t1], snapshotFiles: [] } })));
        expect(factsHash(facts({ complexity: [{ symbol: 'a', cyclomatic: 1 }, { symbol: 'a', cyclomatic: 2 }] })))
            .toBe(factsHash(facts({ complexity: [{ symbol: 'a', cyclomatic: 2 }, { symbol: 'a', cyclomatic: 1 }] })));
    });

    it('is deep-frozen, copies (never freezes) the input, and returns a canonical object as is', () => {
        const raw = facts();
        const c = canonicalFileFacts(raw);
        expect(Object.isFrozen(c) && Object.isFrozen(c.calls) && Object.isFrozen(c.calls[0])).toBe(true);
        expect(Object.isFrozen(raw.calls[0])).toBe(false);
        expect(canonicalFileFacts(c)).toBe(c);
        // A mutated non-canonical input is hashed again, never served a stale cached hash.
        const before = factKindHashes(raw).body;
        raw.calls = [];
        expect(factKindHashes(raw).body).not.toBe(before);
    });

    it('projectFileFacts keeps the header and the read kinds, empties the rest', () => {
        const p = projectFileFacts(facts(), ['structure']);
        expect(p.exports).toHaveLength(2);
        expect(p.calls).toEqual([]);
        expect(p.testFacts).toBeUndefined();
        expect(p.contentHash).toBe('');
        expect(p.extractor).toBe(facts().extractor);
        const c = canonicalFileFacts(facts());
        expect(projectFileFacts(c, ['structure'])).toBe(projectFileFacts(c, ['structure']));
    });
});

describe('factKindHashes', () => {
    it('a body-only change moves the body hash and nothing else', () => {
        const before = factKindHashes(facts());
        const after = factKindHashes(facts({ calls: [{ callee: 'fetch', enclosing: 'b', count: 1 }] }));
        expect(after.body).not.toBe(before.body);
        expect(after.structure).toBe(before.structure);
        expect(after.content).toBe(before.content);
    });

    it('a structure change (a guard removed from a route) moves only the structure hash', () => {
        const before = factKindHashes(facts());
        const ep = facts().entryPoints[0]!;
        const after = factKindHashes(facts({ entryPoints: [{ ...ep, guards: ['audit'] }] }));
        expect(after.structure).not.toBe(before.structure);
        expect(after.body).toBe(before.body);
    });

    it('content is the content hash; another extractor moves structure and body', () => {
        const h = factKindHashes(facts());
        expect(h.content).toBe('c1');
        const other = factKindHashes(facts({ extractor: factsExtractorTag({ family: 'regex' }, 'typescript') }));
        expect(other.structure).not.toBe(h.structure);
        expect(other.body).not.toBe(h.body);
    });
});

describe('readFileFacts (v2 compatibility)', () => {
    it('reads a v2 object as factsVersion 2 with every v3 list empty', () => {
        const v2 = readFileFacts({ contentHash: 'c', symbols: [{ name: 'f', kind: 'function' }] });
        expect(v2.factsVersion).toBe(2);
        expect(isFileFactsV3(v2)).toBe(false);
        expect(v2.exports).toEqual([]);
        expect(v2.calls).toEqual([]);
        expect(v2.entryPoints).toEqual([]);
        expect(v2.testFacts).toBeUndefined();
    });

    it('keeps a v3 object (canonicalized)', () => {
        const v3 = readFileFacts(facts());
        expect(isFileFactsV3(v3)).toBe(true);
        expect(factsHash(v3)).toBe(factsHash(facts()));
    });
});

describe('composite extractor version', () => {
    it('composes code version, revision, the language grammar and the EOL rule', () => {
        expect(factsExtractorTag(STAMP, 'typescript')).toBe('tree-sitter@3.0/typescript=0.23.2/eol-lf@1');
        expect(factsExtractorTag({ ...STAMP, revision: 2 }, 'go')).toBe('tree-sitter@3.2/eol-lf@1');
        expect(factsExtractorTag(STAMP)).toBe('tree-sitter@3.0/python=0.23.6,typescript=0.23.2/eol-lf@1');
    });

    it('a grammar bump owes re-extraction only for files of that language', () => {
        const ts = facts();
        const py = facts({ language: 'python', extractor: factsExtractorTag(STAMP, 'python') });
        const bumped: ExtractorStamp = { ...STAMP, grammars: { ...STAMP.grammars, python: '0.24.0' } };
        expect(isFactsCurrent(ts, bumped)).toBe(true);
        expect(isFactsCurrent(py, bumped)).toBe(false);
        expect(isFactsCurrent(readFileFacts({ ...ts, factsVersion: 2 }), STAMP)).toBe(false);
    });

    it('comparability names what moved: grammar, family, EOL rule', () => {
        const base = extractorStampMeta(STAMP);
        expect(comparable(base, extractorStampMeta(STAMP))).toEqual({ comparable: true, reasons: [] });
        expect(comparable(base, extractorStampMeta({ ...STAMP, grammars: { ...STAMP.grammars, python: '0.24.0' } })).reasons)
            .toEqual(['tool-version-changed:grammar:python']);
        expect(comparable(base, extractorStampMeta({ family: 'regex' })).reasons).toContain('extractor-mismatch');
        expect(comparable(base, extractorStampMeta({ ...STAMP, contentHashRule: 'eol-lf@2' })).reasons).toEqual(['content-hash-rule-changed']);
    });
});
