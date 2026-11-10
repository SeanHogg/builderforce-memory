/**
 * src/knowledge/facts/factsHash.test.ts — the pinned content address (PRD 41 §B4, §B8). The literals below were computed by
 * the FileFacts code as released in 2026.10.72 (before `mounts` / `uses` / `router` / `order` existed). A facts object with
 * none of the added fields must keep exactly this address and these kind hashes, or every stored object would be orphaned.
 */

import type { FileFacts } from './fileFacts.js';
import { readFileFacts } from './fileFacts.js';
import { factKindHashes, factsHash, factsJson } from './factsHash.js';

const GOLDEN: FileFacts = {
    factsVersion: 3,
    extractor: 'tree-sitter@3.0/typescript=0.23.2/eol-lf@1',
    language: 'typescript',
    contentHash: 'golden-content',
    symbols: [{ name: 'b', kind: 'function', signatureHash: 'sb' }, { name: 'a', kind: 'class' }],
    imports: [{ specifier: './x', resolved: 'src/x.ts', names: ['y', 'x'], typeOnly: true }],
    exports: [{ name: 'a', kind: 'class', signatureHash: 'sa' }],
    entryPoints: [{ kind: 'http-route', key: 'POST /api/x', symbol: 'a.post', guards: ['requireAuth'], inputs: ['req.body'], chain: ['src/mw.ts'] }],
    calls: [{ callee: 'fetch', enclosing: 'b', awaited: true, inLoop: true, count: 2 }],
    complexity: [{ symbol: 'b', cyclomatic: 4 }],
    capabilityImports: [{ capability: 'crypto', module: 'node:crypto', algorithm: 'md5' }],
    testFacts: { framework: 'jest', tests: [{ name: 't', assertions: 1, matchers: [{ matcher: 'toBe', strength: 'exact', count: 1 }] }], snapshotFiles: [] },
};

const GOLDEN_FACTS_HASH = 'ac6cc429237854783c0d22fcae60db7163ffe27685b802ef9537bacb31337e27';
const GOLDEN_KINDS = {
    body: '913d8f716d4f2938942b93d42cbdcea8a9b4d653531f35f132a5ab4613fa433b',
    content: 'golden-content',
    structure: '9f2297f03ad517e864cfb9dfbaeba3df522fea94502e4ecee9dda4c16a1d9a33',
};

describe('factsHash: pinned golden address', () => {
    it('a v3 object without the added fields keeps its released address and kind hashes', () => {
        expect(factsHash(GOLDEN)).toBe(GOLDEN_FACTS_HASH);
        expect({ ...factKindHashes(GOLDEN) }).toEqual(GOLDEN_KINDS);
    });

    it('read back through readFileFacts (mounts: [], uses: []) the address does not move', () => {
        const read = readFileFacts(JSON.parse(factsJson(GOLDEN)) as FileFacts);
        expect(read.mounts).toEqual([]);
        expect(factsHash(read)).toBe(GOLDEN_FACTS_HASH);
        expect({ ...factKindHashes(read) }).toEqual(GOLDEN_KINDS);
    });

    it('the wire form omits the empty added lists', () => {
        expect(factsJson(GOLDEN)).not.toContain('"mounts"');
        expect(factsJson(GOLDEN)).not.toContain('"uses"');
    });
});
