/**
 * src/knowledge/merkle.test.ts — directory Merkle hashes (PRD 41 §B2 step 4).
 */

import { jest } from '@jest/globals';
import { canonicalListing, changedPaths, merkleTree, type MerkleTree } from './merkle.js';

const FILES = [
    { path: 'README.md', hash: 'h-readme' },
    { path: 'src/a.ts', hash: 'h-a' },
    { path: 'src/lib/b.ts', hash: 'h-b' },
    { path: 'src/lib/c.ts', hash: 'h-c' },
    { path: 'docs/guide/intro.md', hash: 'h-intro' },
    { path: 'docs/guide/deep/more.md', hash: 'h-more' },
];

/** A copy of `files` with `path` at a new hash. */
const edited = (files: typeof FILES, path: string, hash: string) => files.map((f) => (f.path === path ? { path, hash } : f));

describe('merkleTree', () => {
    it('gives the same root and dirs for any input order', () => {
        const a = merkleTree(FILES);
        const b = merkleTree([...FILES].reverse());
        expect(b.root).toBe(a.root);
        expect([...b.dirs]).toEqual([...a.dirs]);
        expect(a.dirs.get('')).toBe(a.root);
        expect([...a.dirs.keys()].sort()).toEqual(['', 'docs', 'docs/guide', 'docs/guide/deep', 'src', 'src/lib']);
    });

    it('normalizes paths: backslashes, a leading ./, NFC', () => {
        const a = merkleTree([{ path: 'src/café.ts', hash: 'h' }]);
        expect(merkleTree([{ path: '.\\src\\café.ts', hash: 'h' }]).root).toBe(a.root);
    });

    it('sorts entries bytewise (code-point order), not by UTF-16 unit', () => {
        const tree = merkleTree([{ path: '\u{1F600}.md', hash: 'h1' }, { path: '～.md', hash: 'h2' }, { path: 'Z.md', hash: 'h3' }]);
        expect(tree.entries.get('')!.map((e) => e.name)).toEqual(['Z.md', '～.md', '\u{1F600}.md']);
    });

    it('a file change changes only its ancestors', () => {
        const before = merkleTree(FILES);
        const after = merkleTree(edited(FILES, 'src/lib/b.ts', 'h-b2'));
        expect(after.root).not.toBe(before.root);
        for (const dir of before.dirs.keys()) {
            const ancestor = dir === '' || dir === 'src' || dir === 'src/lib';
            expect({ dir, changed: after.dirs.get(dir) !== before.dirs.get(dir) }).toEqual({ dir, changed: ancestor });
        }
    });

    it('a file and a directory of one name differ in kind, so they never hash alike', () => {
        expect(merkleTree([{ path: 'x', hash: 'h' }]).root).not.toBe(merkleTree([{ path: 'x/y', hash: 'h' }]).root);
    });

    it('refuses a path listed twice at different hashes, or as both a file and a directory', () => {
        expect(() => merkleTree([{ path: 'a', hash: '1' }, { path: 'a', hash: '2' }])).toThrow(/listed twice/);
        expect(() => merkleTree([{ path: 'a', hash: '1' }, { path: 'a/b', hash: '2' }])).toThrow(/file and a directory/);
        expect(() => merkleTree([{ path: 'a/b', hash: '2' }, { path: 'a', hash: '1' }])).toThrow(/file and a directory/);
        expect(() => merkleTree([{ path: 'a//b', hash: '1' }])).toThrow(/not a file path/);
        expect(() => merkleTree([{ path: 'a/./b', hash: '1' }])).toThrow(/not a file path/);
        expect(() => merkleTree([{ path: 'a/../b', hash: '1' }])).toThrow(/not a file path/);
        expect(() => merkleTree([{ path: '..', hash: '1' }])).toThrow(/not a file path/);
        expect(merkleTree([{ path: 'a', hash: '1' }, { path: 'a', hash: '1' }]).root).toBe(merkleTree([{ path: 'a', hash: '1' }]).root);
    });

    it('an empty listing has a root', () => {
        expect(merkleTree([]).root).toMatch(/^[0-9a-f]{64}$/);
    });
});

describe('canonicalListing', () => {
    const NFC = 'café.md', NFD = 'café.md';

    it('keeps the first of a colliding pair by (normalized, listed) bytewise, in any input order, and never makes merkleTree throw', () => {
        const pair = [{ path: NFC, hash: 'h1' }, { path: NFD, hash: 'h2' }];
        for (const input of [pair, [...pair].reverse()]) {
            const { kept, skipped } = canonicalListing(input);
            // NFD's 'e' (U+0065) sorts before NFC's 'é' (U+00E9): the NFD spelling is kept, under its NFC path.
            expect(kept).toEqual([{ path: NFC, hash: 'h2', listedPath: NFD }]);
            expect(skipped).toEqual([{ path: NFC, reason: 'duplicate-path' }]);
            expect(() => merkleTree(kept)).not.toThrow();
        }
    });

    it('skips invalid paths, a path listed twice, and a path below a kept file', () => {
        const { kept, skipped } = canonicalListing([
            { path: 'a', hash: '1' }, { path: 'a/b', hash: '2' }, { path: 'x.md', hash: '3' }, { path: './x.md', hash: '4' },
            { path: '../up.md', hash: '5' }, { path: 'p/./q', hash: '6' }, { path: 'p//q', hash: '7' }, { path: '', hash: '8' }, { path: 'src\\k.ts', hash: '9' },
        ]);
        expect(kept).toEqual([{ path: 'a', hash: '1' }, { path: 'src/k.ts', hash: '9', listedPath: 'src\\k.ts' }, { path: 'x.md', hash: '4', listedPath: './x.md' }]);
        expect(skipped).toEqual([
            { path: '../up.md', reason: 'invalid-path' }, { path: 'p/./q', reason: 'invalid-path' }, { path: 'p//q', reason: 'invalid-path' },
            { path: '', reason: 'invalid-path' }, { path: 'a/b', reason: 'duplicate-path' }, { path: 'x.md', reason: 'duplicate-path' },
        ]);
    });
});

/** `tree` with its `entries.get` spied, so a test can see which directories a walk opened. */
function spied(tree: MerkleTree) {
    const entries = new Map(tree.entries);
    const get = jest.spyOn(entries, 'get');
    return { tree: { ...tree, entries } as MerkleTree, opened: () => new Set(get.mock.calls.map((c) => c[0])) };
}

describe('changedPaths', () => {
    it('returns exactly the changed, added and removed files', () => {
        const next = [
            ...edited(FILES, 'src/lib/b.ts', 'h-b2').filter((f) => f.path !== 'docs/guide/deep/more.md'),
            { path: 'src/lib/new.ts', hash: 'h-new' },
        ];
        expect(changedPaths(merkleTree(FILES), merkleTree(next))).toEqual(['docs/guide/deep/more.md', 'src/lib/b.ts', 'src/lib/new.ts']);
        expect(changedPaths(merkleTree(next), merkleTree(FILES))).toEqual(['docs/guide/deep/more.md', 'src/lib/b.ts', 'src/lib/new.ts']);
    });

    it('is empty for equal trees and opens no directory', () => {
        const after = spied(merkleTree(FILES));
        expect(changedPaths(merkleTree(FILES), after.tree)).toEqual([]);
        expect(after.opened().size).toBe(0);
    });

    it('never descends into a subtree whose hash is equal', () => {
        const before = spied(merkleTree(FILES));
        const after = spied(merkleTree(edited(FILES, 'src/lib/b.ts', 'h-b2')));
        expect(changedPaths(before.tree, after.tree)).toEqual(['src/lib/b.ts']);
        expect([...after.opened()].sort()).toEqual(['', 'src', 'src/lib']);
        expect([...before.opened()].sort()).toEqual(['', 'src', 'src/lib']);
    });

    it('a file replaced by a directory reports both sides', () => {
        const before = merkleTree([{ path: 'x', hash: 'h' }, { path: 'keep.md', hash: 'k' }]);
        const after = merkleTree([{ path: 'x/y', hash: 'h' }, { path: 'x/z/w', hash: 'h2' }, { path: 'keep.md', hash: 'k' }]);
        expect(changedPaths(before, after)).toEqual(['x', 'x/y', 'x/z/w']);
    });

    it('from an empty tree, every file is added', () => {
        expect(changedPaths(merkleTree([]), merkleTree(FILES))).toEqual(FILES.map((f) => f.path).sort());
    });
});
