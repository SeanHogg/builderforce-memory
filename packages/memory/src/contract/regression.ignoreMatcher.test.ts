/**
 * src/contract/regression.ignoreMatcher.test.ts — the memoized `ignoreMatcher` answers exactly as the algorithm it replaced.
 *
 * Before: for every path the matcher built every ancestor folder and tested EVERY rule against every ancestor (and the
 * path): O(files × rules × depth) regex tests on the VS Code extension host's main thread, which a 34k-file multi-repo
 * tree (every nested `.gitignore` adds its rules) stalled for seconds. The reference below IS that algorithm, kept
 * verbatim, and is the oracle over a randomized tree with many nested `.gitignore` bases.
 */

import { ignoreMatcher } from '../knowledge/sources.js';
import { matchGlob } from '../knowledge/partitions.js';

const DEFAULT_IGNORES = ['.git/', 'node_modules/', '.builderforce/', '.gitignore', '*.lock'];

/** The matcher as it was before the memoization (sources.ts, 2026-11-14), verbatim. */
function referenceMatcher(texts: ReadonlyArray<string | { text: string; base: string }>): (path: string) => boolean {
    const rules: Array<{ glob: string; negate: boolean; dirOnly: boolean }> = [];
    const lines = [
        ...DEFAULT_IGNORES.map((line) => ({ line, base: '' })),
        ...texts.flatMap((t) => {
            const { text, base } = typeof t === 'string' ? { text: t, base: '' } : t;
            return text.split(/\r?\n/).map((line) => ({ line, base }));
        }),
    ];
    for (const { line, base } of lines) {
        let p = line.replace(/\s+$/, '');
        if (!p || p.startsWith('#')) continue;
        const negate = p.startsWith('!');
        if (negate) p = p.slice(1);
        const dirOnly = p.endsWith('/');
        if (dirOnly) p = p.slice(0, -1);
        const anchored = p.includes('/');
        p = p.replace(/^\//, '');
        const prefix = base ? `${base}/` : '';
        if (p) rules.push({ glob: anchored ? `${prefix}${p}` : `${prefix}**/${p}`, negate, dirOnly });
    }
    return (path) => {
        const segments = path.split('/');
        const dirs = segments.slice(0, -1).map((_, i) => segments.slice(0, i + 1).join('/'));
        let ignored = false;
        for (const r of rules) {
            const hit = (r.dirOnly ? dirs : [...dirs, path]).some((c) => matchGlob(c, r.glob));
            if (hit) ignored = !r.negate;
        }
        return ignored;
    };
}

/** A small deterministic PRNG (mulberry32), so a failure reproduces. */
function rng(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const SEGMENTS = ['src', 'lib', 'dist', 'build', 'out', 'node_modules', 'docs', 'test', 'tmp', 'cache', 'a', 'b', 'repo1', 'repo2', 'pkg', '.git', 'vendor', 'gen'];
const FILES = ['index.ts', 'a.md', 'b.js', 'yarn.lock', '.gitignore', 'x.log', 'notes.txt', 'keep.log', 'main.min.js', 'copy*.txt', 'data.json'];
const PATTERNS = [
    'dist/', 'build', '*.log', '!keep.log', '/out', 'tmp/', '*.min.js', 'cache/**', '!src/keep.log', 'docs/*.md', '!docs/',
    'gen/**/*.ts', '**/vendor', 'a/b', '!a', '*.json', '# a comment', '', 'test/', '!test/index.ts', 'copy\\*.txt', '{a,b}.md',
    'lib/?.js', '!yarn.lock', 'node_modules/', '/src/*.ts', 'pkg/**/b.js',
];

function randomTree(seed: number, opts: { paths: number; bases: number; rulesPerBase: number }) {
    const next = rng(seed);
    const pick = <T>(xs: readonly T[]): T => xs[Math.floor(next() * xs.length)]!;
    const dir = (depth: number) => Array.from({ length: depth }, () => pick(SEGMENTS)).join('/');
    const texts: Array<{ text: string; base: string }> = [];
    for (let i = 0; i < opts.bases; i++) {
        const base = i === 0 ? '' : dir(1 + Math.floor(next() * 4));
        const text = Array.from({ length: opts.rulesPerBase }, () => pick(PATTERNS)).join(i % 2 ? '\r\n' : '\n');
        texts.push({ text, base });
    }
    const paths = Array.from({ length: opts.paths }, () => {
        const depth = Math.floor(next() * 7);
        return depth === 0 ? pick(FILES) : `${dir(depth)}/${pick(FILES)}`;
    });
    return { texts, paths };
}

describe('regression: ignoreMatcher memoization', () => {
    it('regression: answers exactly as the unmemoized matcher over a randomized tree with many nested .gitignore bases', () => {
        for (const seed of [1, 2, 3, 42, 1337]) {
            const { texts, paths } = randomTree(seed, { paths: 2000, bases: 40, rulesPerBase: 8 });
            const fast = ignoreMatcher(texts);
            const ref = referenceMatcher(texts);
            const mismatches = paths.filter((p) => fast(p) !== ref(p));
            expect(mismatches).toEqual([]);
            // Both outcomes occur, so the comparison is not vacuous.
            expect(new Set(paths.map(ref)).size).toBe(2);
            // Asking again (served from the folder memo) changes nothing.
            expect(paths.filter((p) => fast(p) !== ref(p))).toEqual([]);
        }
        // Plain string texts (base ''), and degenerate paths.
        const plain = ['dist/\n!keep.log', '*.log'];
        const paths = ['dist/a.log', 'src/keep.log', 'a/dist/x.ts', 'x.log', 'dist', 'keep.log', '', '/abs/x.log', 'a//b.log'];
        expect(paths.map(ignoreMatcher(plain))).toEqual(paths.map(referenceMatcher(plain)));
    });

    it('perf guard: 30k paths × 300 rules finishes well within a few seconds', () => {
        const { texts, paths } = randomTree(7, { paths: 30_000, bases: 60, rulesPerBase: 5 }); // ≈300 rules with the defaults
        const fast = ignoreMatcher(texts);
        const start = performance.now();
        let ignored = 0;
        for (const p of paths) if (fast(p)) ignored++;
        const ms = performance.now() - start;
        expect(ignored).toBeGreaterThan(0);
        // Measured 2026-11-14 on this shape (folders nearly all distinct, the memo's worst case): the reference ≈4.6 s, this ≈0.2 s.
        expect(ms).toBeLessThan(2000);
    });
});
