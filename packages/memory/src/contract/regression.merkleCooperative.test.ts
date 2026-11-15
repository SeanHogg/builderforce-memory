/**
 * src/contract/regression.merkleCooperative.test.ts — merkleTree, canonicalListing and changedPaths became async and
 * cooperative (they yield to the event loop on a work budget). A yield must never change a result: a run that pauses at
 * every check equals a run that never pauses. The root of a small fixed listing is pinned as a literal, so any future
 * change to hashing or ordering fails loudly here.
 */

import type { CooperativeBudget } from '../knowledge/cooperative.js';
import { canonicalListing, changedPaths, merkleTree } from '../knowledge/merkle.js';

/** Always due: the loops pause at every check. */
const forced = (): CooperativeBudget & { pauses: number } => {
    const b = { pauses: 0, due: () => true, pause: async () => { b.pauses++; await Promise.resolve(); } };
    return b;
};
/** Never due: the loops run straight through. */
const unforced = (): CooperativeBudget => ({ due: () => false, pause: async () => { throw new Error('unforced budget paused'); } });

const hex = (n: number) => n.toString(16).padStart(64, '0');

/** ~8k paths across nested directories, listed out of order, with re-spelled, invalid and duplicate entries mixed in. */
const LARGE: Array<{ path: string; hash: string }> = (() => {
    const out: Array<{ path: string; hash: string }> = [];
    for (let i = 0; i < 8000; i++) out.push({ path: `d${i % 13}/s${i % 7}/t${i % 5}/f${i}.ts`, hash: hex(i) });
    out.reverse();
    out.push({ path: './d0/dup.ts', hash: hex(1) }, { path: 'd0/dup.ts', hash: hex(2) }, { path: '../up.ts', hash: hex(3) }, { path: 'd1/./x.ts', hash: hex(4) });
    return out;
})();

/** LARGE with some files changed, one removed and one added. */
const LARGE_AFTER = LARGE.filter((f) => f.path !== 'd3/s3/t3/f3.ts')
    .map((f) => (f.path.endsWith('7.ts') ? { ...f, hash: hex(1_000_000 + Number.parseInt(f.hash, 16)) } : f))
    .concat({ path: 'new/added.ts', hash: hex(42) });

describe('regression: cooperative merkle yields never change a result', () => {
    it('regression: a budget that is always due pauses, and canonicalListing / merkleTree / changedPaths equal the unforced run', async () => {
        const budget = forced();

        const listForced = await canonicalListing(LARGE, { budget });
        const listPlain = await canonicalListing(LARGE, { budget: unforced() });
        expect(listForced).toEqual(listPlain);
        expect(listPlain.skipped.length).toBeGreaterThan(0);

        const afterForced = await canonicalListing(LARGE_AFTER, { budget });
        const afterPlain = await canonicalListing(LARGE_AFTER, { budget: unforced() });
        expect(afterForced).toEqual(afterPlain);

        const beforeTreeForced = await merkleTree(listForced.kept, { budget });
        const beforeTreePlain = await merkleTree(listPlain.kept, { budget: unforced() });
        const afterTreeForced = await merkleTree(afterForced.kept, { budget });
        const afterTreePlain = await merkleTree(afterPlain.kept, { budget: unforced() });
        expect(beforeTreeForced.root).toBe(beforeTreePlain.root);
        expect(afterTreeForced.root).toBe(afterTreePlain.root);
        expect(beforeTreeForced.dirs).toEqual(beforeTreePlain.dirs);
        expect(beforeTreeForced.entries).toEqual(beforeTreePlain.entries);

        const changedForced = await changedPaths(beforeTreeForced, afterTreeForced, { budget });
        const changedPlain = await changedPaths(beforeTreePlain, afterTreePlain, { budget: unforced() });
        expect(changedForced).toEqual(changedPlain);
        expect(changedPlain).toContain('d3/s3/t3/f3.ts');
        expect(changedPlain).toContain('new/added.ts');
        expect(changedPlain).toContain('d7/s0/t2/f7.ts');

        expect(budget.pauses).toBeGreaterThan(0);
    });

    it('regression: the root of a small fixed listing is pinned (hashing and ordering are frozen)', async () => {
        const FILES = [
            { path: 'README.md', hash: hex(1) },
            { path: 'src/index.ts', hash: hex(2) },
            { path: 'src/lib/util.ts', hash: hex(3) },
            { path: 'src/lib/zeta.ts', hash: hex(4) },
            { path: 'docs/guide.md', hash: hex(5) },
        ];
        const root = '0e62f74f5b92a6d4723875995eba326617d27e27b7a55db60a84d1051ab1deb4';
        expect((await merkleTree(FILES)).root).toBe(root);
        expect((await merkleTree([...FILES].reverse(), { budget: forced() })).root).toBe(root);
    });
});
