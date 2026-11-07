/**
 * knowledge/pack/pack.reconcile.test.ts — the delta-finding halves of PRD 41 A11 rule 2: the manifest
 * tree (reconcile path) and the within-file diff (EXTR `prior`).
 */

import { applyEntryDiffs, decodePack, diffEntries, encodePack, differingBuckets, leafDelta, manifestTree, MANIFEST_BUCKETS, rootLeaves, ZERO_DIGEST, type ManifestLeaf } from './index.js';
import { extractionOf, gen, genPack, ROOT } from './pack.testkit.test-util.js';

const rowsOf = (seed: number, files: number, facts = files) => {
    const p = genPack(seed, { files, facts, rich: false });
    return { files: p.files!, facts: p.facts ?? [], relations: p.relations ?? [] };
};

describe('manifestTree', () => {
    it('is the same on two sides with equal state, whatever the order of the rows', () => {
        const rows = rowsOf(1, 200);
        const a = manifestTree(rootLeaves(ROOT, rows));
        const b = manifestTree(rootLeaves(ROOT, { files: [...rows.files].reverse(), facts: [...rows.facts].reverse(), relations: [...rows.relations].reverse() }));
        expect(a.root).toBe(b.root);
        expect(a.buckets).toHaveLength(MANIFEST_BUCKETS);
        expect(differingBuckets(a.buckets, b.buckets)).toEqual([]);
    });

    it('finds exactly the buckets of the changed, added and removed keys, and the leaf delta inside them', () => {
        const rows = rowsOf(2, 600, 0);
        const server = manifestTree(rootLeaves(ROOT, rows));
        const changed = rows.files.map((f, i) => (i < 6 ? { ...f, hash: 'c'.repeat(64) } : f));
        const local = manifestTree(rootLeaves(ROOT, { ...rows, files: [...changed.slice(0, -3), { path: 'brand/new.ts', hash: 'd'.repeat(64), bytes: 1 }] }));
        expect(local.root).not.toBe(server.root);
        const differing = differingBuckets(local.buckets, server.buckets);
        expect(differing.length).toBeGreaterThan(0);
        expect(differing.length).toBeLessThanOrEqual(6 + 3 + 1 + 5);
        const remote: ManifestLeaf[] = differing.flatMap((b) => server.leaves.get(b) ?? []);
        const delta = leafDelta(local, remote, differing);
        expect(delta.upsert.sort()).toEqual([...rows.files.slice(0, 6).map((f) => `f:${f.path}`), 'f:brand/new.ts'].sort());
        expect(delta.remove.sort()).toEqual(rows.files.slice(-3).map((f) => `f:${f.path}`).sort());
    });

    it('sends nothing for equal roots', () => {
        const rows = rowsOf(3, 50);
        const a = manifestTree(rootLeaves(ROOT, rows)), b = manifestTree(rootLeaves(ROOT, rows));
        expect(differingBuckets(a.buckets, b.buckets)).toEqual([]);
    });

    it('answers a manifest GET in at most 128 KB and a leaf request losslessly', async () => {
        const tree = manifestTree(rootLeaves(ROOT, rowsOf(4, 300)));
        const bytes = await encodePack({ rootId: ROOT, baseGeneration: 3, targetDigest: tree.root, buckets: tree.buckets });
        expect(bytes.length).toBeLessThanOrEqual(128 * 1024 + 256);
        expect((await decodePack(bytes)).buckets).toEqual(tree.buckets);
        const leaves = [...tree.leaves.values()].flat().slice(0, 50);
        const back = await decodePack(await encodePack({ rootId: ROOT, baseGeneration: 0, targetDigest: ZERO_DIGEST, leaves }));
        expect(back.leaves).toEqual([...leaves].sort((a, b) => (a.key < b.key ? -1 : 1)));
    });
});

describe('within-file diff', () => {
    it('adding one function sends that node and its edges, not the file extraction, and reapplies exactly', async () => {
        const g = gen(5);
        const targets = Array.from({ length: 12 }, (_, i) => `src/t${i}.ts`);
        const prior = extractionOf(g, 'src/big.ts', 'a'.repeat(64), targets, 400, false);
        const next = { ...prior, sourceHash: 'b'.repeat(64), nodes: [...prior.nodes, { ...prior.nodes[1]!, ref: 'src:ws:gen::src/big.ts#added' as typeof prior.nodes[0]['ref'], label: 'added' }],
            edges: [...prior.edges, { ...prior.edges[0]!, to: 'src:ws:gen::src/added-target.ts' as typeof prior.edges[0]['to'] }] };
        const diffs = diffEntries([prior], [next])!;
        expect(diffs).toBeDefined();
        expect(diffs[0]!.nodes.add).toHaveLength(1);
        expect(diffs[0]!.nodes.remove).toEqual([]);
        expect(diffs[0]!.edges.add).toHaveLength(1);
        expect(applyEntryDiffs([prior], diffs)).toEqual([next]);

        const base = { rootId: ROOT, baseGeneration: 1, targetDigest: ZERO_DIGEST };
        const small = await encodePack({ ...base, extracted: [{ kind: 'diff', hash: next.sourceHash, prior: prior.sourceHash, entries: diffs }] });
        const whole = await encodePack({ ...base, extracted: [{ kind: 'full', hash: next.sourceHash, entries: [next] }] });
        expect(small.length * 5).toBeLessThan(whole.length);
        const back = await decodePack(small);
        expect(applyEntryDiffs([prior], (back.extracted![0] as { entries: typeof diffs }).entries)).toEqual([next]);
    });

    it('handles removed, changed and inserted items at any position', () => {
        const g = gen(6);
        const prior = extractionOf(g, 'src/a.ts', 'a'.repeat(64), ['src/x.ts', 'src/y.ts'], 6, true);
        const nodes = prior.nodes.filter((_, i) => i !== 2).map((n, i) => (i === 3 ? { ...n, label: 'renamed' } : n));
        nodes.splice(1, 0, { ...prior.nodes[2]!, ref: 'src:ws:gen::src/a.ts#mid' as typeof prior.nodes[0]['ref'] });
        const next = { ...prior, sourceHash: 'e'.repeat(64), nodes };
        expect(applyEntryDiffs([prior], diffEntries([prior], [next])!)).toEqual([next]);
    });

    it('falls back to a whole entry when the surviving items change order or an identity repeats', () => {
        const g = gen(7);
        const prior = extractionOf(g, 'src/a.ts', 'a'.repeat(64), ['src/x.ts'], 4, false);
        expect(diffEntries([prior], [{ ...prior, nodes: [...prior.nodes].reverse() }])).toBeUndefined();
        expect(diffEntries([prior], [{ ...prior, nodes: [...prior.nodes, prior.nodes[0]!] }])).toBeUndefined();
        expect(diffEntries([], [prior])).toBeUndefined();
    });
});
