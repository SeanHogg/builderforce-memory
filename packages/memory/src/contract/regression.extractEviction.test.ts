/**
 * src/contract/regression.extractEviction.test.ts — the extract cache never evicted a body: every edit of a file added a content
 * hash kept forever in memory, the snapshot, the extract log and the host's rows. Each invariant of the fix (extractRefs.ts) is
 * pinned by one test named `regression: <what broke>`. A source keeps its current and its ONE previous body.
 */

import type { Extracted } from '../knowledge/model.js';
import { LocalSnapshotStore, type ExtractedLogPersistence, type RowChange, type RowPersistence } from '../knowledge/stores/localSnapshotStore.js';
import { parseLog } from '../knowledge/stores/snapshotFormat.js';
import { memLog, memPersistence } from '../knowledge/testkit.test-util.js';

const body = (hash: string, ...refs: string[]): Extracted[] =>
    refs.map((sourceRef) => ({ sourceRef, sourceHash: hash, nodes: [], edges: [], keys: [] }) as unknown as Extracted);
const N = 25;
const hashOf = (i: number) => `${i}`.padStart(64, '0');
const linesOf = (text: string | undefined) => (text ?? '').split('\n').filter((l) => l.trim()).length;
const keys = async (s: LocalSnapshotStore) => Object.keys(await s.extractedEntries()).sort();

/** A map-backed row host. */
function rowHost(): RowPersistence & { applies: RowChange[]; db: Map<string, Extracted[]> } {
    const db = new Map<string, Extracted[]>();
    const applies: RowChange[] = [];
    return {
        db, applies,
        async load() { return undefined; },
        async apply(change) {
            applies.push(change);
            for (const [h, e] of change.upserts.extracted) db.set(h, e);
            for (const h of change.deletes.extracted) db.delete(h);
            return { generation: applies.length };
        },
        async loadExtracted(h) { return db.get(h); },
        async loadAllExtracted() { return Object.fromEntries(db); },
    };
}

/** The three store modes, each opened fresh over the same persistence by `reopen`. */
function modes(): Array<{ name: string; open(): LocalSnapshotStore }> {
    const snap = memPersistence();
    const logSnap = memPersistence(), log: ExtractedLogPersistence = memLog();
    const rows = rowHost();
    return [
        { name: 'snapshot', open: () => new LocalSnapshotStore(snap) },
        { name: 'log', open: () => new LocalSnapshotStore(logSnap, { extractedLog: log }) },
        { name: 'rows', open: () => new LocalSnapshotStore(undefined, { rows }) },
    ];
}

describe('regression: extract cache eviction', () => {
    it('regression: editing a file N times keeps O(1) bodies — its current and its previous (snapshot mode: memory and the snapshot)', async () => {
        const snap = memPersistence();
        const store = new LocalSnapshotStore(snap);
        for (let i = 0; i < N; i++) await store.putExtracted(hashOf(i), body(hashOf(i), 'src:r::a.ts'));
        expect(await keys(store)).toEqual([hashOf(N - 2), hashOf(N - 1)]);
        expect(Object.keys((JSON.parse(snap.value!) as { extracted: Record<string, unknown> }).extracted).sort()).toEqual([hashOf(N - 2), hashOf(N - 1)]);
        expect(await store.getExtracted(hashOf(0))).toBeUndefined();
    });

    it('regression: editing a file N times keeps O(1) bodies in the extract log once it is compacted, and the compacted log reloads the same', async () => {
        const log = memLog();
        const first = new LocalSnapshotStore(memPersistence(), { extractedLog: log });
        for (let i = 0; i < N; i++) await first.putExtracted(hashOf(i), body(hashOf(i), 'src:r::a.ts'));
        expect(await keys(first)).toEqual([hashOf(N - 2), hashOf(N - 1)]);

        const reopened = new LocalSnapshotStore(memPersistence(), { extractedLog: log });
        expect(await keys(reopened)).toEqual([hashOf(N - 2), hashOf(N - 1)]);
        expect(log.replaces).toBe(1);
        expect(linesOf(log.text)).toBe(2);
        // The compacted log restores who is current: the next edit evicts the previous body, not the current one.
        await reopened.putExtracted(hashOf(N), body(hashOf(N), 'src:r::a.ts'));
        expect(await keys(reopened)).toEqual([hashOf(N - 1), hashOf(N)]);
        expect(await keys(new LocalSnapshotStore(memPersistence(), { extractedLog: log }))).toEqual([hashOf(N - 1), hashOf(N)]);
    });

    it('regression: an evicted body still waiting to be appended is never appended', async () => {
        const log = memLog();
        const store = new LocalSnapshotStore(memPersistence(), { extractedLog: log, checkpointMs: 60_000 });
        await store.batched(async () => {
            for (let i = 0; i < N; i++) await store.putExtracted(hashOf(i), body(hashOf(i), 'src:r::a.ts'));
        });
        expect(linesOf(log.text)).toBe(2);
    });

    it('regression: a throwing log replay surfaces; only a line that does not parse counts as torn', async () => {
        const text = `not json\nnull\n${JSON.stringify({ h: 'x', e: [] })}\n`;
        const seen: string[] = [];
        expect(parseLog(text, (h) => { seen.push(h); }).liveLines.size).toBe(1);
        expect(seen).toEqual(['x']);
        expect(() => parseLog(text, () => { throw new Error('replay failed'); })).toThrow('replay failed');
    });

    it('regression: row mode deletes the evicted body through the delta delete path', async () => {
        const rows = rowHost();
        const store = new LocalSnapshotStore(undefined, { rows });
        for (const i of [1, 2, 3]) await store.putExtracted(hashOf(i), body(hashOf(i), 'src:r::a.ts'));
        const last = rows.applies[rows.applies.length - 1]!;
        expect(last.upserts.extracted.map(([h]) => h)).toEqual([hashOf(3)]);
        expect(last.deletes.extracted).toEqual([hashOf(1)]);
    });

    it('regression: row mode evicts a stored body it never loaded, known only through extractedSources', async () => {
        const rows = rowHost();
        rows.db.set(hashOf(1), body(hashOf(1), 'src:r::a.ts'));
        rows.db.set(hashOf(2), body(hashOf(2), 'src:r::a.ts'));
        const host: RowPersistence = { ...rows, async extractedSources() { return { [hashOf(1)]: ['src:r::a.ts'], [hashOf(2)]: ['src:r::a.ts'] }; } };
        const store = new LocalSnapshotStore(undefined, { rows: host });
        await store.putExtracted(hashOf(3), body(hashOf(3), 'src:r::a.ts'));
        // Both were current (their order unknown): one becomes the previous body, the other is evicted.
        expect(rows.applies[rows.applies.length - 1]!.deletes.extracted).toHaveLength(1);
        expect(rows.db.size).toBe(2);
    });

    it('regression: switching a file back to its previous content hits the cache (one previous body per source is kept)', async () => {
        const store = new LocalSnapshotStore(memPersistence());
        await store.putExtracted(hashOf(1), body(hashOf(1), 'src:r::a.ts'));
        await store.putExtracted(hashOf(2), body(hashOf(2), 'src:r::a.ts'));
        expect(await store.getExtracted(hashOf(1))).toEqual(body(hashOf(1), 'src:r::a.ts'));
    });

    it('regression: a body shared by two files is kept until neither file is current or previous at it', async () => {
        const store = new LocalSnapshotStore(memPersistence());
        const shared = hashOf(1);
        await store.putExtracted(shared, body(shared, 'src:r::a.ts', 'src:r::b.ts'));
        await store.putExtracted(hashOf(2), body(hashOf(2), 'src:r::a.ts'));
        await store.putExtracted(hashOf(3), body(hashOf(3), 'src:r::b.ts'));
        await store.putExtracted(hashOf(4), body(hashOf(4), 'src:r::a.ts'));
        expect(await store.getExtracted(shared)).toBeDefined(); // still b's previous body
        await store.putExtracted(hashOf(5), body(hashOf(5), 'src:r::b.ts'));
        expect(await store.getExtracted(shared)).toBeUndefined();
    });

    it('regression: root-level outputs of different analyzers (~tag hashes) for one ref never evict each other', async () => {
        const store = new LocalSnapshotStore(memPersistence());
        for (const h of ['~graph:1', '~systems:1', '~graph:2', '~graph:3']) await store.putExtracted(h, body(h, 'src:r::'));
        expect(await keys(store)).toEqual(['~graph:2', '~graph:3', '~systems:1']);
    });

    it('regression: an empty body (nothing to extract) and a body without a sourceRef are never evicted by another put', async () => {
        const store = new LocalSnapshotStore(memPersistence());
        await store.putExtracted('bin', []);
        await store.putExtracted('odd', [{ name: 'x' }] as unknown as Extracted[]);
        for (const i of [1, 2, 3]) await store.putExtracted(hashOf(i), body(hashOf(i), 'src:r::a.ts'));
        expect(await keys(store)).toEqual([hashOf(2), hashOf(3), 'bin', 'odd'].sort());
    });

    it.each(modes())('regression: re-put of a shared hash does not evict another file\'s current body ($name mode)', async ({ open }) => {
        // a shares H9 with x, then H0 with b, then moves to H5. New files c (in H0) and y (in H9) are merged into those cached bodies,
        // which still carry a's old entry: the re-puts must not make a current there again (that evicted H5, a's real body).
        const [h9, h0, h5] = [hashOf(9), hashOf(0), hashOf(5)];
        const store = open();
        await store.putExtracted(h9, body(h9, 'src:r::a.ts'));
        await store.putExtracted(h9, body(h9, 'src:r::a.ts', 'src:r::x.ts'));
        await store.putExtracted(h0, body(h0, 'src:r::a.ts'));
        await store.putExtracted(h0, body(h0, 'src:r::a.ts', 'src:r::b.ts'));
        await store.putExtracted(h5, body(h5, 'src:r::a.ts'));
        await store.putExtracted(h0, body(h0, 'src:r::a.ts', 'src:r::b.ts', 'src:r::c.ts'));
        await store.putExtracted(h9, body(h9, 'src:r::a.ts', 'src:r::x.ts', 'src:r::y.ts'));
        expect(await store.getExtracted(h5)).toEqual(body(h5, 'src:r::a.ts'));
        // …and after reopening over the same persistence.
        expect(await open().getExtracted(h5)).toEqual(body(h5, 'src:r::a.ts'));
    });
});
