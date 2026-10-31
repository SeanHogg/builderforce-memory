/**
 * src/knowledge/review.batched.test.ts
 * A review over a store that can `batched` writes the snapshot a bounded number of times
 * (not per file), and persists the snapshot BEFORE it saves the files manifest: a manifest
 * ahead of the snapshot would make the next review skip files whose knowledge was lost.
 * `preloadExtracted` is asked once, with the distinct changed hashes, before any `getExtracted`.
 */

import { hashText } from '../ingest/hash.js';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import type { IngestManifest } from '../ingest/types.js';
import type { KnowledgeStore } from './model.js';
import { reviewDirectory } from './review.js';
import { filesManifestKey } from './sources.js';
import { LocalSnapshotStore, type SnapshotPersistence } from './stores/localSnapshotStore.js';
import { args, fakeJudge, memSource, memStore, ROOT, similar } from './testkit.test-util.js';

interface Log extends SnapshotPersistence { events: string[]; snapshots: string[]; value?: string }

function logging(events: string[] = []): Log {
    const p: Log = {
        events, snapshots: [],
        async read() { return p.value; },
        async write(s: string) { events.push('snapshot'); p.snapshots.push(s); p.value = s; },
    };
    return p;
}

function loggingManifest(events: string[]): IngestManifest {
    const inner = new InMemoryIngestManifest();
    return {
        get: (k) => inner.get(k),
        set: async (k, v) => { if (k === filesManifestKey(ROOT)) events.push('files-manifest'); await inner.set(k, v); },
        delete: (k) => inner.delete(k),
    };
}

const files = (n: number) => Object.fromEntries(Array.from({ length: n }, (_, i) => [`d${i}.md`, `# T${i}\nbody ${i}\n`]));
const storeOver = (p: SnapshotPersistence) => new LocalSnapshotStore(p, { now: () => 'T', nowMs: () => 0 });
/** The snapshot as stored, minus the wall-clock stamp the review adds. */
const content = (text: string) => {
    const s = JSON.parse(text);
    for (const scope of s.scopes) delete scope.lastReviewedAt;
    return s;
};

describe('reviewDirectory over a batched store', () => {
    it('writes the snapshot a bounded number of times and stores what an unbatched review stores', async () => {
        const source = memSource(files(50));
        const batchedP = logging();
        await reviewDirectory(args(source, storeOver(batchedP), new InMemoryIngestManifest(), { judgement: fakeJudge() }));
        // One flush before the manifest, one at the end of the batch (the lastDigest stamp).
        expect(batchedP.snapshots).toHaveLength(2);

        const plainP = logging();
        const plain = storeOver(plainP);
        (plain as { batched?: unknown }).batched = undefined;
        await reviewDirectory(args(source, plain, new InMemoryIngestManifest(), { judgement: fakeJudge() }));
        expect(plainP.snapshots.length).toBeGreaterThan(100);
        expect(content(batchedP.value!)).toEqual(content(plainP.value!));
    });

    it('persists the snapshot before it saves the files manifest', async () => {
        const events: string[] = [];
        const p = logging(events);
        await reviewDirectory(args(memSource(files(20)), storeOver(p), loggingManifest(events), { judgement: fakeJudge() }));
        expect(events).toEqual(['snapshot', 'files-manifest', 'snapshot']);
        // The write that precedes the manifest already holds every file's work.
        expect(Object.keys(JSON.parse(p.snapshots[0]!).extracted)).toHaveLength(20);
        expect(JSON.parse(p.snapshots[0]!).facts[ROOT].length).toBeGreaterThan(0);
    });

    it('an aborted review still persists what it wrote', async () => {
        const source = memSource(files(10));
        const controller = new AbortController();
        const read = source.read.bind(source);
        let reads = 0;
        source.read = async (path) => {
            reads += 1;
            if (reads === 4) controller.abort();
            return read(path);
        };
        const p = logging();
        const store = storeOver(p);
        const report = await reviewDirectory(args(source, store, new InMemoryIngestManifest(), { judgement: fakeJudge(), signal: controller.signal }));
        expect(report.aborted).toBe(true);
        expect(report.extracted).toBeLessThan(10);
        const persisted = content(p.value!);
        const byKey = (a: { key: string }, b: { key: string }) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
        expect(persisted.facts[ROOT].length).toBeGreaterThan(0);
        expect([...persisted.facts[ROOT]].sort(byKey)).toEqual([...(await store.facts({ rootId: ROOT }))].sort(byKey));
        expect(Object.keys(persisted.extracted)).toHaveLength(report.extracted);
    });

    it('stores the relations the compare phase produced before the files manifest is saved', async () => {
        const events: string[] = [];
        const p = logging(events);
        const text = { 'a.md': '# A\nalpha one\n', 'b.md': '# B\nbeta two\n' };
        await reviewDirectory(args(memSource(text), storeOver(p), loggingManifest(events), { judgement: fakeJudge(), similar }));
        const manifestAt = events.indexOf('files-manifest');
        const writesBefore = events.slice(0, manifestAt).filter((e) => e === 'snapshot').length;
        const relations = (JSON.parse(p.snapshots[writesBefore - 1]!).relations[ROOT] ?? []) as Array<{ kind: string }>;
        expect(relations.some((r) => r.kind === 'contradicts')).toBe(true);
        // Per-file writes are held; the compare relations are stored, then flushed once, then the digest is stamped.
        expect(events).toEqual(['snapshot', 'files-manifest', 'snapshot']);
    });

    it('a store without batched still works', async () => {
        const p = logging();
        const store = storeOver(p);
        (store as { batched?: unknown }).batched = undefined;
        const report = await reviewDirectory(args(memSource(files(5)), store, new InMemoryIngestManifest(), { judgement: fakeJudge() }));
        expect(report.extracted).toBe(5);
        expect(content(p.value!).facts[ROOT].length).toBeGreaterThan(0);
    });
});

describe('reviewDirectory preloads the extract cache', () => {
    /** A store over a memory store that logs every `getExtracted` and `preloadExtracted`. */
    function spyStore(withPreload = true) {
        const inner = memStore();
        const calls: Array<{ op: 'preload'; hashes: readonly string[] } | { op: 'get'; hash: string }> = [];
        const store: KnowledgeStore = {
            scopes: () => inner.scopes(),
            facts: (s) => inner.facts(s),
            relations: (s) => inner.relations(s),
            write: (b) => inner.write(b),
            markStale: (s, r, h) => inner.markStale(s, r, h),
            getExtracted: async (h) => { calls.push({ op: 'get', hash: h }); return inner.getExtracted(h); },
            putExtracted: (h, e) => inner.putExtracted(h, e),
            ...(withPreload ? { preloadExtracted: async (hashes: readonly string[]) => { calls.push({ op: 'preload', hashes }); } } : {}),
        };
        return { store, calls };
    }

    it('is called once with the distinct changed hashes, before the first getExtracted', async () => {
        const text = { 'a.md': '# A\nalpha\n', 'b.md': '# A\nalpha\n', 'c.md': '# C\ngamma\n' };
        const { store, calls } = spyStore();
        await reviewDirectory(args(memSource(text), store));
        const preloads = calls.filter((c) => c.op === 'preload');
        expect(preloads).toHaveLength(1);
        const first = preloads[0] as { hashes: readonly string[] };
        expect([...first.hashes].sort()).toEqual([hashText(text['a.md']), hashText(text['c.md'])].sort());
        expect(calls[0]!.op).toBe('preload');
        expect(calls.some((c) => c.op === 'get')).toBe(true);
    });

    it('is not called when nothing changed', async () => {
        const source = memSource(files(3));
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, memStore(), manifest));
        const { store, calls } = spyStore();
        await reviewDirectory(args(source, store, manifest));
        expect(calls).toEqual([]);
    });

    it('a store without it still works', async () => {
        const { store, calls } = spyStore(false);
        const report = await reviewDirectory(args(memSource(files(3)), store));
        expect(report.extracted).toBe(3);
        expect(calls.every((c) => c.op === 'get')).toBe(true);
    });

    it('a preload failure propagates', async () => {
        const { store } = spyStore();
        store.preloadExtracted = async () => { throw new Error('bulk fetch failed'); };
        await expect(reviewDirectory(args(memSource(files(2)), store))).rejects.toThrow('bulk fetch failed');
    });
});
