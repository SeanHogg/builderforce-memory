/**
 * src/contract/regression.recheck.test.ts — the re-check channel between a replica and the review, each bug pinned by one test named
 * `regression: <what broke>`: a replica that holds a file back asks the next review about its hash; the review answers only when the work is done.
 */

import { hashText } from '../ingest/hash.js';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { ExtractFailed, extractJob, type ExtractJob, type ExtractPool } from '../knowledge/extractJob.js';
import { CHANGED_SINCE_PLAN } from '../knowledge/reviewPrefetch.js';
import { DOCS_PACK } from '../knowledge/packs/docs.js';
import type { KnowledgePack } from '../knowledge/model.js';
import { reviewDirectory } from '../knowledge/review.js';
import { extractorManifestKey, filesManifestKey, manifestMap, saveManifestMap } from '../knowledge/sources.js';
import { args, memSource, memStore, ROOT, stateOf, type SpySource } from '../knowledge/testkit.test-util.js';

describe('regression: a root reviewed before binary files were recorded is settled by its next review, without reading a file', () => {
    const text = '# A\nalpha\n';

    it('regression: a binary-by-extension file the manifest names, with no stored body, gets a stored [] and no file is read', async () => {
        const source = memSource({ 'a.md': text, 'img.png': 'x' });
        const manifest = new InMemoryIngestManifest();
        await saveManifestMap(manifest, filesManifestKey(ROOT), new Map([['img.png', 'imghash']]));
        const store = memStore();
        await reviewDirectory(args(source, store, manifest));
        expect(await store.getExtracted('imghash')).toEqual([]);
        expect(source.reads).not.toContain('img.png');
    });

    it('regression: the review reads ONLY the files a replica asked to re-check: a NUL-binary gets [], a text file is extracted, the set is cleared', async () => {
        const binary = 'bin\u0000ary';
        const files = { 'blob.txt': binary, 'a.md': text, 'b.md': '# B\nbeta\n' };
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(memSource(files), memStore(), manifest));
        const legacy = memStore(); // a root reviewed before: the manifest names the files, this store holds no bodies
        await legacy.markRecheck([hashText(binary), hashText(text)]);

        const source = memSource(files);
        await reviewDirectory(args(source, legacy, manifest));
        expect(new Set(source.reads)).toEqual(new Set(['blob.txt', 'a.md'])); // b.md was never named
        expect(await legacy.getExtracted(hashText(binary))).toEqual([]);
        expect((await legacy.getExtracted(hashText(text)))!.length).toBeGreaterThan(0);
        expect(await legacy.getExtracted(hashText(files['b.md']))).toBeUndefined();
        expect(await legacy.recheckHashes()).toEqual([]);

        source.reads.length = 0;
        await reviewDirectory(args(source, legacy, manifest)); // the set is empty now: nothing is read
        expect(source.reads).toEqual([]);
    });

    it('regression: a re-checked hash only a SKIPPED manifest entry carries (here: ignored) is settled with [] and cleared, never left to loop', async () => {
        const manifest = new InMemoryIngestManifest();
        await saveManifestMap(manifest, filesManifestKey(ROOT), new Map([['skip.md', 'skiphash']]));
        const legacy = memStore();
        await legacy.markRecheck(['skiphash', 'gonehash']);
        await reviewDirectory(args(memSource({ '.gitignore': 'skip.md\n', 'skip.md': '# S\nskipped\n', 'a.md': text }), legacy, manifest));
        expect(await legacy.getExtracted('skiphash')).toEqual([]); // no review will ever extract an ignored file
        expect(await legacy.getExtracted('gonehash')).toBeUndefined(); // carried by nothing: nothing to store
        expect(await legacy.recheckHashes()).toEqual([]);
    });

    it('regression: a file whose read fails gets a terminal [] under the hash the manifest holds, so it is never held back forever (and a later read replaces it)', async () => {
        const manifest = new InMemoryIngestManifest();
        await saveManifestMap(manifest, filesManifestKey(ROOT), new Map([['locked.md', 'oldhash']]));
        const store = memStore();
        const inner = memSource({ 'locked.md': '# L\nlocked\n' });
        const locked = { ...inner, read: async () => { throw new Error('EBUSY'); } };
        const report = await reviewDirectory(args(locked as never, store, manifest));
        expect(report.readErrors).toBe(1);
        expect(await store.getExtracted('oldhash')).toEqual([]);
        // The file becomes readable at a new hash: the real body is stored under it, and the terminal answer for the old hash stays harmless.
        await reviewDirectory(args(inner, store, manifest));
        expect((await store.getExtracted(hashText('# L\nlocked\n')))!.length).toBeGreaterThan(0);
    });


    /** A source whose `a.md` read is observed: `onRead(n)` runs on its n-th read and may throw. */
    const watched = (onRead: (n: number) => void): SpySource => {
        const inner = memSource({ 'a.md': text });
        let n = 0;
        return { ...inner, read: async (p: string) => { if (p === 'a.md') onRead(++n); return inner.read(p); } } as SpySource;
    };
    const legacyStore = async (manifest: InMemoryIngestManifest) => {
        await reviewDirectory(args(memSource({ 'a.md': text }), memStore(), manifest));
        const legacy = memStore(); // the manifest names a.md; this store holds no body for it
        await legacy.markRecheck([hashText(text)]);
        return legacy;
    };

    it('regression: a review stopped mid-way (after the backfill queued the file) keeps the re-check request and records no terminal [], and a complete one clears it', async () => {
        const manifest = new InMemoryIngestManifest();
        const legacy = await legacyStore(manifest);
        const stop = new AbortController();
        // Read 1 is the backfill's, read 2 is the review's: the run is stopped there and the read rejects.
        const source = watched((n) => { if (n === 2) { stop.abort(); throw new Error('aborted'); } });
        await reviewDirectory(args(source, legacy, manifest, { signal: stop.signal })).catch(() => undefined);
        expect(await legacy.recheckHashes()).toEqual([hashText(text)]);
        expect(await legacy.getExtracted(hashText(text))).toBeUndefined();
        await reviewDirectory(args(memSource({ 'a.md': text }), legacy, manifest));
        expect(await legacy.recheckHashes()).toEqual([]);
        expect((await legacy.getExtracted(hashText(text)))!.length).toBeGreaterThan(0);
    });

    it('regression: a read error at the manifest\'s own bytes keeps the request and makes the file owed; the next review stores the body', async () => {
        const manifest = new InMemoryIngestManifest();
        const legacy = await legacyStore(manifest);
        const report = await reviewDirectory(args(watched(() => { throw new Error('EBUSY'); }), legacy, manifest));
        expect(report.readErrors).toBe(1);
        expect(await legacy.getExtracted(hashText(text))).toBeUndefined(); // not a terminal answer: the bytes may be readable next time
        expect(await legacy.recheckHashes()).toEqual([hashText(text)]);
        expect((await stateOf(manifest)).owed).toContain('a.md');
        await reviewDirectory(args(memSource({ 'a.md': text }), legacy, manifest)); // a normal review: it is owed, so it is retried
        expect((await legacy.getExtracted(hashText(text)))!.length).toBeGreaterThan(0);
        expect(await legacy.recheckHashes()).toEqual([]);
        expect((await stateOf(manifest)).owed).toEqual([]);
    });

    it('regression: a PROVEN-final extraction failure (the parser threw) at the same bytes gets a terminal [] and settles the request', async () => {
        const manifest = new InMemoryIngestManifest();
        const legacy = await legacyStore(manifest);
        const exploding: ExtractPool = async () => { throw new ExtractFailed('parser exploded'); }; // proven final: the parser itself threw
        const report = await reviewDirectory(args(memSource({ 'a.md': text }), legacy, manifest, { extractPool: exploding, deferRead: true }));
        expect(report.extractErrors).toBe(1);
        expect(await legacy.getExtracted(hashText(text))).toEqual([]);
        expect(await legacy.recheckHashes()).toEqual([]);
    });

    it('regression: the body is stored BEFORE the request is cleared (a store that observes the put still sees the request)', async () => {
        const manifest = new InMemoryIngestManifest();
        const legacy = await legacyStore(manifest);
        let sawRequest: boolean | undefined;
        const put = legacy.putExtracted.bind(legacy);
        legacy.putExtracted = async (h, e) => { sawRequest ??= await legacy.hasRecheck(h); await put(h, e); };
        await reviewDirectory(args(memSource({ 'a.md': text }), legacy, manifest));
        expect(sawRequest).toBe(true);
        expect(await legacy.recheckHashes()).toEqual([]);
    });


    /** The request is kept, the file is owed, and nothing terminal is stored: the failure said nothing final about the file. */
    const expectTransient = async (legacy: ReturnType<typeof memStore>, manifest: InMemoryIngestManifest): Promise<void> => {
        expect(await legacy.getExtracted(hashText(text))).toBeUndefined();
        expect(await legacy.recheckHashes()).toEqual([hashText(text)]);
        expect((await stateOf(manifest)).owed).toContain('a.md');
    };

    it('regression: a deferRead pool whose read fails (EBUSY) at the manifest\'s bytes is transient, not a terminal []', async () => {
        const manifest = new InMemoryIngestManifest();
        const legacy = await legacyStore(manifest);
        const busy: ExtractPool = async () => { throw new Error('EBUSY: resource busy or locked'); };
        const report = await reviewDirectory(args(memSource({ 'a.md': text }), legacy, manifest, { extractPool: busy, deferRead: true }));
        expect(report.extractErrors).toBe(1);
        await expectTransient(legacy, manifest);
    });

    it('regression: a pool that times out (plain Error "timeout") is transient', async () => {
        const manifest = new InMemoryIngestManifest();
        const legacy = await legacyStore(manifest);
        const stuck: ExtractPool = async () => { throw new Error('timeout'); };
        await reviewDirectory(args(memSource({ 'a.md': text }), legacy, manifest, { extractPool: stuck, deferRead: true }));
        await expectTransient(legacy, manifest);
    });

    it('guard: an in-thread read that does not match the planned hash (changed since plan) at the same hash is transient', async () => {
        const manifest = new InMemoryIngestManifest();
        const legacy = await legacyStore(manifest);
        const drifting = { ...memSource({ 'a.md': text }), hashOf: () => 'some-other-hash' } as SpySource;
        const report = await reviewDirectory(args(drifting, legacy, manifest));
        expect(report.failures).toEqual([{ path: 'a.md', phase: 'extract', reason: 'changed since plan' }]);
        await expectTransient(legacy, manifest);
    });

    it('guard: an extraction past its ceiling (a slow pool under a tiny extractMs) is transient', async () => {
        const manifest = new InMemoryIngestManifest();
        const legacy = await legacyStore(manifest);
        const slow: ExtractPool = () => new Promise(() => undefined);
        await reviewDirectory(args(memSource({ 'a.md': text }), legacy, manifest, { extractPool: slow, deferRead: true, timeouts: { extractMs: 5 } }));
        await expectTransient(legacy, manifest);
    });

    it('regression: the third transient failure in a row at the same hash is recorded as final, and STICKY: the file is not re-read while its bytes stand', async () => {
        const manifest = new InMemoryIngestManifest();
        const legacy = await legacyStore(manifest);
        let calls = 0;
        const busy: ExtractPool = async () => { calls += 1; throw new Error('EBUSY'); };
        const run = () => reviewDirectory(args(memSource({ 'a.md': text }), legacy, manifest, { extractPool: busy, deferRead: true }));
        await run();
        await run();
        await expectTransient(legacy, manifest); // two in a row: still asked for
        await run();
        expect(await legacy.getExtracted(hashText(text))).toEqual([]); // the third: final
        expect(await legacy.recheckHashes()).toEqual([]);
        const settledAt = calls;
        const report = await run();
        expect(calls).toBe(settledAt);                // sticky: not re-read ...
        expect(report.failures).toEqual([]);          // ... and not reported failed again
        expect((await stateOf(manifest)).owed).toEqual([]);
    });

    it('regression: a pool that rejects with "changed since plan" is transient (it was terminal before)', async () => {
        const manifest = new InMemoryIngestManifest();
        const legacy = await legacyStore(manifest);
        const drifting: ExtractPool = async () => { throw new Error(CHANGED_SINCE_PLAN); };
        await reviewDirectory(args(memSource({ 'a.md': text }), legacy, manifest, { extractPool: drifting, deferRead: true }));
        await expectTransient(legacy, manifest);
    });

    it('regression: the failure count resets on a success: fail, fail, succeed, then fail, fail is still transient and `fail:a.md` is gone after the success', async () => {
        const manifest = new InMemoryIngestManifest();
        const source = memSource({ 'a.md': text });
        await reviewDirectory(args(source, memStore(), manifest));
        const busy: ExtractPool = async () => { throw new Error('EBUSY'); };
        const reading: ExtractPool = async (job) => extractJob({ ...job, doc: await source.read(job.path) } as ExtractJob, [DOCS_PACK]);
        const failIds = async () => ((await manifest.get(extractorManifestKey(ROOT))) ?? []).map((f) => f.id).filter((id) => id.startsWith('fail:'));
        const withPool = (store: ReturnType<typeof memStore>, pool: ExtractPool, extra = {}) =>
            reviewDirectory(args(source, store, manifest, { extractPool: pool, deferRead: true, ...extra }));
        await withPool(memStore(), busy, { force: true });
        await withPool(memStore(), busy, { force: true });
        expect(await failIds()).toEqual(['fail:a.md']);
        await withPool(memStore(), reading, { force: true }); // a success
        expect(await failIds()).toEqual([]);
        const last = memStore();
        await withPool(last, busy, { force: true });
        await withPool(last, busy, { force: true });
        expect(await last.getExtracted(hashText(text))).toBeUndefined(); // two since the success: not final (it would be the fourth without the reset)
        expect((await stateOf(manifest)).owed).toContain('a.md');
    });

    it('regression: two paths share a hash; the unchanged one stays readable content, so a final failure of the other stores no [] under it and the request stays asked', async () => {
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(memSource({ 'x.md': text, 'y.md': text }), memStore(), manifest)); // both at hash H
        const legacy = memStore();
        await legacy.markRecheck([hashText(text)]);
        const inner = memSource({ 'x.md': text, 'y.md': '# Y\nchanged\n' });
        const source = { ...inner, read: async (p: string) => { if (p === 'y.md') throw new Error('EBUSY'); return inner.read(p); } } as SpySource;
        await reviewDirectory(args(source, legacy, manifest, { only: ['y.md'] })); // y.md moved off H and cannot be read
        expect(await legacy.getExtracted(hashText(text))).toBeUndefined();
        expect(await legacy.recheckHashes()).toEqual([hashText(text)]);
    });

    it('regression: a Stop that lands during the LAST file still applies it, but retires nothing, runs no analyzers and stamps no digest', async () => {
        const manifest = new InMemoryIngestManifest();
        const store = memStore();
        await reviewDirectory(args(memSource({ 'a.md': text, 'gone.md': '# G\nbye\n' }), store, manifest));
        const stamped = (await store.scopes()).find((s) => s.rootId === ROOT)!.lastDigest;
        const stop = new AbortController();
        const report = await reviewDirectory(args(memSource({ 'a.md': '# A\nedited\n' }), store, manifest, {
            signal: stop.signal, onProgress: (p) => { if (p.phase === 'write') stop.abort(); },
        }));
        expect(report.aborted).toBe(true);
        expect([...(await manifestMap(manifest, filesManifestKey(ROOT))).keys()]).toContain('gone.md'); // not retired
        expect((await store.scopes()).find((s) => s.rootId === ROOT)!.lastDigest).toBe(stamped);
    });

    it('guard: a forced review whose read fails keeps the request (the backfill does not clear a changed file\'s hash)', async () => {
        const manifest = new InMemoryIngestManifest();
        const legacy = await legacyStore(manifest);
        const failing = { ...memSource({ 'a.md': text }), read: async () => { throw new Error('EBUSY'); } } as SpySource;
        await reviewDirectory(args(failing, legacy, manifest, { force: true }));
        expect(await legacy.recheckHashes()).toEqual([hashText(text)]);
    });

    it('guard: a file the packs accept at plan time but not once read (job "none") is recorded as nothing-to-extract and its request is answered', async () => {
        const picky: KnowledgePack = { ...DOCS_PACK, parsers: [{ ...DOCS_PACK.parsers[0]!, accepts: (s) => s.content.kind === 'text' && s.content.text === '' }] };
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(memSource({ 'a.md': text }), memStore(), manifest, { packs: [picky] }));
        const legacy = memStore();
        await legacy.markRecheck([hashText(text)]);
        await reviewDirectory(args(memSource({ 'a.md': text }), legacy, manifest, { packs: [picky], force: true }));
        expect(await legacy.getExtracted(hashText(text))).toEqual([]);
        expect(await legacy.recheckHashes()).toEqual([]);
    });

    it('regression: an unchanged text file with no stored body is NOT given [] (it is "not extracted yet")', async () => {
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(memSource({ 'a.md': text }), memStore(), manifest));
        const legacy = memStore();
        await reviewDirectory(args(memSource({ 'a.md': text }), legacy, manifest));
        expect(await legacy.getExtracted(hashText(text))).toBeUndefined();
    });

    it('regression: a healthy root performs no extra reads on an incremental review', async () => {
        const source = memSource({ 'a.md': text, 'blob.txt': 'bin\u0000ary' });
        const manifest = new InMemoryIngestManifest();
        const store = memStore();
        await reviewDirectory(args(source, store, manifest));
        source.reads.length = 0;
        await reviewDirectory(args(source, store, manifest));
        expect(source.reads).toEqual([]);
    });
});
