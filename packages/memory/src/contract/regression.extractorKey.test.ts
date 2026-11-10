/**
 * src/contract/regression.extractorKey.test.ts — the caller-supplied extractor key (`extractorKey`): an input the host's extractors
 * read besides a file's bytes (agent-tools: a digest of the tsconfig path aliases). A changed key re-extracts unchanged files and
 * owes them; no key behaves exactly as before keys existed. Each bug pinned by one test named `regression: …`.
 */

import { hashText } from '../ingest/hash.js';
import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { extractFileDetailed } from '../knowledge/extractFile.js';
import { EXTRACTOR_KEY_MISMATCH, extractJob, type ExtractJob, type ExtractPool } from '../knowledge/extractJob.js';
import type { Extracted } from '../knowledge/model.js';
import type { ReviewArgs } from '../knowledge/reviewModel.js';
import type { ReviewStep } from '../knowledge/reviewRunModel.js';
import { loadExtractorState } from '../knowledge/owedFiles.js';
import { decodePack, encodePack } from '../knowledge/pack/index.js';
import { ZERO_DIGEST } from '../knowledge/pack/types.js';
import { DOCS_PACK } from '../knowledge/packs/docs.js';
import { planDetail } from '../knowledge/plan.js';
import { reviewDirectory } from '../knowledge/review.js';
import { EXTRACTOR_VERSION, extractorManifestKey, isCurrentExtractor } from '../knowledge/sources.js';
import { args, FILES, memSource, memStore, ref, ROOT } from '../knowledge/testkit.test-util.js';

/** One review; the extract step outcome per file ('ok' = extracted now, 'cached' = reused). */
async function review(
    ctx: { source: ReturnType<typeof memSource>; store: ReturnType<typeof memStore>; manifest: InMemoryIngestManifest }, extractorKey?: string,
    extra: Partial<ReviewArgs> = {},
) {
    const steps: ReviewStep[] = [];
    await reviewDirectory(args(ctx.source, ctx.store, ctx.manifest, {
        ...(extractorKey !== undefined ? { extractorKey } : {}), ...extra, onStep: (s) => { steps.push(s); extra.onStep?.(s); },
    }));
    return Object.fromEntries(steps.filter((s) => s.phase === 'extract').map((s) => [s.path, s.outcome]));
}

const fresh = () => ({ source: memSource({ ...FILES }), store: memStore(), manifest: new InMemoryIngestManifest() });
const ALL_OWED = [['a.md', true, true], ['b.md', true, true], ['c.md', true, true]];
const NONE_OWED = [['a.md', false, false], ['b.md', false, false], ['c.md', false, false]];

const plan = async (ctx: { source: ReturnType<typeof memSource>; manifest: InMemoryIngestManifest }, extractorKey?: string) =>
    (await planDetail(ctx.source, ctx.manifest, [DOCS_PACK], extractorKey !== undefined ? { extractorKey } : {})).files.map((f) => [f.path, f.changed, f.owed]);

const storedOf = async (store: ReturnType<typeof memStore>, path: string): Promise<Extracted | undefined> =>
    (await store.getExtracted(hashText(FILES[path as keyof typeof FILES])))?.find((e) => e.sourceRef === ref(path));

describe('regression: extractor key', () => {
    it('regression: a changed extractorKey re-extracts unchanged files and owes them until a run finishes them', async () => {
        const ctx = { source: memSource({ ...FILES }), store: memStore(), manifest: new InMemoryIngestManifest() };
        await review(ctx, 'aliases@1');
        expect((await loadExtractorState(ctx.manifest, ROOT)).extractorKey).toBe('aliases@1');
        expect((await storedOf(ctx.store, 'a.md'))?.extractorKey).toBe('aliases@1');
        // Same key, same bytes: nothing owed.
        expect(await plan(ctx, 'aliases@1')).toEqual([['a.md', false, false], ['b.md', false, false], ['c.md', false, false]]);

        // The tsconfig changed: same bytes, another key. Every file is owed, and the extract cache is bypassed.
        expect(await plan(ctx, 'aliases@2')).toEqual([['a.md', true, true], ['b.md', true, true], ['c.md', true, true]]);
        expect(await review(ctx, 'aliases@2')).toEqual({ 'a.md': 'ok', 'b.md': 'ok', 'c.md': 'ok' });
        expect((await storedOf(ctx.store, 'a.md'))?.extractorKey).toBe('aliases@2');
        expect((await loadExtractorState(ctx.manifest, ROOT)).extractorKey).toBe('aliases@2');
        expect(await plan(ctx, 'aliases@2')).toEqual([['a.md', false, false], ['b.md', false, false], ['c.md', false, false]]);
    });

    it('regression: a key given where none was recorded owes every file and does not reuse the keyless extracts', async () => {
        const ctx = { source: memSource({ ...FILES }), store: memStore(), manifest: new InMemoryIngestManifest() };
        await review(ctx);
        expect(await plan(ctx, 'aliases@1')).toEqual([['a.md', true, true], ['b.md', true, true], ['c.md', true, true]]);
        const one = await extractFileDetailed({
            rootId: ROOT, path: 'a.md', hash: hashText(FILES['a.md']), doc: await ctx.source.read('a.md'), packs: [DOCS_PACK], store: ctx.store,
            extractorKey: 'aliases@1',
        });
        expect(one?.cached).toBe(false);
        expect(one?.extracted.extractorKey).toBe('aliases@1');
    });

    it('no key keeps today\'s behaviour: keyless records match, nothing is owed, nothing records a key', async () => {
        const ctx = { source: memSource({ ...FILES }), store: memStore(), manifest: new InMemoryIngestManifest() };
        expect(await review(ctx)).toEqual({ 'a.md': 'ok', 'b.md': 'ok', 'c.md': 'ok' });
        expect(await ctx.manifest.get(extractorManifestKey(ROOT))).toEqual([{ id: 'version', hash: String(EXTRACTOR_VERSION) }]);
        expect('extractorKey' in ((await storedOf(ctx.store, 'a.md')) ?? {})).toBe(false);
        expect(await plan(ctx)).toEqual([['a.md', false, false], ['b.md', false, false], ['c.md', false, false]]);
        // An old keyless record is reused by a keyless call.
        const again = await extractFileDetailed({
            rootId: ROOT, path: 'a.md', hash: hashText(FILES['a.md']), doc: await ctx.source.read('a.md'), packs: [DOCS_PACK], store: ctx.store,
        });
        expect(again?.cached).toBe(true);
    });

    it('isCurrentExtractor is the one comparison: version and key, an empty key read as none', () => {
        expect(isCurrentExtractor({ version: EXTRACTOR_VERSION, key: undefined }, undefined)).toBe(true);
        expect(isCurrentExtractor({ version: EXTRACTOR_VERSION, key: '' }, undefined)).toBe(true);
        expect(isCurrentExtractor({ version: EXTRACTOR_VERSION, key: 'k' }, 'k')).toBe(true);
        expect(isCurrentExtractor({ version: EXTRACTOR_VERSION, key: undefined }, 'k')).toBe(false);
        expect(isCurrentExtractor({ version: EXTRACTOR_VERSION, key: 'k' }, 'j')).toBe(false);
        expect(isCurrentExtractor({ version: EXTRACTOR_VERSION - 1, key: 'k' }, 'k')).toBe(false);
    });

    it('regression: a key-only change estimates no model calls; a version upgrade still does', async () => {
        const ctx = fresh();
        await review(ctx, 'aliases@1');
        const estimate = async (key: string) => (await planDetail(ctx.source, ctx.manifest, [DOCS_PACK], { extractorKey: key, compare: false })).plan;
        const same = await estimate('aliases@1');
        const keyed = await estimate('aliases@2');
        expect(keyed.changed).toHaveLength(3); // every file is owed …
        expect(keyed.estimatedModelCalls).toBe(same.estimatedModelCalls); // … at no extra model calls
        await ctx.manifest.delete(extractorManifestKey(ROOT)); // a version upgrade (no entry recorded)
        expect((await estimate('aliases@1')).estimatedModelCalls).toBeGreaterThan(keyed.estimatedModelCalls);
    });

    it('regression: an aborted run under a new key keeps the old key, so every file stays owed', async () => {
        const ctx = fresh();
        await review(ctx, 'aliases@1');
        const stop = new AbortController();
        await review(ctx, 'aliases@2', { signal: stop.signal, onStep: (s) => { if (s.phase === 'extract') stop.abort(); } });
        expect((await loadExtractorState(ctx.manifest, ROOT)).extractorKey).toBe('aliases@1');
        expect(await plan(ctx, 'aliases@2')).toEqual(ALL_OWED);
        // The finishing run reuses what the aborted one extracted under the new key, and then records it.
        await review(ctx, 'aliases@2');
        expect((await loadExtractorState(ctx.manifest, ROOT)).extractorKey).toBe('aliases@2');
        expect(await plan(ctx, 'aliases@2')).toEqual(NONE_OWED);
    });

    it('regression: Continue runs under the plan\'s key when given none, and under an overriding key when given one', async () => {
        const viaPlan = fresh();
        await review(viaPlan, 'aliases@1');
        const detail = await planDetail(viaPlan.source, viaPlan.manifest, [DOCS_PACK], { extractorKey: 'aliases@2' });
        expect(await review(viaPlan, undefined, { plan: detail })).toEqual({ 'a.md': 'ok', 'b.md': 'ok', 'c.md': 'ok' });
        expect((await storedOf(viaPlan.store, 'a.md'))?.extractorKey).toBe('aliases@2');
        expect((await loadExtractorState(viaPlan.manifest, ROOT)).extractorKey).toBe('aliases@2');

        const overridden = fresh();
        await review(overridden, 'aliases@1');
        const stale = await planDetail(overridden.source, overridden.manifest, [DOCS_PACK], { extractorKey: 'aliases@2' });
        await review(overridden, 'aliases@3', { plan: stale });
        expect((await storedOf(overridden.store, 'a.md'))?.extractorKey).toBe('aliases@3');
        expect((await loadExtractorState(overridden.manifest, ROOT)).extractorKey).toBe('aliases@3');

        // '' is an explicit "no key", also over a keyed plan: nothing is stamped or recorded.
        const cleared = fresh();
        await review(cleared, 'aliases@1');
        const keyedPlan = await planDetail(cleared.source, cleared.manifest, [DOCS_PACK], { extractorKey: 'aliases@2' });
        await review(cleared, '', { plan: keyedPlan });
        expect((await loadExtractorState(cleared.manifest, ROOT)).extractorKey).toBeUndefined();
        expect('extractorKey' in ((await storedOf(cleared.store, 'a.md')) ?? {})).toBe(false);
    });

    describe('a host extraction pool', () => {
        /** A pool whose workers were built under `workerKey`; records the jobs it was handed. */
        const poolUnder = (workerKey: string | undefined, jobs: ExtractJob[]): ExtractPool => async (job) => {
            jobs.push(job);
            return extractJob(job, [DOCS_PACK], workerKey);
        };

        it('regression: the job carries the requested key and the extract is stamped from the worker\'s matching echo', async () => {
            const ctx = fresh();
            const jobs: ExtractJob[] = [];
            expect(await review(ctx, 'aliases@1', { extractPool: poolUnder('aliases@1', jobs) })).toEqual({ 'a.md': 'ok', 'b.md': 'ok', 'c.md': 'ok' });
            expect(jobs.every((j) => j.extractorKey === 'aliases@1')).toBe(true);
            expect((await storedOf(ctx.store, 'a.md'))?.extractorKey).toBe('aliases@1');
            expect(await plan(ctx, 'aliases@1')).toEqual(NONE_OWED);
        });

        it('regression: a worker still holding old aliases (another echoed key) is neither cached nor used: the files stay owed', async () => {
            const ctx = fresh();
            await review(ctx, 'aliases@1');
            const steps: ReviewStep[] = [];
            await review(ctx, 'aliases@2', { extractPool: poolUnder('aliases@1', []), onStep: (s) => steps.push(s) });
            const extracts = steps.filter((s) => s.phase === 'extract');
            expect(extracts.map((s) => [s.outcome, s.reason])).toEqual(Array(3).fill(['failed', EXTRACTOR_KEY_MISMATCH]));
            expect((await storedOf(ctx.store, 'a.md'))?.extractorKey).toBe('aliases@1'); // the old entry, not one stamped aliases@2
            expect(await plan(ctx, 'aliases@2')).toEqual(ALL_OWED);
            // The pool rebuilt for the new key: the files extract and settle.
            await review(ctx, 'aliases@2', { extractPool: poolUnder('aliases@2', []) });
            expect(await plan(ctx, 'aliases@2')).toEqual(NONE_OWED);
        });

        it('a worker that echoes no key with no key requested: as before keys existed', async () => {
            const ctx = fresh();
            expect(await review(ctx, undefined, { extractPool: poolUnder(undefined, []) })).toEqual({ 'a.md': 'ok', 'b.md': 'ok', 'c.md': 'ok' });
            expect('extractorKey' in ((await storedOf(ctx.store, 'a.md')) ?? {})).toBe(false);
        });
    });

    it('regression: an extract that came through a BFKP pack arrives keyless and is re-extracted under a key, not reused', async () => {
        const ctx = fresh();
        await review(ctx, 'aliases@1');
        const hash = hashText(FILES['a.md']);
        const keyed = (await ctx.store.getExtracted(hash))!;
        expect(keyed[0]!.extractorKey).toBe('aliases@1');
        const arrived = await decodePack(await encodePack({
            rootId: ROOT, baseGeneration: 0, targetDigest: ZERO_DIGEST, extracted: [{ kind: 'full', hash, entries: keyed }],
        }));
        const entries = (arrived.extracted![0] as { entries: Extracted[] }).entries;
        expect('extractorKey' in entries[0]!).toBe(false); // the pack does not carry the key

        const replica = memStore();
        await replica.putExtracted(hash, entries);
        const one = await extractFileDetailed({
            rootId: ROOT, path: 'a.md', hash, doc: await ctx.source.read('a.md'), packs: [DOCS_PACK], store: replica, extractorKey: 'aliases@1',
        });
        expect(one?.cached).toBe(false);
        expect(one?.extracted.extractorKey).toBe('aliases@1');
    });
});
