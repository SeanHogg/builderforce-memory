/**
 * src/knowledge/review.owed.test.ts
 * The extractor entry carries `{ version, owed }`: an upgrade run cut short, a file that
 * cannot be read, an abort, a removed or ignored path — whatever a run leaves unfinished is
 * owed to the next review / *Continue*, whatever its content does meanwhile.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import type { SourceDocument } from '../ingest/types.js';
import type { ReviewArgs } from './model.js';
import { loadExtractorState } from './owedFiles.js';
import { DOCS_PACK } from './packs/docs.js';
import { planDetail, planReview } from './plan.js';
import { reviewDirectory } from './review.js';
import {
    compareManifestKey, EXTRACTOR_VERSION, extractorManifestKey, filesManifestKey, judgedManifestKey, manifestMap, saveManifestMap,
} from './sources.js';
import { ALL, args, fakeJudge, FILES, memSource, memStore, ref, ROOT, similar, stateOf, type SpySource } from './testkit.test-util.js';

const CURRENT = EXTRACTOR_VERSION;
const Q_TEXT = '# Q\nquux\n';
const TEXT = 'Intro text.\n# Body\nunder the heading\n';
const PREAMBLE = `${ref('p.md')}#~preamble`;
/** p.md has a preamble section, q.md one section. */
const PQ = { 'p.md': TEXT, 'q.md': Q_TEXT };

/** A reviewed root, the extractor entry deleted: the next review is the upgrade run. */
async function reviewed(files: Record<string, string> = FILES, extra: Partial<ReviewArgs> = {}) {
    const source = memSource({ ...files });
    const store = memStore();
    const manifest = new InMemoryIngestManifest();
    await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), ...extra }));
    await manifest.delete(extractorManifestKey(ROOT));
    return { source, store, manifest };
}

/** `path` throws on read, as a permission error would; returns the original `read`. */
function unreadable(source: SpySource, path: string) {
    const read = source.read.bind(source);
    source.read = async (p) => {
        if (p !== path) return read(p);
        source.reads.push(p);
        throw Object.assign(new Error('permission denied'), { code: 'EACCES' });
    };
    return read;
}

/** What the upgraded extractor adds to p.md: a preamble section nobody has judged. */
async function forgetPreamble(manifest: InMemoryIngestManifest) {
    const judged = await manifestMap(manifest, judgedManifestKey(ROOT));
    expect(judged.delete(PREAMBLE)).toBe(true);
    await saveManifestMap(manifest, judgedManifestKey(ROOT), judged);
}

/**
 * The upgrade has run with the budget spent: p.md's preamble section, new to the extractor, was not
 * judged. The first review had a `similar` port, so no claims are left awaiting a compare.
 */
async function upgraded() {
    const { source, store, manifest } = await reviewed(PQ, { similar });
    await forgetPreamble(manifest);
    const detail = await planDetail(source, manifest, [DOCS_PACK]);
    expect(detail.files.map((f) => [f.path, f.changed, f.owed])).toEqual([['p.md', true, true], ['q.md', true, true]]);
    const cut = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), similar, budget: { modelCalls: 0, ms: 60_000 } }));
    expect((await manifestMap(manifest, compareManifestKey(ROOT))).size).toBe(0);
    expect(cut.unjudged).toBe(1);
    expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['p.md'] });
    source.reads.length = 0;
    return { source, store, manifest, detail };
}

describe('an upgrade run cut by the budget on an unchanged file', () => {
    it('records the file with the calls it still costs', async () => {
        const { manifest } = await upgraded();
        expect(await manifest.get(extractorManifestKey(ROOT))).toEqual([
            { id: 'version', hash: String(CURRENT) }, { id: 'owed:p.md', hash: '1' },
        ]);
    });

    it('the next review judges the new section, once', async () => {
        const { source, store, manifest } = await upgraded();
        const judge = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: judge }));
        expect(judge.requests.map((r) => r.items.map((i) => i.ref))).toEqual([[PREAMBLE]]);
        expect(source.reads).toEqual(['p.md']);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });

        const third = fakeJudge();
        source.reads.length = 0;
        await reviewDirectory(args(source, store, manifest, { judgement: third }));
        expect(third.requests).toHaveLength(0);
        expect(source.reads).toEqual([]);
    });

    it('Continue with the same plan picks the owed file up without a re-walk, once', async () => {
        const { source, store, manifest, detail } = await upgraded();
        const judge = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: judge, plan: detail }));
        expect(judge.requests.map((r) => r.items.map((i) => i.ref))).toEqual([[PREAMBLE]]);
        expect(source.reads).toEqual(['p.md']);

        const third = fakeJudge();
        source.reads.length = 0;
        await reviewDirectory(args(source, store, manifest, { judgement: third, plan: detail }));
        expect(third.requests).toHaveLength(0);
        expect(source.reads).toEqual([]);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
    });

    it('the estimate covers the section and the compare its claim may reach, so it is never below what is spent', async () => {
        const { source, store, manifest } = await upgraded();
        expect((await manifestMap(manifest, compareManifestKey(ROOT))).size).toBe(0); // nothing awaiting: the term below is the owed section's
        const sections = (await planReview(source, manifest, [DOCS_PACK], { compare: false })).estimatedModelCalls;
        const estimate = (await planReview(source, manifest, [DOCS_PACK])).estimatedModelCalls;
        expect(sections).toBe(1);
        expect(estimate - sections).toBe(1); // ceil(1 owed section / 10 per compare call)

        const judge = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: judge, similar }));
        expect(judge.requests.map((r) => r.kind)).toEqual(['extract']); // a lone new claim has nothing to be compared with
        expect(estimate).toBeGreaterThanOrEqual(judge.requests.length);
    });

    it('stays owed through a forced run that is cut short, which does not widen it', async () => {
        const { source, store, manifest } = await upgraded();
        const controller = new AbortController();
        controller.abort();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), force: true, signal: controller.signal }));
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['p.md'] });
        expect((await loadExtractorState(manifest, ROOT)).owed.get('p.md')).toBe(1);
        const forced = await planDetail(source, manifest, [DOCS_PACK], { force: true });
        expect(forced.files.map((f) => [f.path, f.changed, f.owed])).toEqual([['p.md', true, true], ['q.md', true, false]]);
    });
});

describe('a file that stays unreadable', () => {
    it('is the only file the next review reads, and the only one owed — at the one call an upgrade owes it', async () => {
        const { source, store, manifest } = await reviewed(PQ, { similar });
        await forgetPreamble(manifest); // the upgrade adds a section to p.md
        const read = unreadable(source, 'p.md');
        source.reads.length = 0;
        const upgrade = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(upgrade).toMatchObject({ readErrors: 1, aborted: false });
        expect(source.reads).toEqual(['p.md', 'q.md']);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['p.md'] });
        expect((await loadExtractorState(manifest, ROOT)).owed.get('p.md')).toBe(1);
        const estimate = (await planReview(source, manifest, [DOCS_PACK], { compare: false })).estimatedModelCalls;
        expect(estimate).toBe(1);

        source.reads.length = 0;
        const next = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(next.readErrors).toBe(1);
        expect(source.reads).toEqual(['p.md']);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['p.md'] });

        source.read = read; // readable again: the new section is judged, within what the estimate promised
        source.reads.length = 0;
        const judge = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: judge }));
        expect(source.reads).toEqual(['p.md']);
        expect(judge.requests.map((r) => r.items.map((i) => i.ref))).toEqual([[PREAMBLE]]);
        expect(estimate).toBeGreaterThanOrEqual(judge.requests.length);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
    });

    it('keeps the count it was already owed at: a read error learns nothing about the file', async () => {
        const { source, store, manifest } = await reviewed(PQ, { similar });
        // p.md is owed at three calls, the version is today's (q.md owes nothing).
        await manifest.set(extractorManifestKey(ROOT), [{ id: 'version', hash: String(CURRENT) }, { id: 'owed:p.md', hash: '3' }]);
        unreadable(source, 'p.md');
        source.reads.length = 0;
        const report = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(report.readErrors).toBe(1);
        expect(source.reads).toEqual(['p.md']);
        expect((await loadExtractorState(manifest, ROOT)).owed.get('p.md')).toBe(3);
        expect((await planReview(source, manifest, [DOCS_PACK], { compare: false })).estimatedModelCalls).toBe(3);
    });
});

describe('what an upgrade run leaves owed', () => {
    it('an abort partway through the files: the finished ones are clear, the unreached ones owed', async () => {
        const { source, store, manifest } = await reviewed();
        const controller = new AbortController();
        const read = source.read.bind(source);
        source.read = async (p) => {
            const doc = await read(p);
            if (p === 'b.md') controller.abort(); // b.md is still handled; the loop stops before c.md
            return doc;
        };
        const report = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), signal: controller.signal }));
        expect(report.aborted).toBe(true);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['c.md'] });

        source.read = read;
        source.reads.length = 0;
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(source.reads).toEqual(['c.md']);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
    });

    it('a file found binary on read, or taken by no pack, is cleared', async () => {
        const { source, store, manifest } = await reviewed();
        const read = source.read.bind(source);
        source.read = async (p) => {
            const doc = await read(p);
            if (p === 'a.md') return { ...doc, content: { kind: 'text', text: 'bin\u0000ary' } };
            if (p === 'b.md') return { ...doc, content: { kind: 'text', text: 'x', mediaType: 'application/pdf' } };
            return doc;
        };
        const report = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(report.aborted).toBe(false);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
    });

    it('a file removed while owed drops out of a clean run', async () => {
        const { source, store, manifest } = await reviewed();
        delete source.files['b.md'];
        const report = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(report.aborted).toBe(false);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
        expect([...(await manifestMap(manifest, filesManifestKey(ROOT))).keys()]).toEqual(['a.md', 'c.md']);
    });

    it('a file removed while owed stays owed through an aborted run, and drops out of the next clean one', async () => {
        const { source, store, manifest } = await reviewed();
        delete source.files['b.md'];
        const controller = new AbortController();
        controller.abort();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), signal: controller.signal }));
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ALL });

        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
    });

    it('a path claimed by no pack stays owed while the manifest holds it', async () => {
        const { source, store, manifest } = await reviewed({ ...FILES, 'x.md': '# X\nxx\n' });
        const narrow = {
            ...DOCS_PACK, parsers: DOCS_PACK.parsers.map((p) => ({ ...p, accepts: (s: SourceDocument) => s.id !== 'x.md' && p.accepts(s) })),
        };
        const plan = await planReview(source, manifest, [narrow]);
        expect(plan.skipped).toEqual([{ path: 'x.md', reason: 'no-pack' }]);
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), packs: [narrow] }));
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['x.md'] });

        source.reads.length = 0;
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(source.reads).toEqual(['x.md']);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
    });

    it('Continue with a plan made before a path existed carries that owed path, unread', async () => {
        const source = memSource({ 'a.md': FILES['a.md'], 'b.md': FILES['b.md'] });
        const store = memStore();
        const manifest = new InMemoryIngestManifest();
        const stale = await planDetail(source, manifest, [DOCS_PACK]); // lists a.md and b.md only
        source.files['c.md'] = FILES['c.md'];
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        await manifest.delete(extractorManifestKey(ROOT)); // the upgrade: c.md is owed, though the plan never heard of it

        source.reads.length = 0;
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), plan: stale }));
        expect(source.reads).toEqual(['a.md', 'b.md']);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['c.md'] });

        source.reads.length = 0;
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(source.reads).toEqual(['c.md']);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
    });

    it('a file now ignored stays owed while the manifest holds it, and is reviewed when it is listed again', async () => {
        const { source, store, manifest } = await reviewed();
        source.files['.gitignore'] = 'b.md\n';
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['b.md'] });

        delete source.files['.gitignore'];
        source.reads.length = 0;
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(source.reads).toEqual(['b.md']);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
    });

    it('a content-changed file left unfinished stays owed when its bytes revert to the manifest hash', async () => {
        const { source, store, manifest } = await reviewed();
        source.files['c.md'] = '# C\ngamma\nmore\n';
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), budget: { modelCalls: 0, ms: 60_000 } }));
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['c.md'] });

        source.files['c.md'] = FILES['c.md']; // the manifest holds this hash again
        expect((await planReview(source, manifest, [DOCS_PACK])).changed).toEqual(['c.md']);
        const report = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(report.extracted).toBe(1);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
    });

    it('the same with the version current: a content-changed file that was never owed is recorded, then re-reviewed after the revert', async () => {
        const { source, store, manifest } = await reviewed();
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() })); // settles the upgrade
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });

        source.files['c.md'] = '# C\ngamma\nmore\n';
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), budget: { modelCalls: 0, ms: 60_000 } }));
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['c.md'] });

        source.files['c.md'] = FILES['c.md'];
        expect((await planReview(source, manifest, [DOCS_PACK])).changed).toEqual(['c.md']);
        const report = await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() }));
        expect(report.extracted).toBe(1);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
    });

    it('records every section a content-changed file had to judge, not only those the budget cut', async () => {
        const { source, store, manifest } = await reviewed();
        source.files['c.md'] = '# C\ngamma two\n## D\ndelta\n'; // two sections to judge
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), budget: { modelCalls: 1, ms: 60_000 } }));
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['c.md'] });
        expect((await loadExtractorState(manifest, ROOT)).owed.get('c.md')).toBe(2); // one was judged, but the revert re-judges it

        source.files['c.md'] = FILES['c.md'];
        const estimate = (await planReview(source, manifest, [DOCS_PACK], { compare: false })).estimatedModelCalls;
        const judge = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: judge }));
        expect(estimate).toBeGreaterThanOrEqual(judge.requests.length);
        expect(judge.requests.length).toBeGreaterThan(0);
    });
});

describe('a debt carried through several cuts', () => {
    it('a content-changed file cut three times, then reverted, is estimated at no less than the revert spends', async () => {
        const original = '# C\ngamma\n## D\ndelta\n## E\nepsilon\n';
        const { source, store, manifest } = await reviewed({ ...FILES, 'c.md': original });
        await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge() })); // settles the upgrade
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });

        source.files['c.md'] = '# C\ngamma 2\n## D\ndelta 2\n## E\nepsilon 2\n'; // three sections to judge
        for (const modelCalls of [1, 1, 0]) {
            await reviewDirectory(args(source, store, manifest, { judgement: fakeJudge(), budget: { modelCalls, ms: 60_000 } }));
            expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: ['c.md'] });
        }

        source.files['c.md'] = original; // the manifest holds this hash again
        const estimate = (await planReview(source, manifest, [DOCS_PACK], { compare: false })).estimatedModelCalls;
        const judge = fakeJudge();
        await reviewDirectory(args(source, store, manifest, { judgement: judge }));
        expect(judge.requests.length).toBeGreaterThan(1); // the sections judged under the edit are re-judged
        expect(estimate).toBeGreaterThanOrEqual(judge.requests.length);
    });
});

describe('the stored extractor value', () => {
    it('an upgrade adds its one call to the debt a file already carries', async () => {
        const { source, manifest } = await reviewed({ 'p.md': TEXT }, { similar });
        await manifest.set(extractorManifestKey(ROOT), [{ id: 'version', hash: String(CURRENT - 1) }, { id: 'owed:p.md', hash: '5' }]);
        expect((await planReview(source, manifest, [DOCS_PACK], { compare: false })).estimatedModelCalls).toBe(5 + 1);
    });

    it('a missing, empty or older value is an upgrade; a legacy value of today is not', async () => {
        const source = memSource({ ...FILES });
        const manifest = new InMemoryIngestManifest();
        await reviewDirectory(args(source, memStore(), manifest, { judgement: fakeJudge() }));
        const key = extractorManifestKey(ROOT);
        const owedPaths = async () => (await planDetail(source, manifest, [DOCS_PACK])).files.filter((f) => f.owed).map((f) => f.path);
        expect(await owedPaths()).toEqual([]);

        await manifest.delete(key);
        expect(await owedPaths()).toEqual(ALL);
        await manifest.set(key, []);
        expect(await owedPaths()).toEqual(ALL);
        await manifest.set(key, [{ id: 'version', hash: String(CURRENT - 1) }]);
        expect(await owedPaths()).toEqual(ALL);

        // The shape an older build wrote parses as `{ version, owed: none }`.
        await manifest.set(key, [{ id: 'version', hash: String(CURRENT) }]);
        expect(await stateOf(manifest)).toEqual({ version: CURRENT, owed: [] });
        expect(await owedPaths()).toEqual([]);
    });
});
