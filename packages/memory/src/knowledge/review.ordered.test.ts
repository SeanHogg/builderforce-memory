/**
 * src/knowledge/review.ordered.test.ts
 * PRD 41 A8: files are read, extracted and judged in parallel, but ONE writer applies them in plan
 * order — the snapshot bytes AND the manifests of `concurrency 8 / judgeConcurrency 4` (in-thread or
 * through an extraction pool) equal those of `1 / 1`, on a first review and on an incremental one, with
 * same-hash files in the fixture and model calls that settle out of order.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { extractJob, type ExtractJob, type ExtractPool } from './extractJob.js';
import type { JudgementPort } from './model.js';
import { DOCS_PACK } from './packs/docs.js';
import { reviewDirectory } from './review.js';
import { compareManifestKey, filesManifestKey, judgedManifestKey, manifestMap } from './sources.js';
import { LocalSnapshotStore } from './stores/localSnapshotStore.js';
import { args, fakeJudge, memPersistence, memSource, ROOT, similar } from './testkit.test-util.js';

const N = 34;
const SAME = '# Same\nidentical body\n';
const files = (): Record<string, string> => ({
    ...Object.fromEntries(Array.from({ length: N }, (_, i) => [`d${String(i).padStart(2, '0')}.md`, `# T${i}\nbody ${i} lease ends ${i}\n\n## S${i}\nmore ${i}\n`])),
    'same1.md': SAME, 'same2.md': SAME, 'same3.md': SAME,
});
const EDITED = ['d03.md', 'd10.md', 'd20.md'];

const tick = (ms: number) => new Promise((r) => setTimeout(r, ms));
const weight = (s: string) => [...s].reduce((a, c) => a + c.charCodeAt(0), 0);

/** Settles each call after a delay unrelated to plan order, so completion order differs from dispatch order. */
function scrambled(): JudgementPort {
    const inner = fakeJudge();
    return { async judge(req) { await tick(weight(req.items[0]!.ref) % 6); return inner.judge(req); } };
}

/** A pool that clones the job as a thread boundary would and settles out of order. */
const scrambledPool = (): ExtractPool => async (job: ExtractJob) => {
    await tick(weight(job.path) % 5);
    return extractJob(JSON.parse(JSON.stringify(job)) as ExtractJob, [DOCS_PACK]);
};

interface Shape { concurrency: number; judgeConcurrency: number; modelCalls: number; pool?: boolean }

async function runTwice(shape: Shape) {
    const persistence = memPersistence();
    const store = new LocalSnapshotStore(persistence, { now: () => '2026-10-28T00:00:00.000Z' });
    const manifest = new InMemoryIngestManifest();
    const source = memSource(files());
    const read = source.read.bind(source);
    let live = 0;
    let max = 0;
    source.read = async (path) => {
        live += 1; max = Math.max(max, live);
        const doc = await read(path);
        await tick(path.length % 3);
        live -= 1;
        return doc;
    };
    const normalize = (s: string) => s.replace(/"lastReviewedAt":\s*"[^"]*"/g, '"lastReviewedAt":"T"'); // the run's wall clock
    const go = async () => {
        const report = await reviewDirectory(args(source, store, manifest, {
            judgement: scrambled(), similar, concurrency: shape.concurrency, judgeConcurrency: shape.judgeConcurrency,
            budget: { modelCalls: shape.modelCalls, ms: 60_000 }, ...(shape.pool ? { extractPool: scrambledPool() } : {}),
        }));
        expect(report.aborted).toBe(false);
        expect(report.failures).toEqual([]);
        return {
            report, bytes: normalize(persistence.value ?? ''),
            manifests: JSON.stringify([
                [...(await manifestMap(manifest, filesManifestKey(ROOT)))],
                [...(await manifestMap(manifest, judgedManifestKey(ROOT)))],
                (await manifest.get(compareManifestKey(ROOT))) ?? [],
            ]),
        };
    };
    const first = await go();
    for (const p of EDITED) source.files[p] = `${source.files[p]}\nedited line\n`;
    const second = await go();
    return { first, second, inFlight: max };
}

describe('parallel review, ordered writer', () => {
    it('stores the same bytes and manifests at 8/4 as at 1/1, on a first and an incremental review', async () => {
        const wide = await runTwice({ concurrency: 8, judgeConcurrency: 4, modelCalls: 500 });
        const narrow = await runTwice({ concurrency: 1, judgeConcurrency: 1, modelCalls: 500 });
        expect(wide.inFlight).toBeGreaterThan(1);
        expect(narrow.inFlight).toBe(1);
        expect(wide.first.bytes.length).toBeGreaterThan(1000);
        expect(wide.first.bytes).toBe(narrow.first.bytes);
        expect(wide.first.manifests).toBe(narrow.first.manifests);
        expect(wide.second.bytes).not.toBe(wide.first.bytes);
        expect(wide.second.bytes).toBe(narrow.second.bytes);
        expect(wide.second.manifests).toBe(narrow.second.manifests);
    });

    it('stores the same bytes through an extraction pool at width as in-thread at 1/1', async () => {
        const pooled = await runTwice({ concurrency: 8, judgeConcurrency: 4, modelCalls: 500, pool: true });
        const narrow = await runTwice({ concurrency: 1, judgeConcurrency: 1, modelCalls: 500 });
        expect(pooled.first.bytes).toBe(narrow.first.bytes);
        expect(pooled.second.bytes).toBe(narrow.second.bytes);
        expect(pooled.second.manifests).toBe(narrow.second.manifests);
    });

    it('spends a short model-call budget in plan order whatever the widths', async () => {
        const wide = await runTwice({ concurrency: 8, judgeConcurrency: 4, modelCalls: 17 });
        const narrow = await runTwice({ concurrency: 1, judgeConcurrency: 1, modelCalls: 17 });
        expect(wide.first.report.judged).toBe(17);
        expect(wide.first.report.unjudged).toBeGreaterThan(0);
        expect({ judged: wide.first.report.judged, unjudged: wide.first.report.unjudged })
            .toEqual({ judged: narrow.first.report.judged, unjudged: narrow.first.report.unjudged });
        expect(wide.first.bytes).toBe(narrow.first.bytes);
        expect(wide.second.bytes).toBe(narrow.second.bytes);
        expect(wide.second.manifests).toBe(narrow.second.manifests);
    });
});
