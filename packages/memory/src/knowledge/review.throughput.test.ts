/**
 * src/knowledge/review.throughput.test.ts — PRD 41 A10 Accept: a 20,000-file folder is MAPPED by pass 1
 * (`mechanical`) without a model call, through a pool of width 4, in time that measures the pipeline's own overhead (60 s is the CI ceiling,
 * not the goal), and neither a slow snapshot checkpoint nor a 30,000-op replica queue with a slow durable write is waited on inside a file's step.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { extractJob, type ExtractPool } from './extractJob.js';
import { DOCS_PACK } from './packs/docs.js';
import type { KnowledgePack, KnowledgeStore, ReviewProgress } from './model.js';
import { reviewDirectory } from './review.js';
import { LocalSnapshotStore } from './stores/localSnapshotStore.js';
import { args, fakeJudge, memPersistence, memSource } from './testkit.test-util.js';

const TS_FILES = 19_800;
const MD_FILES = 200;
const TOTAL = TS_FILES + MD_FILES;
const CEILING_MS = 60_000;

/** A code-ish pack with no judgement: every `.ts` is one file node. */
const TS_PACK: KnowledgePack = {
    id: 'ts', parsers: [{ id: 'ts', accepts: (s) => s.id.endsWith('.ts'), parse: () => [] }],
};

function fixture(): Record<string, string> {
    const out: Record<string, string> = {};
    for (let i = 0; i < TS_FILES; i++) out[`src/m${Math.floor(i / 100)}/f${i}.ts`] = `export const v${i} = ${i};\n`;
    for (let i = 0; i < MD_FILES; i++) out[`docs/d${i}.md`] = `# D${i}\nbody ${i}\n`;
    return out;
}

/** A pool of width 4 that extracts in-process, one job per tick, and records how many jobs it held at once. */
function fakePool(): { pool: ExtractPool; peak: () => number; calls: () => number } {
    let busy = 0, peak = 0, calls = 0;
    const packs = [TS_PACK, DOCS_PACK];
    return {
        peak: () => peak, calls: () => calls,
        pool: async (job) => {
            busy += 1; calls += 1; peak = Math.max(peak, busy);
            await Promise.resolve();
            const out = extractJob(job, packs);
            busy -= 1;
            return out;
        },
    };
}

describe('pass 1 throughput (PRD 41 A10)', () => {
    it('maps 20,000 files with no model call through a pool of width 4 within the CI ceiling', async () => {
        const judge = fakeJudge();
        const { pool, peak, calls } = fakePool();
        const phases: string[] = [];
        const events: ReviewProgress[] = [];
        const t0 = Date.now();
        const report = await reviewDirectory(args(memSource(fixture()), new LocalSnapshotStore(memPersistence()), new InMemoryIngestManifest(), {
            packs: [TS_PACK, DOCS_PACK], judgement: judge, mechanical: true, extractPool: pool, concurrency: 4,
            onProgress: (p) => { events.push(p); if (phases[phases.length - 1] !== p.phase) phases.push(p.phase); },
        }));
        const elapsed = Date.now() - t0;

        expect(judge.requests).toHaveLength(0);
        expect(report.aborted).toBe(false);
        expect(report.failures).toEqual([]);
        expect(report.extracted).toBe(TOTAL);
        expect(calls()).toBe(TOTAL);
        expect(peak()).toBeLessThanOrEqual(4);
        expect(report.awaitingJudge).toHaveLength(MD_FILES); // the docs are judged later; code never is
        expect(report.awaitingJudgeCount).toBe(MD_FILES);
        // The map is complete: memory's last word is `done` after every file reached extract and write.
        expect(phases[phases.length - 1]).toBe('done');
        expect(events.filter((e) => e.phase === 'write')).toHaveLength(TOTAL);
        expect(elapsed).toBeLessThan(CEILING_MS);
    }, 120_000);

    it('does not wait for a snapshot checkpoint or a 30,000-op replica queue inside a file\'s step', async () => {
        let release!: () => void;
        const gate = new Promise<void>((r) => { release = r; });
        const durable = memPersistence();
        const write = durable.write;
        let started = 0;
        // Every durable write (the snapshot's checkpoints, the queue's saves) hangs until the run has applied its last file.
        durable.write = async (s) => { started += 1; await gate; await write(s); };
        const local = new LocalSnapshotStore(durable, { checkpointMs: 1 });

        // The replica queue as the A7 wrapper uses it: 30,000 ops already queued, a durable save that is slow, and an `enqueue` that only appends.
        const queue = { ops: Array.from({ length: 30_000 }, (_, i) => `op${i}`), saves: 0, enqueues: 0 };
        const save = async (): Promise<void> => { queue.saves += 1; await gate; };
        let saving: Promise<void> | undefined;
        const enqueue = (op: string): void => {
            queue.ops.push(op); queue.enqueues += 1;
            saving ??= save().finally(() => { saving = undefined; }); // at most one save in flight; never awaited by a step
        };
        const replicated: KnowledgeStore = {
            scopes: () => local.scopes(), facts: (s) => local.facts(s), relations: (s) => local.relations(s),
            getExtracted: (h) => local.getExtracted(h),
            batched: (work) => local.batched(work),
            async write(batch) { await local.write(batch); enqueue(`w:${batch.facts.length}`); },
            async markStale(scope, ref, hash) { await local.markStale(scope, ref, hash); enqueue(`s:${ref}`); },
            async putExtracted(hash, ex) { await local.putExtracted(hash, ex); enqueue(`x:${hash}`); },
        };

        const files = fixture();
        const small = Object.fromEntries(Object.entries(files).slice(0, 3_000));
        const { pool } = fakePool();
        let lastWrite = 0;
        const total = Object.keys(small).length;
        const report = await reviewDirectory(args(memSource(small), replicated, new InMemoryIngestManifest(), {
            packs: [TS_PACK, DOCS_PACK], judgement: fakeJudge(), mechanical: true, extractPool: pool, concurrency: 4,
            onProgress: (p) => {
                if (p.phase !== 'write') return;
                lastWrite = p.done + 1;
                if (lastWrite === total) release(); // had a step awaited a pending durable write, this would never be reached
            },
        }));

        expect(lastWrite).toBe(total);
        expect(report.extracted).toBe(total);
        expect(started).toBeGreaterThan(0); // checkpoints really were pending while files were applied
        expect(queue.enqueues).toBeGreaterThan(total);
        expect(queue.saves).toBeLessThanOrEqual(1 + Math.ceil(queue.enqueues / 500)); // saves are batched, not per op
    }, 120_000);
});
