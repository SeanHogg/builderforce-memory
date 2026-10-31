/**
 * src/knowledge/stores/localSnapshotStore.batched.test.ts
 * `batched`: persistence is held inside a batch (one write at the end, a checkpoint at most
 * once per `checkpointMs`), `flush` persists once when dirty, the final flush always runs and
 * the right error surfaces, and outside a batch every mutation still persists on its own.
 */

import type { SnapshotPersistence } from './localSnapshotStore.js';
import { LocalSnapshotStore } from './localSnapshotStore.js';

interface Counting extends SnapshotPersistence { writes: number; failWrites: boolean; value?: string }

function counting(): Counting {
    const p: Counting = {
        writes: 0, failWrites: false,
        async read() { return p.value; },
        async write(s: string) {
            if (p.failWrites) throw new Error('disk full');
            p.writes += 1;
            p.value = s;
        },
    };
    return p;
}

const SCOPE = { rootId: 'r', title: 'R', packs: ['docs'] };

function setup(checkpointMs = 30_000) {
    const persistence = counting();
    const clock = { ms: 0 };
    const store = new LocalSnapshotStore(persistence, { now: () => 'T', checkpointMs, nowMs: () => clock.ms });
    return { persistence, clock, store };
}

describe('LocalSnapshotStore.batched', () => {
    it('outside a batch each mutation persists once, before it resolves', async () => {
        const { persistence, store } = setup();
        await store.putExtracted('h1', []);
        expect(persistence.writes).toBe(1);
        await store.markStale({ rootId: 'r' }, 'src:r::a.md', 'h');
        expect(persistence.writes).toBe(2);
        await store.write({ scope: SCOPE, facts: [], relations: [] });
        expect(persistence.writes).toBe(3);
    });

    it('holds N mutations in memory and writes once when the batch ends', async () => {
        const { persistence, store } = setup();
        await store.batched(async () => {
            for (let i = 0; i < 25; i++) await store.putExtracted(`h${i}`, []);
            expect(persistence.writes).toBe(0);
            expect(await store.getExtracted('h24')).toEqual([]);
        });
        expect(persistence.writes).toBe(1);
        expect(Object.keys(JSON.parse(persistence.value!).extracted)).toHaveLength(25);
    });

    it('writes nothing at the end of a batch that changed nothing', async () => {
        const { persistence, store } = setup();
        await store.batched(async () => { await store.facts({ rootId: 'r' }); });
        expect(persistence.writes).toBe(0);
    });

    it('checkpoints once a mutation lands past checkpointMs, not before', async () => {
        const { persistence, clock, store } = setup(1000);
        await store.batched(async () => {
            clock.ms = 400;
            await store.putExtracted('a', []);
            clock.ms = 999;
            await store.putExtracted('b', []);
            expect(persistence.writes).toBe(0);
            clock.ms = 1000;
            await store.putExtracted('c', []);
            expect(persistence.writes).toBe(1);
            clock.ms = 1500;
            await store.putExtracted('d', []);
            expect(persistence.writes).toBe(1);
            expect(Object.keys(JSON.parse(persistence.value!).extracted)).toEqual(['a', 'b', 'c']);
        });
        expect(persistence.writes).toBe(2);
        expect(Object.keys(JSON.parse(persistence.value!).extracted)).toHaveLength(4);
    });

    it('flush persists once when dirty and is a no-op when clean', async () => {
        const { persistence, store } = setup();
        await store.batched(async (flush) => {
            await flush();
            expect(persistence.writes).toBe(0);
            await store.putExtracted('a', []);
            await store.putExtracted('b', []);
            await flush();
            expect(persistence.writes).toBe(1);
            await flush();
            expect(persistence.writes).toBe(1);
            await store.putExtracted('c', []);
        });
        expect(persistence.writes).toBe(2);
    });

    it('persists what a rejecting work wrote and surfaces the work error', async () => {
        const { persistence, store } = setup();
        await expect(store.batched(async () => {
            await store.putExtracted('a', []);
            throw new Error('work failed');
        })).rejects.toThrow('work failed');
        expect(persistence.writes).toBe(1);
        expect(JSON.parse(persistence.value!).extracted).toHaveProperty('a');
    });

    it('surfaces the work error even when the final flush also fails', async () => {
        const { persistence, store } = setup();
        await expect(store.batched(async () => {
            await store.putExtracted('a', []);
            persistence.failWrites = true;
            throw new Error('work failed');
        })).rejects.toThrow('work failed');
    });

    it('rejects with the flush error when work succeeded and the final flush fails', async () => {
        const { persistence, store } = setup();
        await expect(store.batched(async () => {
            await store.putExtracted('a', []);
            persistence.failWrites = true;
        })).rejects.toThrow('disk full');
    });

    it('a failed checkpoint rejects the mutation that triggered it', async () => {
        const { persistence, clock, store } = setup(1000);
        await store.batched(async () => {
            await store.putExtracted('a', []);
            persistence.failWrites = true;
            clock.ms = 2000;
            await expect(store.putExtracted('b', [])).rejects.toThrow('disk full');
            persistence.failWrites = false;
        });
        expect(JSON.parse(persistence.value!).extracted).toHaveProperty('b');
    });

    it('nested and overlapping batches flush once, at the outermost end', async () => {
        const { persistence, store } = setup();
        await store.batched(async () => {
            await store.putExtracted('a', []);
            await store.batched(async () => { await store.putExtracted('b', []); });
            expect(persistence.writes).toBe(0);
        });
        expect(persistence.writes).toBe(1);

        let release!: () => void;
        const gate = new Promise<void>((r) => { release = r; });
        const first = store.batched(async () => { await store.putExtracted('c', []); await gate; });
        const second = store.batched(async () => { await store.putExtracted('d', []); });
        await second;
        expect(persistence.writes).toBe(1);
        release();
        await first;
        expect(persistence.writes).toBe(2);
        expect(Object.keys(JSON.parse(persistence.value!).extracted)).toHaveLength(4);
    });

    it('holds a plain mutation from another caller during a batch like the others', async () => {
        const { persistence, store } = setup();
        await store.batched(async () => {
            await store.putExtracted('mine', []);
            await store.write({ scope: SCOPE, facts: [], relations: [] }); // e.g. the file watcher
            expect(persistence.writes).toBe(0);
        });
        expect(persistence.writes).toBe(1);
    });

    it('persists each mutation again once the batch is over', async () => {
        const { persistence, store } = setup();
        await store.batched(async () => { await store.putExtracted('a', []); });
        await store.putExtracted('b', []);
        expect(persistence.writes).toBe(2);
    });
});
