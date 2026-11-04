/**
 * src/knowledge/orderedPool.test.ts
 * prefetchOrdered: input-order results, bounded in-flight work, abort / early-stop start nothing new.
 */

import { prefetchOrdered } from './orderedPool.js';

interface Deferred<T> { promise: Promise<T>; resolve(v: T): void; reject(e: unknown): void }
function deferred<T>(): Deferred<T> {
    let resolve!: (v: T) => void;
    let reject!: (e: unknown) => void;
    const promise = new Promise<T>((a, b) => { resolve = a; reject = b; });
    return { promise, resolve, reject };
}
const flush = async () => { for (let k = 0; k < 10; k++) await Promise.resolve(); };

describe('prefetchOrdered', () => {
    it('yields in input order even when later jobs finish first', async () => {
        const jobs = [0, 1, 2, 3].map(() => deferred<number>());
        const gen = prefetchOrdered([10, 11, 12, 13], 4, (_, i) => jobs[i]!.promise);
        const out: number[] = [];
        const consumed = (async () => { for await (const r of gen) out.push((r.result as PromiseFulfilledResult<number>).value); })();
        for (const i of [3, 2, 1, 0]) { jobs[i]!.resolve(i * 100); await flush(); }
        await consumed;
        expect(out).toEqual([0, 100, 200, 300]);
    });

    it('never has more than limit jobs in flight', async () => {
        let inFlight = 0;
        let max = 0;
        const items = Array.from({ length: 10 }, (_, i) => i);
        const pending: Array<Deferred<void>> = [];
        const gen = prefetchOrdered(items, 3, async () => {
            inFlight += 1; max = Math.max(max, inFlight);
            const d = deferred<void>(); pending.push(d);
            await d.promise;
            inFlight -= 1;
        });
        const consumed = (async () => { let n = 0; for await (const _ of gen) n += 1; return n; })();
        for (let k = 0; k < 10; k++) {
            await flush();
            pending[k]!.resolve();
        }
        expect(await consumed).toBe(10);
        expect(max).toBe(3);
    });

    it('stops starting work once the signal aborts', async () => {
        const controller = new AbortController();
        const started: number[] = [];
        const gen = prefetchOrdered([0, 1, 2, 3, 4, 5], 2, async (_, i) => { started.push(i); return i; }, controller.signal);
        const first = await gen.next();
        expect(first.value.i).toBe(0);
        controller.abort();
        const rest: number[] = [];
        for await (const r of gen) rest.push(r.i);
        // The stream ends at the boundary: read-ahead is discarded and nothing past the first window starts.
        expect(rest).toEqual([]);
        expect(Math.max(...started)).toBeLessThanOrEqual(1);
    });

    it('yields a rejected job as rejected without throwing', async () => {
        const boom = new Error('boom');
        const out: string[] = [];
        for await (const r of prefetchOrdered([1, 2, 3], 2, async (n) => { if (n === 2) throw boom; return n; })) {
            out.push(r.result.status);
            if (r.result.status === 'rejected') expect(r.result.reason).toBe(boom);
        }
        expect(out).toEqual(['fulfilled', 'rejected', 'fulfilled']);
    });

    it('with limit 1 starts job i+1 only when the consumer pulls it', async () => {
        const started: number[] = [];
        const gen = prefetchOrdered([0, 1, 2], 1, async (_, i) => { started.push(i); return i; });
        await gen.next();
        await flush();
        expect(started).toEqual([0]);
        await gen.next();
        await flush();
        expect(started).toEqual([0, 1]);
    });

    it('starts nothing new after an early return()', async () => {
        const started: number[] = [];
        const gen = prefetchOrdered([0, 1, 2, 3, 4, 5], 2, async (_, i) => { started.push(i); return i; });
        await gen.next();
        const before = started.length;
        await gen.return(undefined);
        await flush();
        expect(started.length).toBe(before);
    });

    it('settles every started job before return() resolves', async () => {
        const jobs = [0, 1, 2, 3].map(() => deferred<number>());
        const settled: number[] = [];
        const gen = prefetchOrdered([0, 1, 2, 3], 3, async (_, i) => { try { return await jobs[i]!.promise; } finally { settled.push(i); } });
        jobs[0]!.resolve(0);
        await gen.next();
        let returned = false;
        const ret = gen.return(undefined).then(() => { returned = true; });
        await flush();
        expect(returned).toBe(false);
        jobs[1]!.resolve(1); jobs[2]!.reject(new Error('x'));
        await ret;
        expect(settled.sort()).toEqual([0, 1, 2]);
    });
});
