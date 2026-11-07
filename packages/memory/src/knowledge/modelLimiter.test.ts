/**
 * src/knowledge/modelLimiter.test.ts
 * The review's ONE cap on model calls in flight (PRD 41 A8): bounded, AIMD on rate limits, and
 * nothing starts once the signal has aborted.
 */

import { jest } from '@jest/globals';
import { createModelLimiter, isRateLimit } from './modelLimiter.js';

const rateLimited = () => Object.assign(new Error('429 rate limited'), { status: 429 });
const ok = async () => 'ok';

describe('createModelLimiter', () => {
    it('never has more calls in flight than its width', async () => {
        const limiter = createModelLimiter(3);
        let live = 0;
        let max = 0;
        await Promise.all(Array.from({ length: 12 }, () => limiter.run(async () => {
            live += 1; max = Math.max(max, live);
            await new Promise((r) => setTimeout(r, 1));
            live -= 1;
        })));
        expect(max).toBe(3);
    });

    it('halves on a rate limit (minimum 1) and grows back by 1 per growEvery successes', async () => {
        const limiter = createModelLimiter(8, { growEvery: 3 });
        await expect(limiter.run(async () => { throw rateLimited(); })).rejects.toThrow('429');
        expect(limiter.width).toBe(4);
        await expect(limiter.run(async () => { throw rateLimited(); })).rejects.toThrow('429');
        expect(limiter.width).toBe(2);
        await expect(limiter.run(async () => { throw rateLimited(); })).rejects.toThrow('429');
        await expect(limiter.run(async () => { throw rateLimited(); })).rejects.toThrow('429');
        expect(limiter.width).toBe(1);

        for (let i = 0; i < 3; i++) await limiter.run(ok);
        expect(limiter.width).toBe(2);
        for (let i = 0; i < 3 * 10; i++) await limiter.run(ok);
        expect(limiter.width).toBe(8); // back at the cap, never above
    });

    it('a failure that is not a rate limit leaves the width alone', async () => {
        const limiter = createModelLimiter(4);
        await expect(limiter.run(async () => { throw new Error('boom'); })).rejects.toThrow('boom');
        expect(limiter.width).toBe(4);
    });

    it('rejects a waiting call whose signal aborted instead of starting it', async () => {
        const limiter = createModelLimiter(1);
        const controller = new AbortController();
        let started = 0;
        const first = limiter.run(async () => { started += 1; await new Promise((r) => setTimeout(r, 5)); });
        const waiting = limiter.run(async () => { started += 1; }, controller.signal);
        controller.abort();
        await expect(waiting).rejects.toBeDefined();
        await first;
        expect(started).toBe(1);
    });
});

describe('isRateLimit', () => {
    it('recognises a 429 status or the usual wording', () => {
        expect(isRateLimit({ status: 429 })).toBe(true);
        expect(isRateLimit(new Error('Rate limit exceeded'))).toBe(true);
        expect(isRateLimit(new Error('HTTP 429'))).toBe(true);
        expect(isRateLimit(new Error('401 unauthorized'))).toBe(false);
        expect(isRateLimit(null)).toBe(false);
    });
});

describe('createModelLimiter: abort of queued calls', () => {
    it('regression: a queued call rejects at the abort while slot holders still run, and holders finish normally', async () => {
        const limiter = createModelLimiter(2);
        const ctl = new AbortController();
        const release: Array<() => void> = [];
        const holders = [0, 1].map((i) => limiter.run(() => new Promise<string>((r) => { release.push(() => r(`held${i}`)); }), ctl.signal));
        const queued = limiter.run(ok, ctl.signal);
        const rejected = expect(queued).rejects.toThrow('stop');
        ctl.abort(new Error('stop'));
        await rejected;
        release.forEach((r) => r());
        await expect(Promise.all(holders)).resolves.toEqual(['held0', 'held1']);
        // Slots were released exactly once: a new call still runs.
        await expect(limiter.run(ok)).resolves.toBe('ok');
    });

    it('regression: 500 queued calls on one signal add exactly one abort listener', async () => {
        const limiter = createModelLimiter(1);
        const ctl = new AbortController();
        const add = jest.spyOn(ctl.signal, 'addEventListener');
        const remove = jest.spyOn(ctl.signal, 'removeEventListener');
        let free!: () => void;
        const holder = limiter.run(() => new Promise<string>((r) => { free = () => r('done'); }), ctl.signal);
        const queued = Array.from({ length: 500 }, () => limiter.run(ok, ctl.signal));
        const settled = queued.map((q) => q.then(() => 'ran', () => 'rejected'));
        // The slot holder's listener is dropped when it starts; the 500 waiters share ONE live listener.
        expect(add.mock.calls.length - remove.mock.calls.length).toBe(1);
        ctl.abort(new Error('stop'));
        expect((await Promise.all(settled)).every((s) => s === 'rejected')).toBe(true);
        free();
        await expect(holder).resolves.toBe('done');
    });
});
