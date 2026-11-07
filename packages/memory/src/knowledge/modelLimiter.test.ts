/**
 * src/knowledge/modelLimiter.test.ts
 * The review's ONE cap on model calls in flight (PRD 41 A8): bounded, AIMD on rate limits, and
 * nothing starts once the signal has aborted.
 */

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
