/**
 * src/knowledge/cooperative.test.ts — the yield is chosen once, in order setImmediate > scheduler.yield > MessageChannel >
 * setTimeout, from functions captured at pick time (fake timers installed later never reach it).
 */

import { jest } from '@jest/globals';
import { cooperativeBudget, pickYield, yieldToEventLoop, type YieldHost } from './cooperative.js';

/** A MessageChannel stand-in that delivers on a real immediate, and counts posts. */
function fakeChannel(posts: { n: number }): YieldHost['MessageChannel'] {
    return class {
        port1: { onmessage: ((ev: unknown) => void) | null } = { onmessage: null };
        port2 = { postMessage: (_: unknown) => { posts.n++; setImmediate(() => this.port1.onmessage?.({})); } };
    };
}

describe('cooperative yield selection', () => {
    it('prefers setImmediate, then scheduler.yield, then MessageChannel, then setTimeout', () => {
        const timeout = (fn: () => void) => setTimeout(fn, 0);
        const posts = { n: 0 };
        const all: YieldHost = { setImmediate, scheduler: { yield: async () => undefined }, MessageChannel: fakeChannel(posts), setTimeout: timeout };
        expect(pickYield(all).kind).toBe('setImmediate');
        expect(pickYield({ ...all, setImmediate: undefined }).kind).toBe('scheduler.yield');
        expect(pickYield({ ...all, setImmediate: undefined, scheduler: {} }).kind).toBe('MessageChannel');
        expect(pickYield({ setTimeout: timeout }).kind).toBe('setTimeout');
    });

    it('the MessageChannel fallback resolves each yield, in order', async () => {
        const posts = { n: 0 };
        const { yield: y } = pickYield({ MessageChannel: fakeChannel(posts) });
        const order: number[] = [];
        await Promise.all([y().then(() => order.push(1)), y().then(() => order.push(2))]);
        expect(order).toEqual([1, 2]);
        expect(posts.n).toBe(2);
    });

    it('the setTimeout fallback is captured at pick time: replacing the host timer afterwards does not reach it', async () => {
        let used = 0;
        const host: YieldHost = { setTimeout: (fn) => { used++; return setTimeout(fn, 0); } };
        const { yield: y } = pickYield(host);
        host.setTimeout = () => { throw new Error('a later (fake) timer was used'); };
        await y();
        expect(used).toBe(1);
    });

    it('fake timers installed after load do not stall the module yield or the budget', async () => {
        jest.useFakeTimers();
        try {
            await yieldToEventLoop();
            const budget = cooperativeBudget(0);
            expect(budget.due()).toBe(true);
            await budget.pause();
        } finally {
            jest.useRealTimers();
        }
    });
});
