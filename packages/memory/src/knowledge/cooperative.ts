/**
 * knowledge/cooperative.ts — THE cooperative-yield helper of long synchronous loops (the plan's walk, the fact store's load).
 *
 * A host may run a review on its UI thread (the VS Code extension host does): a loop over tens of thousands of files must
 * hand the event loop back every few milliseconds or the host stalls. {@link cooperativeBudget} times the work since the
 * last yield and {@link yieldToEventLoop} returns to the event loop (a macrotask, so I/O and input run in between).
 *
 * Portable: `setImmediate`, `scheduler.yield`, a `MessageChannel` or `setTimeout(0)` ({@link pickYield}) and
 * `performance.now()` — Node, browsers and Workers alike. All are captured when the module loads, so a test's fake timers
 * never stall a yield.
 */

/** Default work budget between two yields, in ms. */
export const COOPERATIVE_BUDGET_MS = 12;

/** The host globals the yield may use. */
export interface YieldHost {
    setImmediate?: (fn: () => void) => unknown;
    scheduler?: { yield?: () => Promise<void> };
    MessageChannel?: new () => { port1: { onmessage: ((ev: unknown) => void) | null }; port2: { postMessage(msg: unknown): void } };
    setTimeout?: (fn: () => void, ms?: number) => unknown;
}

export type YieldKind = 'setImmediate' | 'scheduler.yield' | 'MessageChannel' | 'setTimeout';

/**
 * THE choice of how to return to the event loop, made ONCE (at module load for `globalThis`): `setImmediate` (Node), else
 * `scheduler.yield` (Chromium), else a `MessageChannel` message (browsers, Workers — a hidden tab throttles `setTimeout(0)` to
 * ≥ 1 s, a message is not throttled), else `setTimeout(0)`. Every function is captured here, so fake timers installed later
 * never reach it.
 */
export function pickYield(host: YieldHost): { kind: YieldKind; yield: () => Promise<void> } {
    const immediate = host.setImmediate;
    if (typeof immediate === 'function') return { kind: 'setImmediate', yield: () => new Promise((r) => { immediate.call(host, r); }) };
    const scheduler = host.scheduler;
    const schedulerYield = scheduler?.yield;
    if (typeof schedulerYield === 'function') return { kind: 'scheduler.yield', yield: () => schedulerYield.call(scheduler) };
    if (typeof host.MessageChannel === 'function') {
        const channel = new host.MessageChannel();
        const waiting: Array<() => void> = [];
        channel.port1.onmessage = () => { waiting.shift()?.(); };
        return { kind: 'MessageChannel', yield: () => new Promise((r) => { waiting.push(r); channel.port2.postMessage(0); }) };
    }
    const timeout = host.setTimeout;
    if (typeof timeout === 'function') return { kind: 'setTimeout', yield: () => new Promise((r) => { timeout.call(host, r, 0); }) };
    // No timer at all: a resolved promise still lets other microtasks run.
    return { kind: 'setTimeout', yield: () => Promise.resolve() };
}

const g = globalThis as YieldHost & { performance?: { now(): number } };
const picked = pickYield(g);
const clock: () => number = g.performance && typeof g.performance.now === 'function' ? g.performance.now.bind(g.performance) : Date.now;

/** Returns to the event loop once (one macrotask). */
export function yieldToEventLoop(): Promise<void> {
    return picked.yield();
}

export interface CooperativeBudget {
    /** Whether the work since the last yield exceeded the budget (cheap: one clock read). */
    due(): boolean;
    /** Yields to the event loop and restarts the budget. */
    pause(): Promise<void>;
}

/** A work budget: `if (budget.due()) await budget.pause();` inside a long loop. */
export function cooperativeBudget(budgetMs = COOPERATIVE_BUDGET_MS): CooperativeBudget {
    let since = clock();
    return {
        due: () => clock() - since >= budgetMs,
        pause: async () => {
            await yieldToEventLoop();
            since = clock();
        },
    };
}
