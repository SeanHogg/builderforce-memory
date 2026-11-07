/**
 * knowledge/modelLimiter.ts — ONE cap on the model calls in flight across every file of a review
 * (PRD 41 A8), so judging N files at once does not multiply the provider's load.
 *
 * AIMD: a rate-limit answer halves the width (minimum 1); every `growEvery` successful calls grow it
 * by 1, back up to the cap. Waiting calls start in order; one whose signal has aborted is rejected
 * instead of started, so nothing starts after an abort. A queued call is rejected AT the abort, through ONE
 * listener per distinct signal (not per waiter); a call already holding a slot is never rejected by it.
 */

export interface ModelLimiter {
    /** Calls allowed in flight right now (1..max). */
    readonly width: number;
    readonly max: number;
    run<T>(fn: () => Promise<T>, signal?: AbortSignal): Promise<T>;
}

/** An error that says the provider is rate limiting us (HTTP 429 or the usual wording). */
export function isRateLimit(err: unknown): boolean {
    const e = err as { status?: unknown; statusCode?: unknown; message?: unknown } | null;
    if (e?.status === 429 || e?.statusCode === 429) return true;
    return typeof e?.message === 'string' && /\b429\b|rate.?limit|too many requests/i.test(e.message);
}

interface Waiter { start(): void; reject(reason: unknown): void; signal: AbortSignal | undefined }

export function createModelLimiter(max: number, opts: { growEvery?: number } = {}): ModelLimiter {
    const cap = Math.max(1, Math.floor(max));
    const growEvery = Math.max(1, Math.floor(opts.growEvery ?? 20));
    let width = cap;
    let active = 0;
    let successes = 0;
    const waiting: Waiter[] = [];
    // ONE abort listener per distinct signal, however many calls are queued on it.
    const queuedBySignal = new Map<AbortSignal, { waiters: Set<Waiter>; onAbort: () => void }>();

    const unqueue = (w: Waiter): void => {
        const s = w.signal && queuedBySignal.get(w.signal);
        if (!s || !w.signal) return;
        s.waiters.delete(w);
        if (s.waiters.size === 0) { w.signal.removeEventListener('abort', s.onAbort); queuedBySignal.delete(w.signal); }
    };

    const pump = () => {
        while (active < width && waiting.length > 0) {
            const w = waiting.shift()!;
            unqueue(w);
            if (w.signal?.aborted) { w.reject(w.signal.reason ?? new Error('aborted')); continue; }
            active += 1;
            w.start();
        }
    };

    return {
        get width() { return width; },
        max: cap,
        run<T>(fn: () => Promise<T>, signal?: AbortSignal): Promise<T> {
            return new Promise<T>((resolve, reject) => {
                // A queued call whose signal aborts is rejected AT ONCE (not when its turn comes), so a stopped run drains its queue immediately.
                if (signal?.aborted) { reject(signal.reason ?? new Error('aborted')); return; }
                const w: Waiter = {
                    signal, reject,
                    start: () => {
                        (async () => fn())().then(
                            (v) => {
                                successes += 1;
                                if (successes >= growEvery && width < cap) { width += 1; successes = 0; }
                                resolve(v);
                            },
                            (err) => {
                                if (isRateLimit(err)) { width = Math.max(1, Math.floor(width / 2)); successes = 0; }
                                reject(err);
                            },
                        ).finally(() => { active -= 1; pump(); });
                    },
                };
                if (signal) {
                    let entry = queuedBySignal.get(signal);
                    if (!entry) {
                        const waiters = new Set<Waiter>();
                        const onAbort = (): void => {
                            queuedBySignal.delete(signal);
                            for (const q of waiters) {
                                const at = waiting.indexOf(q);
                                if (at >= 0) waiting.splice(at, 1);
                                q.reject(signal.reason ?? new Error('aborted'));
                            }
                            waiters.clear();
                        };
                        entry = { waiters, onAbort };
                        queuedBySignal.set(signal, entry);
                        signal.addEventListener('abort', onAbort, { once: true });
                    }
                    entry.waiters.add(w);
                }
                waiting.push(w);
                pump();
            });
        },
    };
}
