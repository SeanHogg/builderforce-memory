/**
 * knowledge/jobDeadline.ts — a per-job ceiling for the extract path (PRD 41 A10 rule 2).
 *
 * `createDeadlines(parent)` returns `withDeadline(ms, work)`: it runs `work(signal)` and rejects with a {@link JobTimeout}
 * (message `timeout`) when it has not settled in `ms`. `signal` is the job's OWN: it aborts on the deadline and when
 * `parent` aborts, so a pool terminates (recycles) the worker that holds the job. A job over its ceiling never
 * holds the run: the caller records a failed step and moves on. A late result of the work is discarded.
 *
 * ONE listener sits on `parent` for the whole run (it aborts every job still in flight), so thousands of jobs, 32+ of them at once,
 * never stack listeners on the run's signal.
 */

export class JobTimeout extends Error {
    constructor() { super('timeout'); this.name = 'JobTimeout'; }
}

export type WithDeadline = <T>(ms: number, work: (signal: AbortSignal) => Promise<T>) => Promise<T>;

export function createDeadlines(parent?: AbortSignal): WithDeadline {
    const live = new Set<AbortController>();
    if (parent) {
        const stop = (): void => { for (const c of [...live]) c.abort(parent.reason); };
        if (parent.aborted) queueMicrotask(stop); else parent.addEventListener('abort', stop, { once: true });
    }
    return async <T>(ms: number, work: (signal: AbortSignal) => Promise<T>): Promise<T> => {
        const own = new AbortController();
        if (parent?.aborted) own.abort(parent.reason);
        live.add(own);
        let timer: ReturnType<typeof setTimeout> | undefined;
        const deadline = new Promise<never>((_, reject) => {
            timer = setTimeout(() => { const e = new JobTimeout(); own.abort(e); reject(e); }, ms);
            // A stop settles the job AT ONCE, even when the work (a pool that ignores its signal) never answers.
            // Already aborted (the parent stopped before this job began): no event will fire, so settle now.
            if (own.signal.aborted) reject(own.signal.reason ?? new Error('aborted'));
            else own.signal.addEventListener('abort', () => reject(own.signal.reason ?? new Error('aborted')), { once: true });
        });
        const running = (async () => work(own.signal))();
        // The deadline can win the race: a late rejection of the work (an abort it was told about) is not an unhandled one.
        running.catch(() => undefined);
        try {
            return await Promise.race([running, deadline]);
        } finally {
            clearTimeout(timer);
            live.delete(own);
        }
    };
}
