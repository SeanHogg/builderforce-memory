/**
 * knowledge/jobDeadline.ts — a per-job ceiling for the extract path (PRD 41 A10 rule 2).
 *
 * `withDeadline(ms, work, parent)` runs `work(signal)` and rejects with a {@link JobTimeout} (message
 * `timeout`) when it has not settled in `ms`. `signal` is the job's OWN: it aborts on the deadline and when
 * `parent` aborts, so a pool terminates (recycles) the worker that holds the job. A job over its ceiling never
 * holds the run: the caller records a failed step and moves on. The work's late result is discarded.
 */

export class JobTimeout extends Error {
    constructor() { super('timeout'); this.name = 'JobTimeout'; }
}

export async function withDeadline<T>(ms: number, work: (signal: AbortSignal) => Promise<T>, parent?: AbortSignal): Promise<T> {
    const own = new AbortController();
    const relay = (): void => own.abort(parent?.reason);
    if (parent?.aborted) relay(); else parent?.addEventListener('abort', relay, { once: true });
    let timer: ReturnType<typeof setTimeout> | undefined;
    const deadline = new Promise<never>((_, reject) => {
        timer = setTimeout(() => { const e = new JobTimeout(); own.abort(e); reject(e); }, ms);
    });
    const running = (async () => work(own.signal))();
    // The deadline can win the race: the work’s late rejection (an abort it was told about) is not an unhandled one.
    running.catch(() => undefined);
    try {
        return await Promise.race([running, deadline]);
    } finally {
        clearTimeout(timer);
        parent?.removeEventListener('abort', relay);
    }
}
