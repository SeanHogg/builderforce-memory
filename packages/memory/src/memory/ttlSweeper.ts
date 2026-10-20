/**
 * memory/ttlSweeper.ts – The background timer behind
 * `MemoryStore.startTtlSweeper`: TTL is otherwise evaluated lazily on read, so
 * an idle store would keep dead entries forever. (EVM-7)
 */

export class TtlSweeper {
    private _timer: ReturnType<typeof setInterval> | null = null;

    /** Runs `sweep` every `intervalMs`. Idempotent while running. */
    start(intervalMs: number, sweep: () => Promise<unknown>): void {
        if (this._timer) return;
        const timer = setInterval(() => { void sweep(); }, intervalMs);
        // Don't keep a Node process alive just for the sweeper.
        const maybeUnref = timer as unknown as { unref?: () => void };
        /* istanbul ignore else -- unref exists in Node; absent in the browser */
        if (typeof maybeUnref.unref === 'function') {
            maybeUnref.unref();
        }
        this._timer = timer;
    }

    stop(): void {
        if (this._timer) {
            clearInterval(this._timer);
            this._timer = null;
        }
    }
}
