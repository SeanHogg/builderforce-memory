/**
 * storage/fileLock.ts – Cross-process exclusive lock for the node-file adapter.
 *
 * Two MCP / extension-host processes sharing one memory file each ran
 * refresh → mutate → tmp+rename with nothing between them, so the later rename
 * silently dropped the other's commit. A read-write transaction now holds an
 * exclusive lock file for its whole refresh + mutate + persist:
 *
 *   - Acquire = `open(<file>.lock, 'wx')` (atomic create-if-absent), then write
 *     our PID into it.
 *   - A lock whose recorded PID is no longer alive is STALE and is reclaimed.
 *     Staleness is decided by PID liveness, never by age: a slow but live
 *     writer (a large fsync) must not have its lock stolen.
 *   - Waiting is bounded: past `timeoutMs` the transaction rejects with
 *     STORAGE_UNAVAILABLE naming the holder, instead of hanging forever.
 */

import { MemoryStorageError } from './types.js';

/** The subset of node:fs the lock needs. */
export interface LockFs {
    openSync(path: string, flags: string): number;
    writeFileSync(fd: number, data: string): void;
    closeSync(fd: number): void;
    readFileSync(path: string, encoding: 'utf8'): string;
    unlinkSync(path: string): void;
}

export interface FileLockOptions {
    /** Give up waiting after this long (ms). Default 10 000. */
    timeoutMs?: number;
    /** Poll interval while the lock is held by a live process (ms). Default 20. */
    retryMs?: number;
    /** This process's PID. Default: `process.pid`. */
    pid?: number;
    /** PID liveness probe. Default: `process.kill(pid, 0)`. */
    isAlive?: (pid: number) => boolean;
    /** Injectable sleep (tests). */
    sleep?: (ms: number) => Promise<void>;
}

interface NodeProcessLike {
    pid: number;
    kill(pid: number, signal: number): void;
}

function nodeProcess(): NodeProcessLike | undefined {
    return (globalThis as { process?: NodeProcessLike }).process;
}

function errorCode(err: unknown): string | undefined {
    return (err as { code?: string } | null)?.code;
}

/** `kill(pid, 0)`: ESRCH = gone; EPERM = alive under another user. */
export function defaultIsAlive(pid: number): boolean {
    const proc = nodeProcess();
    if (!proc) return true;
    try {
        proc.kill(pid, 0);
        return true;
    } catch (err) {
        return errorCode(err) === 'EPERM';
    }
}

function readHolder(fs: LockFs, lockPath: string): number | undefined {
    let text: string;
    try {
        text = fs.readFileSync(lockPath, 'utf8');
    } catch (err) {
        // Released between our failed create and this read: retry the create.
        if (errorCode(err) === 'ENOENT') return undefined;
        throw err;
    }
    const pid = Number.parseInt(text.trim(), 10);
    return Number.isInteger(pid) && pid > 0 ? pid : undefined;
}

export class FileLock {
    private readonly _timeoutMs: number;
    private readonly _retryMs: number;
    private readonly _pid: number;
    private readonly _isAlive: (pid: number) => boolean;
    private readonly _sleep: (ms: number) => Promise<void>;

    constructor(private readonly _fs: LockFs, readonly lockPath: string, opts: FileLockOptions = {}) {
        this._timeoutMs = opts.timeoutMs ?? 10_000;
        this._retryMs = opts.retryMs ?? 20;
        this._pid = opts.pid ?? nodeProcess()?.pid ?? 0;
        this._isAlive = opts.isAlive ?? defaultIsAlive;
        this._sleep = opts.sleep ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    }

    /** Runs `fn` while holding the lock; always releases it. */
    async run<T>(fn: () => Promise<T> | T): Promise<T> {
        await this._acquire();
        try {
            return await fn();
        } finally {
            this._release();
        }
    }

    private _tryCreate(): boolean {
        let fd: number;
        try {
            fd = this._fs.openSync(this.lockPath, 'wx');
        } catch (err) {
            if (errorCode(err) === 'EEXIST') return false;
            throw new MemoryStorageError('STORAGE_UNAVAILABLE', `Cannot create lock "${this.lockPath}": ${String(err)}`, err);
        }
        try {
            this._fs.writeFileSync(fd, String(this._pid));
        } finally {
            this._fs.closeSync(fd);
        }
        return true;
    }

    private async _acquire(): Promise<void> {
        const deadline = Date.now() + this._timeoutMs;
        let holder: number | undefined;
        for (;;) {
            if (this._tryCreate()) return;
            holder = readHolder(this._fs, this.lockPath);
            if (holder !== undefined && holder !== this._pid && !this._isAlive(holder)) {
                this._reclaim(holder);
                continue;
            }
            // A live holder, or a lock whose PID is not written yet: wait.
            if (Date.now() >= deadline) {
                throw new MemoryStorageError(
                    'STORAGE_UNAVAILABLE',
                    `Timed out after ${this._timeoutMs}ms waiting for "${this.lockPath}" (held by pid ${holder ?? 'unknown'}).`,
                );
            }
            await this._sleep(this._retryMs);
        }
    }

    /** Removes a dead holder's lock — only if it still names that holder. */
    private _reclaim(deadPid: number): void {
        if (readHolder(this._fs, this.lockPath) !== deadPid) return;
        try {
            this._fs.unlinkSync(this.lockPath);
        } catch (err) {
            // Another waiter reclaimed it first; the next create decides.
            if (errorCode(err) !== 'ENOENT') throw err;
        }
    }

    private _release(): void {
        try {
            this._fs.unlinkSync(this.lockPath);
        } catch (err) {
            if (errorCode(err) !== 'ENOENT') {
                throw new MemoryStorageError('STORAGE_UNAVAILABLE', `Cannot release lock "${this.lockPath}": ${String(err)}`, err);
            }
        }
    }
}
