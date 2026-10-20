/**
 * tests/storage-file-lock.test.ts — the cross-process lock's edge paths.
 *
 * tests/nodefile-lock.test.ts covers the main flows through the adapter; these
 * drive FileLock directly — on a real temp dir for the defaults and bounded
 * wait, and on a scripted fs for the races a real disk cannot be made to
 * produce on demand (a lock released or re-taken between two reads, unlink
 * failures).
 */

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { FileLock, type LockFs } from '../src/storage/fileLock.js';
import { MemoryStorageError } from '../src/storage/types.js';

let dir: string;
beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bf-filelock-')); });
afterEach(() => { fs.rmSync(dir, { recursive: true, force: true }); });

const fsError = (code: string) => Object.assign(new Error(`${code}: scripted`), { code });

/**
 * A scripted LockFs: `open` answers each create attempt in turn (an error code
 * to throw, or 'ok'), `read` each holder read (a PID string or an error code),
 * `unlink` each unlink ('ok' or an error code). The last entry repeats.
 */
function scriptedFs(script: { open: string[]; read?: string[]; unlink?: string[] }) {
    const calls = { open: 0, read: 0, unlink: 0, closed: 0, written: [] as string[] };
    const next = (list: string[], i: number) => list[Math.min(i, list.length - 1)]!;
    const lockFs: LockFs = {
        openSync: () => {
            const step = next(script.open, calls.open++);
            if (step !== 'ok') throw fsError(step);
            return 7;
        },
        writeFileSync: (_fd, data) => { calls.written.push(data); },
        closeSync: () => { calls.closed++; },
        readFileSync: () => {
            const step = next(script.read ?? ['1'], calls.read++);
            if (/^E[A-Z]+$/.test(step)) throw fsError(step);
            return step;
        },
        unlinkSync: () => {
            const step = next(script.unlink ?? ['ok'], calls.unlink++);
            if (step !== 'ok') throw fsError(step);
        },
    };
    return { lockFs, calls };
}

const noSleep = () => Promise.resolve();

describe('FileLock — defaults and the bounded wait', () => {
    it('with default options, writes this PID while held and removes the lock after', async () => {
        const lockPath = path.join(dir, 'default.lock');
        const lock = new FileLock(fs, lockPath);
        const held = await lock.run(() => fs.readFileSync(lockPath, 'utf8'));
        expect(held).toBe(String(process.pid));
        expect(fs.existsSync(lockPath)).toBe(false);
    });

    it('times out naming an unknown holder when the lock file holds no PID', async () => {
        const lockPath = path.join(dir, 'garbage.lock');
        fs.writeFileSync(lockPath, 'not-a-pid');
        const lock = new FileLock(fs, lockPath, { timeoutMs: 0, isAlive: () => false });
        const err = await lock.run(() => 'ran').then(() => null, (e: unknown) => e);
        expect(err).toBeInstanceOf(MemoryStorageError);
        expect(err).toMatchObject({ storageCode: 'STORAGE_UNAVAILABLE' });
        expect(String((err as Error).message)).toContain('held by pid unknown');
        // A lock whose holder cannot be read is never reclaimed.
        expect(fs.readFileSync(lockPath, 'utf8')).toBe('not-a-pid');
    });

    it('never reclaims a lock that names this process, even if the probe says dead', async () => {
        const lockPath = path.join(dir, 'self.lock');
        fs.writeFileSync(lockPath, '4242');
        const lock = new FileLock(fs, lockPath, { pid: 4242, timeoutMs: 20, retryMs: 5, isAlive: () => false });
        await expect(lock.run(() => 'ran')).rejects.toThrow(/held by pid 4242/);
        expect(fs.existsSync(lockPath)).toBe(true);
    });

    it('waits on a live holder, polling at the retry interval, until the deadline', async () => {
        const { lockFs, calls } = scriptedFs({ open: ['EEXIST'], read: ['31337'] });
        const sleeps: number[] = [];
        const lock = new FileLock(lockFs, '/x.lock', {
            timeoutMs: 30,
            retryMs: 7,
            isAlive: () => true,
            sleep: (ms) => { sleeps.push(ms); return new Promise((resolve) => setTimeout(resolve, ms)); },
        });
        await expect(lock.run(() => 'ran')).rejects.toThrow(/held by pid 31337/);
        expect(sleeps.length).toBeGreaterThan(0);
        expect(sleeps.every((ms) => ms === 7)).toBe(true);
        expect(calls.unlink).toBe(0); // a live holder's lock is left alone
    });
});

describe('FileLock — races and failures', () => {
    it('retries the create when the holder released between the failed create and the read', async () => {
        const { lockFs, calls } = scriptedFs({ open: ['EEXIST', 'ok'], read: ['ENOENT'] });
        const lock = new FileLock(lockFs, '/x.lock', { pid: 5, sleep: noSleep });
        expect(await lock.run(() => 'ran')).toBe('ran');
        expect(calls.open).toBe(2);
        expect(calls.written).toEqual(['5']);
        expect(calls.unlink).toBe(1); // the release only
    });

    it('surfaces a holder read that fails for any other reason', async () => {
        const { lockFs } = scriptedFs({ open: ['EEXIST'], read: ['EACCES'] });
        const lock = new FileLock(lockFs, '/x.lock', { sleep: noSleep });
        let ran = false;
        await expect(lock.run(() => { ran = true; })).rejects.toMatchObject({ code: 'EACCES' });
        expect(ran).toBe(false);
    });

    it('does not remove a stale lock that another process re-took before the reclaim', async () => {
        // First read: dead pid 111. Reclaim's re-read: live pid 222 took it — leave it.
        const { lockFs, calls } = scriptedFs({ open: ['EEXIST', 'ok'], read: ['111', '222'] });
        const lock = new FileLock(lockFs, '/x.lock', { pid: 5, isAlive: (pid) => pid !== 111, sleep: noSleep });
        expect(await lock.run(() => 'ran')).toBe('ran');
        expect(calls.read).toBe(2);
        expect(calls.unlink).toBe(1); // the release only, no reclaim unlink
    });

    it('reclaims a dead holder, tolerating another waiter removing it first', async () => {
        const { lockFs, calls } = scriptedFs({ open: ['EEXIST', 'ok'], read: ['111'], unlink: ['ENOENT', 'ok'] });
        const lock = new FileLock(lockFs, '/x.lock', { pid: 5, isAlive: () => false, sleep: noSleep });
        expect(await lock.run(() => 'ran')).toBe('ran');
        expect(calls.unlink).toBe(2); // the reclaim (already gone) and the release
    });

    it('surfaces a reclaim unlink that fails for any other reason', async () => {
        const { lockFs } = scriptedFs({ open: ['EEXIST'], read: ['111'], unlink: ['EPERM'] });
        const lock = new FileLock(lockFs, '/x.lock', { pid: 5, isAlive: () => false, sleep: noSleep });
        await expect(lock.run(() => 'ran')).rejects.toMatchObject({ code: 'EPERM' });
    });

    it('fails with STORAGE_UNAVAILABLE when the lock cannot be created at all', async () => {
        const { lockFs } = scriptedFs({ open: ['EACCES'] });
        const lock = new FileLock(lockFs, '/x.lock', { sleep: noSleep });
        const err = await lock.run(() => 'ran').then(() => null, (e: unknown) => e);
        expect(err).toBeInstanceOf(MemoryStorageError);
        expect(err).toMatchObject({ storageCode: 'STORAGE_UNAVAILABLE' });
        expect(String((err as Error).message)).toContain('Cannot create lock');
    });

    it('closes the lock file even when writing the PID fails', async () => {
        const { lockFs, calls } = scriptedFs({ open: ['ok'] });
        lockFs.writeFileSync = () => { throw fsError('ENOSPC'); };
        const lock = new FileLock(lockFs, '/x.lock', { sleep: noSleep });
        await expect(lock.run(() => 'ran')).rejects.toMatchObject({ code: 'ENOSPC' });
        expect(calls.closed).toBe(1);
    });

    it('releases after the unit of work throws, and keeps its error', async () => {
        const { lockFs, calls } = scriptedFs({ open: ['ok'] });
        const lock = new FileLock(lockFs, '/x.lock', { sleep: noSleep });
        await expect(lock.run(() => { throw new Error('work failed'); })).rejects.toThrow('work failed');
        expect(calls.unlink).toBe(1);
    });

    it('a lock already gone at release is fine; any other release failure is reported', async () => {
        const gone = scriptedFs({ open: ['ok'], unlink: ['ENOENT'] });
        expect(await new FileLock(gone.lockFs, '/x.lock', { sleep: noSleep }).run(() => 'ran')).toBe('ran');

        const stuck = scriptedFs({ open: ['ok'], unlink: ['EBUSY'] });
        const err = await new FileLock(stuck.lockFs, '/x.lock', { sleep: noSleep }).run(() => 'ran').then(() => null, (e: unknown) => e);
        expect(err).toBeInstanceOf(MemoryStorageError);
        expect(err).toMatchObject({ storageCode: 'STORAGE_UNAVAILABLE' });
        expect(String((err as Error).message)).toContain('Cannot release lock');
    });
});
