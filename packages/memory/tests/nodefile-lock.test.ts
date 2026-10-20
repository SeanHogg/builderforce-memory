/**
 * tests/nodefile-lock.test.ts — the node-file adapter's cross-process write lock.
 *
 * Two adapters on one file stand in for two MCP processes: each holds its own
 * in-memory document and transaction queue, exactly as separate processes do.
 * Without the lock, refresh → mutate → tmp+rename let the later rename drop the
 * other's commit; with it, every write builds on the latest file.
 */

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { MemoryStore } from '../src/memory/MemoryStore.js';
import { FileLock } from '../src/storage/fileLock.js';
import { NodeFileStorageAdapter } from '../src/storage/nodeFileAdapter.js';
import { MemoryStorageError } from '../src/storage/types.js';

let dir: string;
beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bf-lock-')); });
afterEach(() => { fs.rmSync(dir, { recursive: true, force: true }); });

// fs/path injected: the adapter's own indirect import() is for bundlers, not jest's VM.
const open = (file: string, lock = {}) => NodeFileStorageAdapter.open(file, { watchIntervalMs: 0, lock, fs, path });

describe('node-file write lock', () => {
    it('two adapters on one file interleave writes without losing any', async () => {
        const file = path.join(dir, 'memory.json');
        const a = new MemoryStore({ storage: await open(file) });
        const b = new MemoryStore({ storage: await open(file) });
        const writes: Promise<void>[] = [];
        for (let i = 0; i < 20; i++) {
            writes.push(a.remember(`a${i}`, `from a ${i}`));
            writes.push(b.remember(`b${i}`, `from b ${i}`));
        }
        await Promise.all(writes);

        const fresh = new MemoryStore({ storage: await open(file) });
        const keys = (await fresh.recallAll()).map((e) => e.key);
        expect(keys).toHaveLength(40);
        expect(fs.existsSync(`${file}.lock`)).toBe(false);
        a.close(); b.close(); fresh.close();
    });

    it('reclaims a lock left by a dead process (PID liveness, not age)', async () => {
        const file = path.join(dir, 'memory.json');
        fs.writeFileSync(`${file}.lock`, '999999');
        const store = new MemoryStore({ storage: await open(file, { isAlive: (pid: number) => pid !== 999999 }) });
        await store.remember('k', 'v');
        expect((await store.recall('k'))?.content).toBe('v');
        expect(fs.existsSync(`${file}.lock`)).toBe(false);
        store.close();
    });

    it('waits for a live holder, bounded, then rejects naming it', async () => {
        const file = path.join(dir, 'memory.json');
        fs.writeFileSync(`${file}.lock`, '424242');
        const store = new MemoryStore({ storage: await open(file, { isAlive: () => true, timeoutMs: 60, retryMs: 5 }) });
        const err = await store.remember('k', 'v').then(() => null, (e: unknown) => e);
        expect(err).toBeInstanceOf(MemoryStorageError);
        expect(err).toMatchObject({ code: 'MEMORY_UNAVAILABLE', storageCode: 'STORAGE_UNAVAILABLE' });
        expect(String((err as Error).message)).toContain('424242');
        // The live holder's lock is untouched.
        expect(fs.readFileSync(`${file}.lock`, 'utf8')).toBe('424242');
        store.close();
    });

    it('proceeds as soon as a live holder releases', async () => {
        const lockPath = path.join(dir, 'x.lock');
        fs.writeFileSync(lockPath, '424242');
        const lock = new FileLock(fs, lockPath, { isAlive: () => true, retryMs: 5, timeoutMs: 5_000 });
        setTimeout(() => fs.unlinkSync(lockPath), 30);
        const order: string[] = [];
        await lock.run(() => { order.push('ran'); expect(fs.readFileSync(lockPath, 'utf8')).toBe(String(process.pid)); });
        expect(order).toEqual(['ran']);
        expect(fs.existsSync(lockPath)).toBe(false);
    });

    it('releases the lock when the unit of work throws, and rolls the document back', async () => {
        const file = path.join(dir, 'memory.json');
        const adapter = await open(file);
        await expect(adapter.transact(['facts'], 'readwrite', async (tx) => {
            await tx.put('facts', { id: 'half', key: 'half' });
            throw new Error('boom');
        })).rejects.toThrow('boom');
        expect(fs.existsSync(`${file}.lock`)).toBe(false);
        expect(await adapter.transact(['facts'], 'readonly', (tx) => tx.getAll('facts'))).toEqual([]);
        adapter.close();
    });
});
