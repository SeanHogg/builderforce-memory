/**
 * tests/storage-node-file-adapter.test.ts — the node-file adapter's boot,
 * recovery, freshness, rollback and watcher paths, against a real temp dir.
 *
 *   - boot: a missing directory is created; a corrupt or unreadable main file
 *     recovers from `<file>.tmp`, else is kept at `<file>.corrupt` and the store
 *     starts empty; a partial document gets its missing tables defaulted;
 *   - a write refreshes from disk first (another process's commit is built on;
 *     a deleted file is an empty document);
 *   - a persist that fails rolls the in-memory document back;
 *   - the polling watcher reports foreign revisions and stops on unsubscribe
 *     and on close.
 */

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { NodeFileStorageAdapter, type NodeFs, type NodeFileStorageAdapterOptions } from '../src/storage/nodeFileAdapter.js';
import { MemoryStorageError, type ConsistencyNote, type MetaCounters, type StoredFactRecord } from '../src/storage/types.js';

let dir: string;
beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bf-nodefile-')); });
afterEach(() => { fs.rmSync(dir, { recursive: true, force: true }); });

// fs/path injected: the adapter's own indirect import() is for bundlers, not jest's VM.
const open = (file: string, opts: NodeFileStorageAdapterOptions = {}) =>
    NodeFileStorageAdapter.open(file, { watchIntervalMs: 0, fs, path, ...opts });

/** node:fs with some methods replaced. */
function fsWith(overrides: Partial<NodeFs>): NodeFs {
    const real: NodeFs = {
        existsSync: (p) => fs.existsSync(p),
        readFileSync: (p, e) => fs.readFileSync(p, e),
        statSync: (p) => fs.statSync(p),
        mkdirSync: (p, o) => { fs.mkdirSync(p, o); },
        openSync: (p, f) => fs.openSync(p, f),
        writeFileSync: (fd, d) => fs.writeFileSync(fd, d),
        fsyncSync: (fd) => fs.fsyncSync(fd),
        closeSync: (fd) => fs.closeSync(fd),
        renameSync: (a, b) => fs.renameSync(a, b),
        unlinkSync: (p) => fs.unlinkSync(p),
        copyFileSync: (a, b) => fs.copyFileSync(a, b),
    };
    return { ...real, ...overrides };
}

function fact(id: string): StoredFactRecord {
    return { id, key: id, scope: {}, content: `content of ${id}`, timestamp: 1, seq: 1, revision: 1, opId: `op-${id}`, pendingSync: false };
}

const facts = (adapter: NodeFileStorageAdapter) =>
    adapter.transact(['facts'], 'readonly', (tx) => tx.getAll<StoredFactRecord>('facts'));

const putFact = (adapter: NodeFileStorageAdapter, id: string) =>
    adapter.transact(['facts'], 'readwrite', (tx) => tx.put('facts', fact(id)));

const counters = (revision: number): MetaCounters => ({ key: 'counters', seq: 0, revision, opSeq: 0 });

async function waitFor(condition: () => boolean, timeoutMs = 2_000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (!condition()) {
        if (Date.now() > deadline) throw new Error('condition not met in time');
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
}

const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

describe('node-file adapter — boot and recovery', () => {
    it('creates a missing directory and reports its identity', async () => {
        const file = path.join(dir, 'nested', 'deeper', 'memory.json');
        const adapter = await open(file, { contextId: 'ctx-test' });
        expect(fs.existsSync(path.dirname(file))).toBe(true);
        expect(adapter.contextId).toBe('ctx-test');
        expect(adapter.filePath).toBe(file);
        expect(adapter.kind).toBe('node-file');
        await putFact(adapter, 'a');
        expect(JSON.parse(fs.readFileSync(file, 'utf8')).facts).toHaveLength(1);
        adapter.close();
    });

    it('skips the directory check when the path has no directory part', async () => {
        const file = path.join(dir, 'memory.json');
        const mkdirs: string[] = [];
        const adapter = await NodeFileStorageAdapter.open(file, {
            watchIntervalMs: 0,
            fs: fsWith({ mkdirSync: (p) => { mkdirs.push(p); } }),
            path: { dirname: () => '' },
        });
        expect(mkdirs).toEqual([]);
        expect(adapter.contextId).toMatch(/^node-/);
        expect(await facts(adapter)).toEqual([]);
        adapter.close();
    });

    it('recovers a corrupt main file from the last fsynced tmp file', async () => {
        const file = path.join(dir, 'memory.json');
        fs.writeFileSync(file, '{ torn');
        fs.writeFileSync(`${file}.tmp`, JSON.stringify({ formatVersion: 1, facts: [fact('saved')] }));
        const adapter = await open(file);
        expect((await facts(adapter)).map((f) => f.id)).toEqual(['saved']);
        expect(fs.existsSync(`${file}.corrupt`)).toBe(false);
        adapter.close();
    });

    it.each([
        ['torn JSON', '{ torn'],
        ['a JSON null', 'null'],
        ['a JSON number', '42'],
        ['an unknown format version', JSON.stringify({ formatVersion: 2, facts: [] })],
        ['facts that are not a list', JSON.stringify({ formatVersion: 1, facts: {} })],
    ])('keeps %s at <file>.corrupt and starts empty when there is no tmp file', async (_label, content) => {
        const file = path.join(dir, 'memory.json');
        fs.writeFileSync(file, content);
        const adapter = await open(file);
        expect(await facts(adapter)).toEqual([]);
        expect(fs.readFileSync(`${file}.corrupt`, 'utf8')).toBe(content);
        expect(fs.existsSync(file)).toBe(false);
        adapter.close();
    });

    it('defaults the tables and counters a partial document leaves out', async () => {
        const file = path.join(dir, 'memory.json');
        fs.writeFileSync(file, JSON.stringify({ formatVersion: 1, facts: [fact('a')], tombstones: 'x', meta: { key: 'wrong', revision: 9 } }));
        const adapter = await open(file);
        const read = await adapter.transact(['facts', 'tombstones', 'journal', 'meta'], 'readonly', async (tx) => ({
            facts: await tx.getAll<StoredFactRecord>('facts'),
            tombstones: await tx.getAll('tombstones'),
            journal: await tx.getAll('journal'),
            meta: await tx.get<MetaCounters>('meta', 'counters'),
        }));
        expect(read.facts.map((f) => f.id)).toEqual(['a']);
        expect(read.tombstones).toEqual([]);
        expect(read.journal).toEqual([]);
        expect(read.meta).toEqual(counters(0));
        adapter.close();
    });

    it('keeps valid counters, tombstones and journal from disk', async () => {
        const file = path.join(dir, 'memory.json');
        const journal = [{ opId: 'c:1', kind: 'put', id: 'a', key: 'a', scope: {}, revision: 3, timestamp: 1 }];
        fs.writeFileSync(file, JSON.stringify({ formatVersion: 1, facts: [], tombstones: [], journal, meta: counters(3) }));
        const adapter = await open(file);
        const read = await adapter.transact(['journal', 'meta'], 'readonly', async (tx) => ({
            journal: await tx.getAll('journal'),
            meta: await tx.get<MetaCounters>('meta', 'counters'),
        }));
        expect(read.journal).toEqual(journal);
        expect(read.meta).toEqual(counters(3));
        adapter.close();
    });
});

describe('node-file adapter — freshness before a write', () => {
    it("builds on another process's commit instead of overwriting it", async () => {
        const file = path.join(dir, 'memory.json');
        const a = await open(file);
        const b = await open(file);
        await putFact(a, 'from-a');
        await putFact(b, 'from-b'); // b re-reads a's commit under the lock first
        await putFact(a, 'from-a-again'); // and a re-reads b's
        expect((await facts(a)).map((f) => f.id).sort()).toEqual(['from-a', 'from-a-again', 'from-b']);
        const onDisk = JSON.parse(fs.readFileSync(file, 'utf8')) as { facts: StoredFactRecord[] };
        expect(onDisk.facts.map((f) => f.id).sort()).toEqual(['from-a', 'from-a-again', 'from-b']);
        a.close(); b.close();
    });

    it('treats a file deleted underneath it as an empty document', async () => {
        const file = path.join(dir, 'memory.json');
        const adapter = await open(file);
        await putFact(adapter, 'old');
        fs.unlinkSync(file);
        const seen = await adapter.transact(['facts'], 'readwrite', async (tx) => {
            const before = await tx.getAll<StoredFactRecord>('facts');
            await tx.put('facts', fact('new'));
            return before;
        });
        expect(seen).toEqual([]);
        expect((await facts(adapter)).map((f) => f.id)).toEqual(['new']);
        expect(JSON.parse(fs.readFileSync(file, 'utf8')).facts).toHaveLength(1);
        adapter.close();
    });

    it('a never-written file stays absent until the first write', async () => {
        const file = path.join(dir, 'memory.json');
        const adapter = await open(file);
        expect(fs.existsSync(file)).toBe(false);
        await putFact(adapter, 'first');
        expect(fs.existsSync(file)).toBe(true);
        adapter.close();
    });
});

describe('node-file adapter — rollback', () => {
    it.each([
        ['an Error', new Error('disk full'), 'disk full'],
        ['a non-Error value', 'rename refused', 'rename refused'],
    ])('rolls the document back when the persist throws %s', async (_label, thrown, text) => {
        const file = path.join(dir, 'memory.json');
        const good = await open(file);
        await putFact(good, 'kept');
        good.close();

        const adapter = await NodeFileStorageAdapter.open(file, {
            watchIntervalMs: 0,
            path,
            fs: fsWith({ renameSync: () => { throw thrown; } }),
        });
        const err = await putFact(adapter, 'lost').then(() => null, (e: unknown) => e);
        expect(err).toBeInstanceOf(MemoryStorageError);
        expect(err).toMatchObject({ storageCode: 'STORAGE_ABORTED' });
        expect(String((err as Error).message)).toContain(text);
        // Neither memory nor disk shows the failed write, and the lock was released.
        expect((await facts(adapter)).map((f) => f.id)).toEqual(['kept']);
        expect(JSON.parse(fs.readFileSync(file, 'utf8')).facts.map((f: StoredFactRecord) => f.id)).toEqual(['kept']);
        expect(fs.existsSync(`${file}.lock`)).toBe(false);
        adapter.close();
    });

    it('keeps serving transactions after one rejects', async () => {
        const file = path.join(dir, 'memory.json');
        const adapter = await open(file);
        await expect(adapter.transact(['facts'], 'readonly', () => { throw new Error('reader failed'); })).rejects.toThrow('reader failed');
        await putFact(adapter, 'after');
        expect((await facts(adapter)).map((f) => f.id)).toEqual(['after']);
        adapter.close();
    });
});

describe('node-file adapter — change notes', () => {
    it('publish reaches co-process subscribers synchronously', async () => {
        const adapter = await open(path.join(dir, 'memory.json'));
        const notes: ConsistencyNote[] = [];
        const stop = adapter.subscribe((n) => notes.push(n));
        const note: ConsistencyNote = { kind: 'facts-changed', revision: 1, origin: adapter.contextId };
        adapter.publish(note);
        expect(notes).toEqual([note]);
        stop();
        adapter.publish(note);
        expect(notes).toHaveLength(1);
        adapter.close();
    });

    it('the watcher reports a foreign revision, and stops when the last subscriber leaves', async () => {
        const file = path.join(dir, 'memory.json');
        const watcher = await open(file, { watchIntervalMs: 10 });
        const writer = await open(file);
        const notes: ConsistencyNote[] = [];
        const stop = watcher.subscribe((n) => notes.push(n));
        const second = watcher.subscribe(() => undefined); // shares the one timer

        await writer.transact(['meta'], 'readwrite', (tx) => tx.put('meta', counters(5)));
        await waitFor(() => notes.length > 0);
        expect(notes[0]).toEqual({ kind: 'facts-changed', revision: 5, origin: 'foreign' });

        // Quiet ticks (no change) report nothing more.
        await pause(40);
        expect(notes).toHaveLength(1);

        stop();
        second();
        await writer.transact(['meta'], 'readwrite', (tx) => tx.put('meta', counters(6)));
        await pause(60);
        expect(notes).toHaveLength(1);
        watcher.close(); writer.close();
    });

    it('close stops a running watcher', async () => {
        const file = path.join(dir, 'memory.json');
        const watcher = await open(file, { watchIntervalMs: 10 });
        const writer = await open(file);
        const notes: ConsistencyNote[] = [];
        watcher.subscribe((n) => notes.push(n));
        watcher.close();
        await writer.transact(['meta'], 'readwrite', (tx) => tx.put('meta', counters(7)));
        await pause(60);
        expect(notes).toEqual([]);
        watcher.close(); // idempotent
        writer.close();
    });

    it('opens no watcher when polling is disabled', async () => {
        const file = path.join(dir, 'memory.json');
        const watcher = await open(file, { watchIntervalMs: 0 });
        const writer = await open(file);
        const notes: ConsistencyNote[] = [];
        const stop = watcher.subscribe((n) => notes.push(n));
        await writer.transact(['meta'], 'readwrite', (tx) => tx.put('meta', counters(8)));
        await pause(40);
        expect(notes).toEqual([]);
        stop();
        watcher.close(); writer.close();
    });
});
