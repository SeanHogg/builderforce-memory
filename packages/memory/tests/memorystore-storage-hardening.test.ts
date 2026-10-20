/**
 * tests/memorystore-storage-hardening.test.ts — the storage-layer contract
 * fixes on top of schema v3:
 *
 *   - storage errors ARE SSMErrors with the public `MEMORY_*` code, the
 *     detailed storage code kept as `storageCode`;
 *   - the sync journal is written only when sync is enabled, and acknowledged
 *     entries are pruned instead of accumulating;
 *   - the IndexedDB adapter opens its BroadcastChannel lazily and unrefs it, so
 *     a store that is never closed cannot keep Node alive;
 *   - cap eviction does not open a read-write transaction while under the cap.
 */

import 'fake-indexeddb/auto';
import { jest } from '@jest/globals';
import { SSMError } from '../src/errors/SSMError.js';
import { MemoryStore } from '../src/memory/MemoryStore.js';
import { IdbStorageAdapter, type BroadcastChannelLike } from '../src/storage/idbAdapter.js';
import {
    MemoryStorageError,
    isStorageConflict,
    type FactStoragePort,
    type JournalEntry,
} from '../src/storage/types.js';

let _db = 0;
const dbName = () => `hardening-${++_db}-${Date.now()}`;

/** Every journal row, synced or not, read straight from the store. */
async function journalRows(store: MemoryStore): Promise<JournalEntry[]> {
    return store.repository.storage.transact(['journal'], 'readonly', (tx) => tx.getAll<JournalEntry>('journal'));
}

describe('storage errors keep the public SSMError contract', () => {
    it('maps every storage code onto an SSMError code and keeps the detail', () => {
        const unavailable = new MemoryStorageError('STORAGE_UNAVAILABLE', 'no idb');
        expect(unavailable).toBeInstanceOf(SSMError);
        expect(unavailable).toBeInstanceOf(MemoryStorageError);
        expect(unavailable).toMatchObject({ code: 'MEMORY_UNAVAILABLE', storageCode: 'STORAGE_UNAVAILABLE', name: 'MemoryStorageError' });
        expect(new MemoryStorageError('STORAGE_ABORTED', 'x')).toMatchObject({ code: 'MEMORY_UNAVAILABLE', storageCode: 'STORAGE_ABORTED' });
        const conflict = new MemoryStorageError('STORAGE_CONFLICT', 'x');
        expect(conflict).toMatchObject({ code: 'MEMORY_CONFLICT', storageCode: 'STORAGE_CONFLICT' });
        expect(isStorageConflict(conflict)).toBe(true);
        expect(isStorageConflict(unavailable)).toBe(false);
    });

    it('rejects a remember without IndexedDB with an SSMError MEMORY_UNAVAILABLE', async () => {
        const store = new MemoryStore({ storage: new IdbStorageAdapter('none', { idbFactory: undefined, channelFactory: () => undefined }) });
        const saved = (globalThis as { indexedDB?: unknown }).indexedDB;
        delete (globalThis as { indexedDB?: unknown }).indexedDB;
        try {
            const err = await store.remember('k', 'v').then(() => null, (e: unknown) => e);
            expect(err).toBeInstanceOf(SSMError);
            expect(err).toMatchObject({ code: 'MEMORY_UNAVAILABLE', storageCode: 'STORAGE_UNAVAILABLE' });
        } finally {
            (globalThis as { indexedDB?: unknown }).indexedDB = saved;
        }
    });

    it('surfaces a CAS conflict as MEMORY_CONFLICT', async () => {
        const store = new MemoryStore({ dbName: dbName() });
        await store.rememberCas('k', 'v1', null);
        const err = await store.rememberCas('k', 'v2', null).then(() => null, (e: unknown) => e);
        expect(isStorageConflict(err)).toBe(true);
        expect(err).toMatchObject({ code: 'MEMORY_CONFLICT' });
        store.close();
    });
});

describe('sync journal', () => {
    it('writes no journal rows when sync is off', async () => {
        const store = new MemoryStore({ dbName: dbName() });
        await store.remember('a', '1');
        await store.remember('b', '2');
        await store.forget('a');
        await store.importAll([{ key: 'c', content: '3', timestamp: Date.now() }], 'merge');
        await store.clear();
        expect(await journalRows(store)).toEqual([]);
        expect(await store.pendingOps()).toEqual([]);
        expect(await store.revision()).toBeGreaterThan(0);
        store.close();
    });

    it('journals unsynced ops when sync is on and prunes them on acknowledgement', async () => {
        const store = new MemoryStore({ dbName: dbName(), sync: true, contextId: 'ctx' });
        await store.remember('a', '1');
        await store.remember('b', '2');
        await store.forget('a');
        const pending = await store.pendingOps();
        expect(pending.map((j) => j.kind)).toEqual(['remember', 'remember', 'forget']);
        expect(pending.every((j) => !j.synced)).toBe(true);
        expect((await store.recall('b'))?.pendingSync).toBe(true);

        await store.markSynced(pending.slice(0, 2).map((j) => j.opId));
        const rows = await journalRows(store);
        expect(rows.map((j) => j.opId)).toEqual([pending[2]!.opId]);
        expect((await store.recall('b'))?.pendingSync).toBe(false);

        await store.markSynced([pending[2]!.opId]);
        expect(await journalRows(store)).toEqual([]);
        expect((await store.tombstones())[0]?.pendingSync).toBe(false);
        store.close();
    });

    it('prunes legacy already-synced rows on the next acknowledgement', async () => {
        const store = new MemoryStore({ dbName: dbName(), sync: true });
        await store.repository.storage.transact(['journal'], 'readwrite', (tx) => tx.put('journal', {
            opId: 'legacy:1', kind: 'remember', id: 'x', key: 'x', scope: {}, revision: 1, timestamp: 1, synced: true,
        } satisfies JournalEntry));
        await store.remember('a', '1');
        const [op] = await store.pendingOps();
        await store.markSynced([op!.opId]);
        expect(await journalRows(store)).toEqual([]);
        store.close();
    });
});

describe('IndexedDB change channel', () => {
    function fakeChannel() {
        return {
            postMessage: jest.fn(),
            close      : jest.fn(),
            onmessage  : null as BroadcastChannelLike['onmessage'],
            unref      : jest.fn(),
        };
    }

    it('opens no channel at construction, opens it lazily, and unrefs it', async () => {
        const channel = fakeChannel();
        const factory = jest.fn((_name: string) => channel);
        const store = new MemoryStore({ storage: new IdbStorageAdapter(dbName(), { channelFactory: factory }) });
        expect(factory).not.toHaveBeenCalled();

        await store.recallAll();
        expect(factory).toHaveBeenCalledTimes(1);
        expect(channel.unref).toHaveBeenCalledTimes(1);
        expect(channel.onmessage).not.toBeNull();

        await store.remember('k', 'v');
        expect(factory).toHaveBeenCalledTimes(1);
        expect(channel.postMessage).toHaveBeenCalled();
        store.close();
        expect(channel.close).toHaveBeenCalled();
    });

    it('still delivers foreign notes after the lazy open', async () => {
        const channel = fakeChannel();
        const store = new MemoryStore({ storage: new IdbStorageAdapter(dbName(), { channelFactory: () => channel, contextId: 'me' }) });
        await store.recallAll();
        channel.onmessage!({ data: { kind: 'facts-changed', revision: 7, origin: 'other-tab' } });
        expect(store.foreignRevision).toBe(7);
        store.close();
    });
});

describe('cap eviction', () => {
    function countingPort(inner: FactStoragePort): { port: FactStoragePort; modes: string[] } {
        const modes: string[] = [];
        const port: FactStoragePort = {
            kind: inner.kind,
            transact: (stores, mode, fn) => {
                modes.push(`${mode}:${stores.join(',')}`);
                return inner.transact(stores, mode, fn);
            },
        };
        return { port, modes };
    }

    it('counts read-only and opens no write transaction while under the cap', async () => {
        const { port, modes } = countingPort(new IdbStorageAdapter(dbName(), { channelFactory: () => undefined }));
        const store = new MemoryStore({ storage: port, maxEntries: 5 });
        await store.remember('a', '1');
        expect(modes).toEqual(['readwrite:facts,tombstones,journal,meta', 'readonly:facts']);
    });

    it('evicts in a write transaction only once over the cap', async () => {
        const { port, modes } = countingPort(new IdbStorageAdapter(dbName(), { channelFactory: () => undefined }));
        const store = new MemoryStore({ storage: port, maxEntries: 1 });
        await store.remember('a', '1', { timestamp: 1_000 });
        modes.length = 0;
        await store.remember('b', '2', { timestamp: 2_000 });
        expect(modes).toEqual(['readwrite:facts,tombstones,journal,meta', 'readonly:facts', 'readwrite:facts']);
        expect((await store.recallAll()).map((e) => e.key)).toEqual(['b']);
    });
});
