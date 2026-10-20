/**
 * tests/storage-idb-adapter.test.ts — the IndexedDB adapter's change channel,
 * index fallback, transaction-open failures and schema migrations, on
 * fake-indexeddb.
 *
 *   - the channel opens lazily (first publish, or once the database is open
 *     for a subscriber), filters malformed and own-origin notes, and a missing,
 *     throwing or dead channel never fails a write;
 *   - an index the database lacks falls back (undefined) instead of throwing;
 *   - a transaction that cannot even start rejects as STORAGE_ABORTED, keeping
 *     a MemoryStorageError as it is;
 *   - a version-1 database migrates its raw-key facts onto scoped records, and
 *     an old database with no facts store gets one.
 */

import 'fake-indexeddb/auto';
import { jest } from '@jest/globals';
import { IdbStorageAdapter, type BroadcastChannelLike } from '../src/storage/idbAdapter.js';
import {
    FACT_KEY_INDEX,
    FACT_TAG_INDEX,
    MemoryStorageError,
    scopedRecordId,
    type ConsistencyNote,
    type StoredFactRecord,
} from '../src/storage/types.js';

let _db = 0;
const dbName = () => `idb-branches-${++_db}-${Date.now()}`;

function fakeChannel() {
    return {
        postMessage: jest.fn<(message: unknown) => void>(),
        close: jest.fn(),
        onmessage: null as BroadcastChannelLike['onmessage'],
        unref: jest.fn(),
    };
}

function fact(id: string, tags?: string[]): StoredFactRecord {
    return { id, key: id, scope: {}, content: `content of ${id}`, timestamp: 1, seq: 1, revision: 1, opId: `op-${id}`, pendingSync: false, ...(tags ? { tags } : {}) };
}

/** Opens the database read-only, which is what opens it (and so the channel) at all. */
const touch = (adapter: IdbStorageAdapter) => adapter.transact(['facts'], 'readonly', (tx) => tx.getAll('facts'));

const note = (origin: string, revision = 1): ConsistencyNote => ({ kind: 'facts-changed', revision, origin });

/** An IDBFactory whose database refuses to start any transaction. */
function refusingFactory(thrown: unknown): IDBFactory {
    return {
        open() {
            const req: Record<string, unknown> = {};
            setTimeout(() => {
                req.result = { transaction() { throw thrown; }, close() { /* nothing to close */ } };
                (req.onsuccess as () => void)();
            }, 0);
            return req;
        },
    } as unknown as IDBFactory;
}

function openRaw(name: string, version: number, upgrade: (db: IDBDatabase) => void): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(name, version);
        req.onupgradeneeded = () => upgrade(req.result);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

describe('IndexedDB adapter — change channel', () => {
    it('opens the channel when the database opens for an early subscriber, and filters what it delivers', async () => {
        const channel = fakeChannel();
        const factory = jest.fn((_name: string) => channel);
        const adapter = new IdbStorageAdapter(dbName(), { channelFactory: factory, contextId: 'me' });
        const received: ConsistencyNote[] = [];
        const stop = adapter.subscribe((n) => received.push(n));
        expect(factory).not.toHaveBeenCalled(); // subscribing alone opens nothing

        await touch(adapter);
        expect(factory).toHaveBeenCalledTimes(1);
        expect(channel.unref).toHaveBeenCalledTimes(1);

        channel.onmessage!({ data: null });
        channel.onmessage!({ data: { kind: 'something-else', revision: 3, origin: 'other' } });
        channel.onmessage!({ data: note('me') }); // our own write echoed back
        expect(received).toEqual([]);

        channel.onmessage!({ data: note('other-tab', 4) });
        expect(received).toEqual([note('other-tab', 4)]);

        stop();
        channel.onmessage!({ data: note('other-tab', 5) });
        expect(received).toHaveLength(1);

        adapter.close();
        expect(channel.close).toHaveBeenCalledTimes(1);
    });

    it('a subscriber arriving after the open gets the channel straight away, once', async () => {
        const channel = fakeChannel();
        const factory = jest.fn((_name: string) => channel);
        const adapter = new IdbStorageAdapter(dbName(), { channelFactory: factory });
        await touch(adapter);
        expect(factory).not.toHaveBeenCalled(); // nobody listening yet
        adapter.subscribe(() => undefined);
        adapter.subscribe(() => undefined);
        expect(factory).toHaveBeenCalledTimes(1);
        adapter.close();
    });

    it('publish opens the channel lazily and posts the note', () => {
        const channel = fakeChannel();
        const adapter = new IdbStorageAdapter(dbName(), { channelFactory: () => channel, contextId: 'me' });
        adapter.publish(note('me'));
        adapter.publish(note('me', 2));
        expect(channel.postMessage.mock.calls.map(([m]) => m)).toEqual([note('me'), note('me', 2)]);
        adapter.close();
    });

    it('a host with no channel, a factory that throws, or a dead channel never fails a publish', () => {
        const none = new IdbStorageAdapter(dbName(), { channelFactory: () => undefined });
        expect(() => none.publish(note('x'))).not.toThrow();

        const throwingFactory = jest.fn((_name: string): BroadcastChannelLike => { throw new Error('no channels here'); });
        const broken = new IdbStorageAdapter(dbName(), { channelFactory: throwingFactory });
        expect(() => broken.publish(note('x'))).not.toThrow();
        expect(() => broken.publish(note('x'))).not.toThrow();
        expect(throwingFactory).toHaveBeenCalledTimes(1); // the failure is remembered

        const dead = fakeChannel();
        dead.postMessage.mockImplementation(() => { throw new Error('channel closed'); });
        const deadAdapter = new IdbStorageAdapter(dbName(), { channelFactory: () => dead });
        expect(() => deadAdapter.publish(note('x'))).not.toThrow();
        expect(dead.postMessage).toHaveBeenCalledTimes(1);

        none.close(); broken.close(); deadAdapter.close();
    });

    it('with no options, uses the global IndexedDB and BroadcastChannel between two adapters', async () => {
        const name = dbName();
        const writer = new IdbStorageAdapter(name);
        const reader = new IdbStorageAdapter(name);
        const received = new Promise<ConsistencyNote>((resolve) => reader.subscribe(resolve));
        await touch(reader);
        await writer.transact(['facts'], 'readwrite', (tx) => tx.put('facts', fact('shared')));
        writer.publish(note(writer.contextId, 9));
        expect(await received).toEqual(note(writer.contextId, 9));
        expect((await touch(reader)) as StoredFactRecord[]).toHaveLength(1);
        writer.close(); reader.close();
    });
});

describe('IndexedDB adapter — transactions', () => {
    it('falls back (undefined) for an index the store lacks, and reads the ones it has', async () => {
        const adapter = new IdbStorageAdapter(dbName(), { channelFactory: () => undefined });
        await adapter.transact(['facts'], 'readwrite', async (tx) => {
            await tx.put('facts', fact('a', ['red']));
            await tx.put('facts', fact('b'));
        });
        const read = await adapter.transact(['facts'], 'readonly', async (tx) => ({
            missing: await tx.getByIndex('facts', 'byNothing', 'a'),
            byKey: await tx.getByIndex<StoredFactRecord>('facts', FACT_KEY_INDEX, 'b'),
            byTag: await tx.getByIndex<StoredFactRecord>('facts', FACT_TAG_INDEX, 'red'),
        }));
        expect(read.missing).toBeUndefined();
        expect(read.byKey!.map((f) => f.id)).toEqual(['b']);
        expect(read.byTag!.map((f) => f.id)).toEqual(['a']);
        adapter.close();
    });

    it('get and delete round-trip, and a missing id reads undefined', async () => {
        const adapter = new IdbStorageAdapter(dbName(), { channelFactory: () => undefined });
        await adapter.transact(['facts'], 'readwrite', (tx) => tx.put('facts', fact('a')));
        expect((await adapter.transact(['facts'], 'readonly', (tx) => tx.get<StoredFactRecord>('facts', 'a')))?.id).toBe('a');
        await adapter.transact(['facts'], 'readwrite', (tx) => tx.delete('facts', 'a'));
        expect(await adapter.transact(['facts'], 'readonly', (tx) => tx.get('facts', 'a'))).toBeUndefined();
        adapter.close();
    });

    it('a transaction that cannot start rejects as STORAGE_ABORTED with the cause', async () => {
        const adapter = new IdbStorageAdapter('refuses-error', { idbFactory: refusingFactory(new Error('store not found')), channelFactory: () => undefined });
        const err = await touch(adapter).then(() => null, (e: unknown) => e);
        expect(err).toBeInstanceOf(MemoryStorageError);
        expect(err).toMatchObject({ storageCode: 'STORAGE_ABORTED' });
        expect(String((err as Error).message)).toContain('store not found');
        adapter.close();
    });

    it('a non-Error refusal is stringified into the message', async () => {
        const adapter = new IdbStorageAdapter('refuses-string', { idbFactory: refusingFactory('quota gone'), channelFactory: () => undefined });
        const err = await touch(adapter).then(() => null, (e: unknown) => e);
        expect(err).toMatchObject({ storageCode: 'STORAGE_ABORTED' });
        expect(String((err as Error).message)).toContain('quota gone');
        adapter.close();
    });

    it('a MemoryStorageError refusal passes through unchanged', async () => {
        const original = new MemoryStorageError('STORAGE_UNAVAILABLE', 'database is gone');
        const adapter = new IdbStorageAdapter('refuses-storage-error', { idbFactory: refusingFactory(original), channelFactory: () => undefined });
        await expect(touch(adapter)).rejects.toBe(original);
        adapter.close();
    });

    it('an unknown store name rejects instead of hanging', async () => {
        const adapter = new IdbStorageAdapter(dbName(), { channelFactory: () => undefined });
        const err = await adapter.transact(['nosuch' as never], 'readonly', () => 'ran').then(() => null, (e: unknown) => e);
        expect(err).toBeInstanceOf(MemoryStorageError);
        expect(err).toMatchObject({ storageCode: 'STORAGE_ABORTED' });
        adapter.close();
    });
});

describe('IndexedDB adapter — schema migration', () => {
    it('migrates version-1 raw-key facts onto scoped records, keeping seq and tags', async () => {
        const name = dbName();
        const legacy = await openRaw(name, 1, (db) => {
            const facts = db.createObjectStore('facts', { keyPath: 'key' });
            facts.put({ key: 'alpha', content: 'A', timestamp: 5, seq: 3, tags: ['red'], ttlMs: 1000, type: 'fact', importance: 0.5 });
            facts.put({ key: 'beta', content: 'B', timestamp: 6 });
            db.createObjectStore('weights');
        });
        legacy.close();

        const adapter = new IdbStorageAdapter(name, { channelFactory: () => undefined });
        const read = await adapter.transact(['facts', 'tombstones', 'journal', 'meta'], 'readonly', async (tx) => ({
            facts: await tx.getAll<StoredFactRecord>('facts'),
            byTag: await tx.getByIndex<StoredFactRecord>('facts', FACT_TAG_INDEX, 'red'),
            tombstones: await tx.getAll('tombstones'),
            journal: await tx.getAll('journal'),
            meta: await tx.getAll('meta'),
        }));
        const byKey = Object.fromEntries(read.facts.map((f) => [f.key, f]));
        expect(byKey.alpha).toMatchObject({
            id: scopedRecordId(undefined, 'alpha'),
            scope: {},
            content: 'A',
            timestamp: 5,
            seq: 3,
            revision: 1,
            opId: 'migrated:v1',
            ttlMs: 1000,
            tags: ['red'],
            importance: 0.5,
            pendingSync: false,
        });
        expect(byKey.beta).toMatchObject({ id: scopedRecordId(undefined, 'beta'), seq: 0, timestamp: 6 });
        expect(read.byTag!.map((f) => f.key)).toEqual(['alpha']);
        expect(read.tombstones).toEqual([]);
        expect(read.journal).toEqual([]);
        expect(read.meta).toEqual([]);
        adapter.close();
    });

    it('gives an old database with no facts store one, with its indexes', async () => {
        const name = dbName();
        const legacy = await openRaw(name, 1, (db) => { db.createObjectStore('weights'); });
        legacy.close();

        const adapter = new IdbStorageAdapter(name, { channelFactory: () => undefined });
        await adapter.transact(['facts'], 'readwrite', (tx) => tx.put('facts', fact('fresh', ['t'])));
        const hits = await adapter.transact(['facts'], 'readonly', async (tx) => ({
            byKey: await tx.getByIndex<StoredFactRecord>('facts', FACT_KEY_INDEX, 'fresh'),
            byTag: await tx.getByIndex<StoredFactRecord>('facts', FACT_TAG_INDEX, 't'),
        }));
        expect(hits.byKey!.map((f) => f.id)).toEqual(['fresh']);
        expect(hits.byTag!.map((f) => f.id)).toEqual(['fresh']);
        adapter.close();
    });
});
