/**
 * tests/memorystore-persistence.test.ts — durable facts across a store reopen.
 *
 * The local store keeps the full body (no stub, no truncation), keeps the
 * timestamp and TTL it wrote (a reboot must not stamp "now" or extend expiry),
 * evicts over the size cap by persisted decay, and looks keys and tags up on
 * indexes that survive the same dbName being opened again.
 */

import 'fake-indexeddb/auto';
import { MemoryStore } from '../src/memory/MemoryStore.js';
import {
    FACT_KEY_INDEX,
    FACT_TAG_INDEX,
    type FactStoragePort,
    type StorageTransaction,
    type StoredFactRecord,
} from '../src/storage/types.js';

let _db = 0;
const dbName = () => `persist-${++_db}-${Date.now()}`;

const LONG_BODY =
    'verbatim body that must survive a reopen unchanged — ' +
    'x'.repeat(120) +
    ' — end of the original fact, not a replacement.';

function fact(partial: Pick<StoredFactRecord, 'key' | 'content' | 'timestamp'> & Partial<StoredFactRecord>): StoredFactRecord {
    return {
        id: partial.key,
        scope: {},
        seq: 1,
        revision: 1,
        opId: 'seed',
        pendingSync: false,
        ...partial,
    };
}

function openVersion(
    name: string,
    version: number,
    upgrade: (db: IDBDatabase, tx: IDBTransaction, oldVersion: number) => void,
): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(name, version);
        req.onupgradeneeded = (event) => {
            const tx = (event.target as IDBOpenDBRequest).transaction!;
            upgrade(req.result, tx, (event as IDBVersionChangeEvent).oldVersion);
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function factIndexNames(name: string): Promise<string[]> {
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
        const req = indexedDB.open(name);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
    try {
        return Array.from(db.transaction('facts', 'readonly').objectStore('facts').indexNames);
    } finally {
        db.close();
    }
}

describe('local store persistence', () => {
    it('round-trips a body longer than 140 characters unchanged, including across reopen', async () => {
        expect(LONG_BODY.length).toBeGreaterThan(140);
        const name = dbName();
        const writtenAt = Date.now() - 5_000;
        const store = new MemoryStore({ dbName: name });
        await store.remember('long', LONG_BODY, {
            tags: ['alpha', 'beta'],
            timestamp: writtenAt,
            ttlMs: 86_400_000,
            importance: 0.7,
        });

        const first = await store.recall('long');
        expect(first?.content).toBe(LONG_BODY);
        expect(first?.content.length).toBe(LONG_BODY.length);
        store.close();

        const reopened = new MemoryStore({ dbName: name });
        const again = await reopened.recall('long');
        expect(again?.content).toBe(LONG_BODY);
        expect(again?.timestamp).toBe(writtenAt);
        expect(again?.ttlMs).toBe(86_400_000);
        expect(again?.tags).toEqual(['alpha', 'beta']);
        expect(again?.importance).toBe(0.7);

        const byTag = await reopened.recallByTag('alpha');
        expect(byTag.find((e) => e.key === 'long')).toEqual(again);
        expect(await reopened.recallByTag('beta')).toEqual(byTag);
        reopened.close();
    });

    it('keeps the original timestamp and expiry across reopen instead of restamping on boot', async () => {
        const name = dbName();
        const past = Date.now() - 10_000;
        const store = new MemoryStore({ dbName: name, defaultTtlMs: 60_000 });
        await store.remember('dead', 'expired body', { timestamp: past, ttlMs: 1_000 });
        await store.remember('live', 'still here', { timestamp: past, ttlMs: 60_000, tags: ['kept'] });
        await store.remember('untimed', 'no ttl');
        const untimed = await store.recall('untimed');
        expect(untimed?.ttlMs).toBe(60_000);
        const untimedStamp = untimed!.timestamp;
        store.close();

        // A reboot with a different default must not extend a stored TTL or
        // rewrite the stamp that expiry is computed from.
        const reopened = new MemoryStore({ dbName: name, defaultTtlMs: 86_400_000 });
        expect(await reopened.recall('dead')).toBeUndefined();
        const live = await reopened.recall('live');
        expect(live?.content).toBe('still here');
        expect(live?.timestamp).toBe(past);
        expect(live?.ttlMs).toBe(60_000);
        expect(live?.tags).toEqual(['kept']);
        const still = await reopened.recall('untimed');
        expect(still?.timestamp).toBe(untimedStamp);
        expect(still?.ttlMs).toBe(60_000);
        reopened.close();
    });

    it('evicts by persisted decay after reopen, not by boot-time timestamps', async () => {
        const name = dbName();
        const store = new MemoryStore({ dbName: name, maxEntries: 2 });
        await store.remember('expired-important', 'x', { timestamp: 1_000, ttlMs: 1, importance: 0.99 });
        await store.remember('old-low', 'y', { timestamp: 2_000, importance: 0.1 });
        await store.remember('newer-low', 'z', { timestamp: 3_000, importance: 0.1 });
        expect((await store.recallAll()).map((e) => e.key).sort()).toEqual(['newer-low', 'old-low']);
        store.close();

        const reopened = new MemoryStore({ dbName: name, maxEntries: 2 });
        expect((await reopened.recall('old-low'))?.timestamp).toBe(2_000);
        expect((await reopened.recall('newer-low'))?.timestamp).toBe(3_000);
        await reopened.remember('newest', 'w', { timestamp: 4_000, importance: 0.1 });
        expect((await reopened.recallAll()).map((e) => e.key).sort()).toEqual(['newer-low', 'newest']);
        expect((await reopened.recall('newer-low'))?.timestamp).toBe(3_000);
        expect((await reopened.recall('newest'))?.timestamp).toBe(4_000);
        reopened.close();
    });

    it('upgrades a v2 database with key and tag indexes without rewriting stored facts', async () => {
        const name = dbName();
        const writtenAt = Date.now() - 5_000;
        const legacy = await openVersion(name, 2, (db) => {
            db.createObjectStore('facts', { keyPath: 'id' });
            db.createObjectStore('weights');
            db.createObjectStore('tombstones', { keyPath: 'id' });
            db.createObjectStore('journal', { keyPath: 'opId' });
            db.createObjectStore('meta', { keyPath: 'key' });
        });
        await new Promise<void>((resolve, reject) => {
            const tx = legacy.transaction('facts', 'readwrite');
            tx.objectStore('facts').put({
                id: 'proj-key',
                key: 'proj-key',
                scope: {},
                content: LONG_BODY,
                timestamp: writtenAt,
                seq: 4,
                revision: 1,
                opId: 'seed:1',
                ttlMs: 86_400_000,
                tags: ['alpha', 'beta'],
                importance: 0.3,
                pendingSync: false,
            });
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
        legacy.close();

        const store = new MemoryStore({ dbName: name });
        const byKey = await store.recall('proj-key');
        expect(byKey?.content).toBe(LONG_BODY);
        expect(byKey?.timestamp).toBe(writtenAt);
        expect(byKey?.ttlMs).toBe(86_400_000);
        expect(byKey?.tags).toEqual(['alpha', 'beta']);
        expect((await store.recallByTag('beta')).find((e) => e.key === 'proj-key')).toEqual(byKey);
        store.close();

        const names = await factIndexNames(name);
        expect(names).toEqual(expect.arrayContaining([FACT_KEY_INDEX, FACT_TAG_INDEX]));
    });

    it('looks up keys and tags on the index and scans only when the index is missing', async () => {
        const seeded = fact({
            key: 'proj-key',
            content: LONG_BODY,
            timestamp: Date.now() - 5_000,
            tags: ['alpha', 'alpha'],
            ttlMs: 86_400_000,
            importance: 0.4,
            seq: 7,
        });

        const indexedCalls: string[] = [];
        const indexed: FactStoragePort = {
            kind: 'indexedDB',
            transact: (_stores, _mode, fn) => fn({
                get: async () => { indexedCalls.push('get'); return undefined; },
                getAll: async () => { indexedCalls.push('getAll'); throw new Error('full scan'); },
                put: async () => undefined,
                delete: async () => undefined,
                getByIndex: async (_store, indexName, query) => {
                    indexedCalls.push(`index:${indexName}:${query}`);
                    if (indexName === FACT_KEY_INDEX) return seeded.key === query ? [seeded, seeded] : [];
                    if (indexName === FACT_TAG_INDEX) return seeded.tags?.includes(query) ? [seeded, seeded] : [];
                    return undefined;
                },
            } satisfies StorageTransaction),
        };
        const viaIndex = new MemoryStore({ storage: indexed });
        const byKey = await viaIndex.recall('proj-key');
        const byTag = await viaIndex.recallByTag('alpha');
        expect(byKey?.content).toBe(LONG_BODY);
        expect(byKey?.timestamp).toBe(seeded.timestamp);
        expect(byKey?.tags).toEqual(['alpha', 'alpha']);
        expect(byTag).toEqual([byKey]);
        expect(indexedCalls).not.toContain('getAll');
        expect(indexedCalls).not.toContain('get');
        expect(indexedCalls).toContain(`index:${FACT_KEY_INDEX}:proj-key`);
        expect(indexedCalls).toContain(`index:${FACT_TAG_INDEX}:alpha`);
        viaIndex.close();

        const fallbackCalls: string[] = [];
        const missingIndex: FactStoragePort = {
            kind: 'indexedDB',
            transact: (_stores, _mode, fn) => fn({
                get: async (_store, id) => {
                    fallbackCalls.push('get');
                    return id === seeded.id ? seeded : undefined;
                },
                getAll: async () => {
                    fallbackCalls.push('getAll');
                    return [seeded];
                },
                put: async () => undefined,
                delete: async () => undefined,
                getByIndex: async () => {
                    fallbackCalls.push('index-missing');
                    return undefined;
                },
            } satisfies StorageTransaction),
        };
        const viaScan = new MemoryStore({ storage: missingIndex });
        expect((await viaScan.recall('proj-key'))?.content).toBe(LONG_BODY);
        expect((await viaScan.recallByTag('alpha')).map((e) => e.key)).toEqual(['proj-key']);
        expect(fallbackCalls).toContain('index-missing');
        expect(fallbackCalls).toContain('get');
        expect(fallbackCalls).toContain('getAll');
        viaScan.close();
    });
});
