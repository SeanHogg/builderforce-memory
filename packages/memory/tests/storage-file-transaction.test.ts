/**
 * tests/storage-file-transaction.test.ts — the node-file adapter's unit of work
 * against the in-memory document: `meta` is ONE counters record (`doc.meta`),
 * the other stores are tables keyed by their id field (`opId` for the journal),
 * and the fact key/tag indexes are rebuilt only after a fact mutation.
 */

import { FileTransaction, emptyDocument, type NodeFileDocument } from '../src/storage/fileTransaction.js';
import {
    FACT_KEY_INDEX,
    FACT_TAG_INDEX,
    type JournalEntry,
    type MetaCounters,
    type StoredFactRecord,
    type TombstoneRecord,
} from '../src/storage/types.js';

function fact(id: string, key: string, tags?: string[]): StoredFactRecord {
    return {
        id,
        key,
        scope: {},
        content: `content of ${id}`,
        timestamp: 1,
        seq: 1,
        revision: 1,
        opId: `op-${id}`,
        pendingSync: false,
        ...(tags ? { tags } : {}),
    };
}

function journal(opId: string, id: string): JournalEntry {
    return { opId, kind: 'put', id, key: id, scope: {}, revision: 1, timestamp: 1 } as unknown as JournalEntry;
}

function tombstone(id: string): TombstoneRecord {
    return { id, key: id, scope: {}, deletedAt: 1, revision: 2, opId: `del-${id}`, pendingSync: false };
}

describe('FileTransaction — the meta singleton', () => {
    it('get and getAll read doc.meta, and only under its own key', async () => {
        const doc = emptyDocument();
        const tx = new FileTransaction(doc);
        expect(await tx.get('meta', 'counters')).toBe(doc.meta);
        expect(await tx.get('meta', 'something-else')).toBeUndefined();
        expect(await tx.getAll('meta')).toEqual([doc.meta]);
    });

    it('a document with no meta record reads as empty', async () => {
        const doc = { ...emptyDocument(), meta: undefined } as unknown as NodeFileDocument;
        const tx = new FileTransaction(doc);
        expect(await tx.get('meta', 'counters')).toBeUndefined();
        expect(await tx.getAll('meta')).toEqual([]);
    });

    it('put replaces the singleton; delete resets it only when the key matches', async () => {
        const doc = emptyDocument();
        const tx = new FileTransaction(doc);
        const next: MetaCounters = { key: 'counters', seq: 4, revision: 9, opSeq: 2 };
        await tx.put('meta', next);
        expect(doc.meta).toBe(next);
        expect(doc.facts).toEqual([]); // meta never lands in a table

        await tx.delete('meta', 'not-the-counters');
        expect(doc.meta).toBe(next);

        await tx.delete('meta', 'counters');
        expect(doc.meta).toEqual(emptyDocument().meta);
    });
});

describe('FileTransaction — tables', () => {
    it('put inserts a new id and replaces an existing one in place', async () => {
        const doc = emptyDocument();
        const tx = new FileTransaction(doc);
        await tx.put('facts', fact('a', 'k1'));
        await tx.put('facts', fact('b', 'k2'));
        const updated = { ...fact('a', 'k1'), content: 'rewritten' };
        await tx.put('facts', updated);
        expect(doc.facts.map((f) => f.id)).toEqual(['a', 'b']);
        expect(await tx.get('facts', 'a')).toEqual(updated);
        expect(await tx.get('facts', 'missing')).toBeUndefined();
    });

    it('keys the journal by opId and tombstones by id', async () => {
        const doc = emptyDocument();
        const tx = new FileTransaction(doc);
        await tx.put('journal', journal('ctx:1', 'f1'));
        await tx.put('journal', journal('ctx:2', 'f1'));
        await tx.put('journal', { ...journal('ctx:1', 'f1'), revision: 5 });
        expect(doc.journal.map((j) => [j.opId, j.revision])).toEqual([['ctx:1', 5], ['ctx:2', 1]]);
        expect((await tx.get<JournalEntry>('journal', 'ctx:2'))?.opId).toBe('ctx:2');

        await tx.put('tombstones', tombstone('t1'));
        expect(await tx.getAll('tombstones')).toEqual([tombstone('t1')]);
        await tx.delete('tombstones', 't1');
        expect(doc.tombstones).toEqual([]);
    });

    it('getAll returns a copy, and deleting a missing id changes nothing', async () => {
        const doc = emptyDocument();
        const tx = new FileTransaction(doc);
        await tx.put('facts', fact('a', 'k'));
        const all = await tx.getAll<StoredFactRecord>('facts');
        all.pop();
        expect(doc.facts).toHaveLength(1);

        await tx.delete('facts', 'nope');
        await tx.delete('journal', 'nope');
        expect(doc.facts).toHaveLength(1);
        await tx.delete('facts', 'a');
        expect(doc.facts).toEqual([]);
    });
});

describe('FileTransaction — fact indexes', () => {
    it('answers only the facts store and the two known indexes', async () => {
        const tx = new FileTransaction(emptyDocument());
        expect(await tx.getByIndex('tombstones', FACT_KEY_INDEX, 'k')).toBeUndefined();
        expect(await tx.getByIndex('facts', 'byNothing', 'k')).toBeUndefined();
        expect(await tx.getByIndex('facts', FACT_KEY_INDEX, 'absent')).toEqual([]);
        expect(await tx.getByIndex('facts', FACT_TAG_INDEX, 'absent')).toEqual([]);
    });

    it('groups facts by key and by every tag (facts without tags are only keyed)', async () => {
        const doc = emptyDocument();
        doc.facts.push(fact('a', 'shared', ['red', 'blue']), fact('b', 'shared', ['red']), fact('c', 'solo'));
        const tx = new FileTransaction(doc);
        expect((await tx.getByIndex<StoredFactRecord>('facts', FACT_KEY_INDEX, 'shared'))!.map((f) => f.id)).toEqual(['a', 'b']);
        expect((await tx.getByIndex<StoredFactRecord>('facts', FACT_KEY_INDEX, 'solo'))!.map((f) => f.id)).toEqual(['c']);
        expect((await tx.getByIndex<StoredFactRecord>('facts', FACT_TAG_INDEX, 'red'))!.map((f) => f.id)).toEqual(['a', 'b']);
        expect((await tx.getByIndex<StoredFactRecord>('facts', FACT_TAG_INDEX, 'blue'))!.map((f) => f.id)).toEqual(['a']);
    });

    it('reuses the built index until a fact put or delete marks it dirty', async () => {
        const doc = emptyDocument();
        const tx = new FileTransaction(doc);
        await tx.put('facts', fact('a', 'k', ['t']));
        expect(await tx.getByIndex('facts', FACT_KEY_INDEX, 'k')).toHaveLength(1);

        // A change behind the transaction's back is not seen: the index is cached.
        doc.facts.push(fact('ghost', 'k'));
        expect(await tx.getByIndex('facts', FACT_KEY_INDEX, 'k')).toHaveLength(1);

        // A non-fact write leaves it cached too.
        await tx.put('tombstones', tombstone('t'));
        expect(await tx.getByIndex('facts', FACT_KEY_INDEX, 'k')).toHaveLength(1);

        // A fact mutation rebuilds it from the document.
        await tx.put('facts', fact('b', 'k'));
        expect(await tx.getByIndex('facts', FACT_KEY_INDEX, 'k')).toHaveLength(3);
        await tx.delete('facts', 'ghost');
        expect((await tx.getByIndex<StoredFactRecord>('facts', FACT_KEY_INDEX, 'k'))!.map((f) => f.id)).toEqual(['a', 'b']);
        expect((await tx.getByIndex<StoredFactRecord>('facts', FACT_TAG_INDEX, 't'))!.map((f) => f.id)).toEqual(['a']);
    });
});
