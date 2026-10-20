/**
 * storage/fileTransaction.ts – The {@link StorageTransaction} the node-file
 * adapter hands to a unit of work: direct reads/writes on the in-memory
 * document, plus lazily rebuilt key/tag indexes over its facts.
 */

import {
    FACT_KEY_INDEX,
    FACT_TAG_INDEX,
    type JournalEntry,
    type MetaCounters,
    type StorageStoreName,
    type StorageTransaction,
    type StoredFactRecord,
    type TombstoneRecord,
} from './types.js';

/** The on-disk document shape. Versioned so future migrations are explicit. */
export interface NodeFileDocument {
    formatVersion: 1;
    facts: StoredFactRecord[];
    tombstones: TombstoneRecord[];
    journal: JournalEntry[];
    meta: MetaCounters;
}

export function emptyDocument(): NodeFileDocument {
    return {
        formatVersion: 1,
        facts: [],
        tombstones: [],
        journal: [],
        meta: { key: 'counters', seq: 0, revision: 0, opSeq: 0 },
    };
}

export class FileTransaction implements StorageTransaction {
    /** Lazy key/tag indexes over the in-memory document. Rebuilt after mutations. */
    private _indexDirty = true;
    private readonly _byKey = new Map<string, StoredFactRecord[]>();
    private readonly _byTag = new Map<string, StoredFactRecord[]>();

    constructor(private readonly _doc: NodeFileDocument) {}

    private _table(store: StorageStoreName): Array<{ id?: string; opId?: string; key?: string }> {
        return this._doc[store] as unknown as Array<{ id?: string; opId?: string; key?: string }>;
    }

    /** The id field of a table store (`meta` is the singleton and never reaches here). */
    private _idField(store: Exclude<StorageStoreName, 'meta'>): 'id' | 'opId' {
        return store === 'journal' ? 'opId' : 'id';
    }

    // `meta` is ONE counters record on disk (`doc.meta`), not an array: the other stores
    // are tables keyed by their id field, `meta` is a singleton keyed 'counters'.
    async get<T>(store: StorageStoreName, id: string): Promise<T | undefined> {
        if (store === 'meta') return (this._doc.meta?.key === id ? this._doc.meta : undefined) as T | undefined;
        const f = this._idField(store);
        return this._table(store).find((r) => r[f] === id) as T | undefined;
    }
    async getAll<T>(store: StorageStoreName): Promise<T[]> {
        if (store === 'meta') return (this._doc.meta ? [this._doc.meta] : []) as unknown as T[];
        return this._table(store).slice() as unknown as T[];
    }
    async put(store: StorageStoreName, value: unknown): Promise<void> {
        if (store === 'meta') {
            this._doc.meta = value as MetaCounters;
            return;
        }
        const f = this._idField(store);
        const table = this._table(store);
        const id = (value as Record<string, string>)[f]!;
        const idx = table.findIndex((r) => r[f] === id);
        if (idx >= 0) table[idx] = value as never;
        else table.push(value as never);
        if (store === 'facts') this._indexDirty = true;
    }
    async delete(store: StorageStoreName, id: string): Promise<void> {
        if (store === 'meta') {
            if (this._doc.meta?.key === id) this._doc.meta = emptyDocument().meta;
            return;
        }
        const f = this._idField(store);
        const table = this._table(store);
        const idx = table.findIndex((r) => r[f] === id);
        if (idx >= 0) table.splice(idx, 1);
        if (store === 'facts') this._indexDirty = true;
    }

    async getByIndex<T>(store: StorageStoreName, indexName: string, query: string): Promise<T[] | undefined> {
        if (store !== 'facts') return undefined;
        if (indexName !== FACT_KEY_INDEX && indexName !== FACT_TAG_INDEX) return undefined;
        this._ensureFactIndexes();
        const hits = indexName === FACT_KEY_INDEX
            ? this._byKey.get(query)
            : this._byTag.get(query);
        return (hits ?? []).slice() as T[];
    }

    private _ensureFactIndexes(): void {
        if (!this._indexDirty) return;
        this._byKey.clear();
        this._byTag.clear();
        for (const fact of this._doc.facts) {
            const keyHits = this._byKey.get(fact.key);
            if (keyHits) keyHits.push(fact);
            else this._byKey.set(fact.key, [fact]);
            for (const tag of fact.tags ?? []) {
                const tagHits = this._byTag.get(tag);
                if (tagHits) tagHits.push(fact);
                else this._byTag.set(tag, [fact]);
            }
        }
        this._indexDirty = false;
    }
}
