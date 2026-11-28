export interface SemanticCacheScope {
    /** Account/workspace partition selected by the host. */
    namespace: string;
    knowledgeVersion: string;
    embeddingVersion: string;
    modelVersion: string;
    tokenizerVersion: string;
    /** Input and tool semantics for this cache instance. */
    inputVersion: string;
}

export interface SemanticCacheEntry {
    embedding: Float32Array;
    response: string;
    timestamp: number;
    context: string;
}

export interface SemanticCachePersistence {
    read(scope: string): Promise<SemanticCacheEntry[]>;
    append(scope: string, entry: SemanticCacheEntry, maxEntries: number): Promise<void>;
    clear(scope: string): Promise<void>;
}

export function semanticCacheScopeKey(scope: SemanticCacheScope): string {
    return JSON.stringify([
        1, scope.namespace, scope.knowledgeVersion, scope.embeddingVersion, scope.modelVersion,
        scope.tokenizerVersion, scope.inputVersion,
    ]);
}

/** Persists vectors directly through IndexedDB structured clone. Each append is a read/write transaction. */
export function indexedDbSemanticCache(
    factory: IDBFactory,
    dbName = 'builderforce-memory-semantic-cache',
): SemanticCachePersistence & { close(): Promise<void> } {
    let database: Promise<IDBDatabase> | undefined;

    function open(): Promise<IDBDatabase> {
        database ??= new Promise<IDBDatabase>((resolve, reject) => {
            const request = factory.open(dbName, 1);
            request.onupgradeneeded = () => { request.result.createObjectStore('scopes'); };
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        }).catch((error) => {
            database = undefined;
            throw error;
        });
        return database;
    }

    async function transact<T>(
        mode: IDBTransactionMode,
        operation: (store: IDBObjectStore, result: (value: T) => void) => void,
    ): Promise<T> {
        const db = await open();
        return new Promise<T>((resolve, reject) => {
            const tx = db.transaction('scopes', mode);
            let result: T;
            tx.oncomplete = () => resolve(result);
            tx.onabort = () => reject(tx.error ?? new Error('semantic_cache_transaction_aborted'));
            operation(tx.objectStore('scopes'), (value) => { result = value; });
        });
    }

    return {
        read(scope) {
            return transact('readonly', (store, result) => {
                const request = store.get(scope);
                request.onsuccess = () => result(request.result ?? []);
            });
        },
        append(scope, entry, maxEntries) {
            return transact('readwrite', (store) => {
                const request = store.get(scope);
                request.onsuccess = () => {
                    const entries: SemanticCacheEntry[] = request.result ?? [];
                    entries.push(entry);
                    store.put(maxEntries > 0 ? entries.slice(-maxEntries) : [], scope);
                };
            });
        },
        clear(scope) {
            return transact('readwrite', (store) => { store.delete(scope); });
        },
        async close() {
            (await database)?.close();
            database = undefined;
        },
    };
}
