/**
 * storage/nodeFileAdapter.ts – Durable Node.js implementation of the storage
 * port, backed by a single atomically-replaced JSON document.
 *
 * This is the VSIX extension-host / MCP adapter the offline-memory scope (M01)
 * requires: real durable storage on Node WITHOUT an in-memory IndexedDB shim.
 * fake-indexeddb evaporates with the process; this adapter fsyncs a temp file
 * and atomically renames it over the database file, so a crash mid-write leaves
 * either the old or the new document — never a torn one.
 *
 * Transaction semantics (M02/M05):
 *   - All transactions are serialized through one promise queue, so concurrent
 *     writers in this process interleave at transaction granularity.
 *   - A read-write transaction holds an exclusive `<file>.lock` (see
 *     `fileLock.ts`: atomic create, stale-lock reclamation by PID liveness,
 *     bounded wait) across refresh + mutate + persist, so two processes can
 *     never both build on the same stale document and lose a write.
 *   - Under that lock, the file is re-read when another process changed it
 *     (mtime/size freshness), so cross-process writers build on each other's
 *     commits instead of blindly overwriting.
 *   - If the persist fails, the in-memory document is rolled back to the
 *     pre-transaction snapshot and the transaction rejects — no partial commits.
 *
 * Cross-process consistency: a polling watcher (default 1s, injectable)
 * republishes foreign changes to subscribers, the Node analogue of the IDB
 * adapter's BroadcastChannel.
 *
 * Node builtins are loaded indirectly (via a non-static import) so a browser
 * bundle of this package never statically pulls in `node:fs`.
 */

import { FileLock, type FileLockOptions } from './fileLock.js';
import { FileTransaction, emptyDocument, type NodeFileDocument } from './fileTransaction.js';
import {
    MemoryStorageError,
    type ConsistencyNote,
    type FactStoragePort,
    type StorageStoreName,
    type StorageTransaction,
} from './types.js';

export type { NodeFileDocument } from './fileTransaction.js';

export interface NodeFileStorageAdapterOptions {
    /** Identity of this context for cross-context notes. Default: random. */
    contextId?: string;
    /**
     * Poll interval (ms) for foreign-change notification. 0 disables the
     * watcher. Default: 1000. Freshness is also checked synchronously before
     * every write transaction regardless of this interval.
     */
    watchIntervalMs?: number;
    /** Injectable fs (tests, sandboxed hosts). Default: node:fs. */
    fs?: NodeFs;
    /** Injectable path helpers. Default: node:path. */
    path?: NodePath;
    /** Cross-process write lock tuning (timeout, liveness probe). */
    lock?: FileLockOptions;
}

/** The subset of node:fs this adapter uses. */
export interface NodeFs {
    existsSync(path: string): boolean;
    readFileSync(path: string, encoding: 'utf8'): string;
    statSync(path: string): { mtimeMs: number; size: number };
    mkdirSync(path: string, opts: { recursive: boolean }): void;
    openSync(path: string, flags: string): number;
    writeFileSync(fd: number, data: string): void;
    fsyncSync(fd: number): void;
    closeSync(fd: number): void;
    renameSync(from: string, to: string): void;
    unlinkSync(path: string): void;
    copyFileSync(from: string, to: string): void;
}

export interface NodePath {
    dirname(p: string): string;
}

/** Load a module without a static specifier, so bundlers skip Node builtins. */
function dynamicImport(m: string): Promise<unknown> {
    // eslint-disable-next-line @typescript-eslint/no-implied-eval, no-new-func
    return new Function('m', 'return import(m)')(m) as Promise<unknown>;
}

export class NodeFileStorageAdapter implements FactStoragePort {
    readonly kind = 'node-file' as const;

    private readonly _fs: NodeFs;
    private readonly _path: NodePath;
    private readonly _contextId: string;
    private _doc: NodeFileDocument;
    private _stamp: { mtimeMs: number; size: number } | null = null;
    /** Transaction queue tail — all transacts serialize through this. */
    private _queue: Promise<unknown> = Promise.resolve();
    private _watchTimer: ReturnType<typeof setInterval> | null = null;
    private readonly _listeners = new Set<(note: ConsistencyNote) => void>();
    private readonly _watchIntervalMs: number;
    private readonly _lock: FileLock;

    private constructor(
        private readonly _filePath: string,
        fs: NodeFs,
        path: NodePath,
        opts: NodeFileStorageAdapterOptions,
    ) {
        this._fs = fs;
        this._path = path;
        this._contextId = opts.contextId ?? `node-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
        this._watchIntervalMs = opts.watchIntervalMs ?? 1000;
        this._lock = new FileLock(fs, `${_filePath}.lock`, opts.lock);
        this._doc = this._loadFromDisk();
    }

    /**
     * Opens (creating if necessary) the durable file at `filePath`. Async only
     * because Node builtins load indirectly; the document itself is read
     * synchronously before this resolves.
     */
    static async open(filePath: string, opts: NodeFileStorageAdapterOptions = {}): Promise<NodeFileStorageAdapter> {
        const fs = (opts.fs ?? (await dynamicImport('node:fs'))) as NodeFs;
        const path = (opts.path ?? (await dynamicImport('node:path'))) as NodePath;
        return new NodeFileStorageAdapter(filePath, fs, path, opts);
    }

    /** Identity of this context, stamped onto change notes and opIds. */
    get contextId(): string {
        return this._contextId;
    }

    /** The durable file backing this adapter. */
    get filePath(): string {
        return this._filePath;
    }

    // ── Boot / crash recovery ─────────────────────────────────────────────────

    /**
     * Reads the document from disk. Recovery order on a corrupt main file
     * (killed mid-rename is impossible, but a torn fsync on exotic filesystems
     * is not): try `<file>.tmp` (the last fully-fsynced candidate), then give up
     * with the corrupt file preserved at `<file>.corrupt` for forensics rather
     * than silently starting empty.
     */
    private _loadFromDisk(): NodeFileDocument {
        const fs = this._fs;
        const dir = this._path.dirname(this._filePath);
        if (dir && !fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

        const parsed = this._readDocument(this._filePath);
        if (parsed) return parsed;

        if (fs.existsSync(this._filePath)) {
            const tmp = this._readDocument(`${this._filePath}.tmp`);
            if (tmp) return tmp;
            try {
                fs.copyFileSync(this._filePath, `${this._filePath}.corrupt`);
                fs.unlinkSync(this._filePath);
            } catch { /* best-effort forensics */ }
        }
        return emptyDocument();
    }

    private _readDocument(path: string): NodeFileDocument | null {
        const fs = this._fs;
        if (!fs.existsSync(path)) return null;
        let raw: unknown;
        try {
            raw = JSON.parse(fs.readFileSync(path, 'utf8'));
        } catch {
            return null;
        }
        if (!raw || typeof raw !== 'object') return null;
        const doc = raw as Partial<NodeFileDocument>;
        if (doc.formatVersion !== 1 || !Array.isArray(doc.facts)) return null;
        const full: NodeFileDocument = {
            formatVersion: 1,
            facts: doc.facts,
            tombstones: Array.isArray(doc.tombstones) ? doc.tombstones : [],
            journal: Array.isArray(doc.journal) ? doc.journal : [],
            meta: doc.meta?.key === 'counters'
                ? doc.meta
                : { key: 'counters', seq: 0, revision: 0, opSeq: 0 },
        };
        try {
            const st = fs.statSync(path);
            this._stamp = { mtimeMs: st.mtimeMs, size: st.size };
        } catch { /* stamp stays null; next write re-checks */ }
        return full;
    }

    /** Re-reads the document when another process rewrote the file. */
    private _refreshIfChanged(): void {
        const fs = this._fs;
        if (!fs.existsSync(this._filePath)) {
            // Deleted underneath us: an empty document is the truth now.
            if (this._stamp) {
                this._stamp = null;
                this._doc = emptyDocument();
            }
            return;
        }
        let st: { mtimeMs: number; size: number };
        try {
            st = fs.statSync(this._filePath);
        } catch {
            return;
        }
        if (this._stamp && st.mtimeMs === this._stamp.mtimeMs && st.size === this._stamp.size) return;
        const doc = this._readDocument(this._filePath);
        if (doc) this._doc = doc;
    }

    /**
     * Atomic durable write: fsync the full document to `<file>.tmp`, then rename
     * over the live file. A crash leaves the old file intact; the tmp file is
     * the recovery source on next boot.
     */
    private _persist(): void {
        const fs = this._fs;
        const text = JSON.stringify(this._doc);
        const tmp = `${this._filePath}.tmp`;
        const fd = fs.openSync(tmp, 'w');
        try {
            fs.writeFileSync(fd, text);
            fs.fsyncSync(fd);
        } finally {
            fs.closeSync(fd);
        }
        fs.renameSync(tmp, this._filePath);
        try {
            const st = fs.statSync(this._filePath);
            this._stamp = { mtimeMs: st.mtimeMs, size: st.size };
        } catch { /* re-stamped on next change check */ }
    }

    // ── Port ──────────────────────────────────────────────────────────────────

    /**
     * Serialized unit of work. Read-write transactions take the cross-process
     * lock, refresh from disk (so we build on other processes' commits), run
     * `fn` against the live document, then persist atomically before
     * releasing. Any failure rolls the in-memory document back to the
     * pre-transaction snapshot — no partial commits, in memory or on disk.
     */
    transact<T>(
        _stores: StorageStoreName[],
        mode: 'readonly' | 'readwrite',
        fn: (tx: StorageTransaction) => Promise<T> | T,
    ): Promise<T> {
        const run = this._queue.then((): Promise<T> => mode === 'readwrite'
            ? this._lock.run(() => this._readWrite(fn))
            : Promise.resolve(fn(new FileTransaction(this._doc))));
        // Keep the queue alive across rejections.
        this._queue = run.catch(() => undefined);
        return run;
    }

    private async _readWrite<T>(fn: (tx: StorageTransaction) => Promise<T> | T): Promise<T> {
        this._refreshIfChanged();
        const snapshot = JSON.stringify(this._doc);
        let result: T;
        try {
            result = await fn(new FileTransaction(this._doc));
        } catch (err) {
            // A unit of work that threw leaves nothing behind, in memory either.
            this._doc = JSON.parse(snapshot) as NodeFileDocument;
            throw err;
        }
        try {
            this._persist();
        } catch (err) {
            // Roll back: the caller must not observe a committed-looking state
            // that never reached disk.
            this._doc = JSON.parse(snapshot) as NodeFileDocument;
            throw new MemoryStorageError(
                'STORAGE_ABORTED',
                `Failed to durably persist "${this._filePath}": ${err instanceof Error ? err.message : String(err)}`,
                err,
            );
        }
        return result;
    }

    // ── Cross-process consistency seam (M05) ──────────────────────────────────

    publish(note: ConsistencyNote): void {
        // Co-process subscribers get the note synchronously (subscribers filter
        // their own origin); foreign processes learn via the watcher.
        for (const l of this._listeners) l(note);
    }

    subscribe(listener: (note: ConsistencyNote) => void): () => void {
        this._listeners.add(listener);
        if (this._watchIntervalMs > 0 && !this._watchTimer) {
            let lastRevision = this._doc.meta.revision;
            const timer = setInterval(() => {
                const before = this._doc.meta.revision;
                this._refreshIfChanged();
                if (this._doc.meta.revision !== before && this._doc.meta.revision !== lastRevision) {
                    lastRevision = this._doc.meta.revision;
                    const note: ConsistencyNote = {
                        kind: 'facts-changed',
                        revision: this._doc.meta.revision,
                        origin: 'foreign',
                    };
                    for (const l of this._listeners) l(note);
                } else {
                    lastRevision = this._doc.meta.revision;
                }
            }, this._watchIntervalMs);
            const maybeUnref = timer as unknown as { unref?: () => void };
            if (typeof maybeUnref.unref === 'function') maybeUnref.unref();
            this._watchTimer = timer;
        }
        return () => {
            this._listeners.delete(listener);
            if (this._listeners.size === 0 && this._watchTimer) {
                clearInterval(this._watchTimer);
                this._watchTimer = null;
            }
        };
    }

    close(): void {
        if (this._watchTimer) {
            clearInterval(this._watchTimer);
            this._watchTimer = null;
        }
    }
}
