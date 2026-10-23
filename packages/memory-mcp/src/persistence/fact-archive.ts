/**
 * FactArchive — where a compacted fact's full original lives once its body is a stub.
 *
 * The archive is deliberately NOT a field on the stored entry: the store snapshot is
 * rebuilt from the store on every write, rewritten in place by other processes, and
 * capped/evicted at {@link DEFAULT_MAX_ENTRIES}; an archived original riding inside
 * it would be lost to any one of those. A separate sidecar (`memory.archive.json`,
 * see `defaultArchiveFile`) is only ever merged into, never pruned, so an evicted or
 * forgotten stub still has its original.
 *
 * Two implementations, one contract: in-memory (a backend with no disk mirror) and
 * file-backed (shared across processes through {@link SharedJsonFile}, with its
 * rotating backups and freshness check).
 */

import { mergeArchive, parseArchive, type ArchivedFact } from "../compaction.js";
import { SharedJsonFile, type SharedFileFs } from "./shared-json-file.js";

function isJson(text: string): boolean {
    try {
        JSON.parse(text);
        return true;
    } catch {
        return false;
    }
}

export interface FactArchive {
    /** The archived original for `key`, if one exists. */
    get(key: string): ArchivedFact | undefined;
    /** Every row, read once — for callers resolving many keys in one pass. */
    all(): Readonly<Record<string, ArchivedFact>>;
    /** Add or replace archive rows. Durable before it returns. */
    putMany(facts: ArchivedFact[]): void;
}

/** Process-local archive — the default for a backend with no disk mirror. */
export class MemoryFactArchive implements FactArchive {
    private readonly rows: Record<string, ArchivedFact> = Object.create(null) as Record<string, ArchivedFact>;

    get(key: string): ArchivedFact | undefined {
        return this.rows[key];
    }

    all(): Readonly<Record<string, ArchivedFact>> {
        return this.rows;
    }

    putMany(facts: ArchivedFact[]): void {
        for (const f of facts) this.rows[f.key] = f;
    }
}

/**
 * Archive mirrored to a JSON file several processes share. Re-reads the file when it
 * changed underneath this process; every write merges into the latest on-disk text,
 * so two writers add to one archive rather than overwriting each other's rows.
 */
export class FileFactArchive implements FactArchive {
    private rows: Record<string, ArchivedFact> = {};
    private text: string | null = null;
    private readonly file: SharedJsonFile;

    constructor(path: string, fs: SharedFileFs) {
        this.file = new SharedJsonFile(path, fs);
        this.refresh();
    }

    get(key: string): ArchivedFact | undefined {
        this.refresh();
        return this.rows[key];
    }

    all(): Readonly<Record<string, ArchivedFact>> {
        this.refresh();
        return this.rows;
    }

    putMany(facts: ArchivedFact[]): void {
        if (facts.length === 0) return;
        this.refresh();
        // An archive that exists but will not parse is refused, never overwritten:
        // merging into "{}" would drop every row it holds. The throw reaches the
        // compactor BEFORE any stub is written, so nothing is compacted either.
        if (this.text && this.text.trim() && !isJson(this.text)) {
            throw new Error(`fact archive ${this.file.path} is unreadable; refusing to overwrite it`);
        }
        const next = mergeArchive(this.text, facts);
        this.file.write(next);
        this.text = next;
        this.rows = parseArchive(next);
    }

    private refresh(): void {
        const changed = this.file.readIfChanged();
        if (!changed) return;
        this.text = changed.text;
        this.rows = parseArchive(changed.text);
        changed.accept();
    }
}
