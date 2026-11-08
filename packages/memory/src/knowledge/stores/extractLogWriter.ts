/**
 * knowledge/stores/extractLogWriter.ts — the write side of the extract cache's append-only log (PRD 41 A10 rule 7), moved
 * out of {@link LocalSnapshotStore}. It owns what is not yet appended and whether the log's last line is torn; the store
 * owns WHEN to flush (inside `persistIfDirty`). No behaviour of its own beyond the log port.
 */

import type { Extracted } from '../model.js';
import type { ExtractedLogPersistence } from './localSnapshotStore.js';
import { parseLog } from './snapshotFormat.js';

export class ExtractLogWriter {
    /** Entries not yet appended. */
    private pending = new Map<string, Extracted[]>();
    /** The log's last line is torn (no newline), so the next append starts on a fresh line. */
    private torn = false;

    constructor(private readonly port: ExtractedLogPersistence) {}

    read(): Promise<string | undefined> { return this.port.read(); }

    /** Queues an entry for the next {@link flush}. */
    queue(hash: string, entries: Extracted[]): void { this.pending.set(hash, entries); }

    /** Fills `into` from the log's text (last line per hash wins) and compacts the log when dead bytes exceed live bytes. */
    async load(text: string | undefined, into: Map<string, Extracted[]>): Promise<void> {
        const parsed = parseLog(text);
        for (const [h, e] of parsed.entries) into.set(h, e);
        this.torn = !!text && !text.endsWith('\n');
        // Needs the port's `replace`; a failure is ignored (the log stays correct, only larger). Another window appending between
        // this read and the replace loses those lines from the cache — a cache miss, re-extracted next review.
        if (!this.port.replace || parsed.deadBytes <= parsed.liveBytes) return;
        try {
            await this.port.replace([...parsed.liveLines.values()].join(''));
            this.torn = false;
        } catch { /* keep the uncompacted log */ }
    }

    /**
     * Appends the not-yet-logged entries as ONE `append` of complete lines. A failed append keeps them pending and marks the
     * log torn (it may hold a partial line), so the next append starts on a fresh line.
     */
    async flush(): Promise<void> {
        if (this.pending.size === 0) return;
        const batch = this.pending;
        this.pending = new Map();
        let text = this.torn ? '\n' : '';
        for (const [h, e] of batch) text += `${JSON.stringify({ h, e })}\n`;
        try {
            await this.port.append(text);
            this.torn = false;
        } catch (err) {
            this.torn = true;
            for (const [h, e] of batch) if (!this.pending.has(h)) this.pending.set(h, e);
            throw err;
        }
    }
}
