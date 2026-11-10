/**
 * knowledge/stores/extractLogWriter.ts — the write side of the extract cache's append-only log (PRD 41 A10 rule 7), moved
 * out of {@link LocalSnapshotStore}. It owns what is not yet appended and whether the log's last line is torn; the store
 * owns WHEN to flush (inside `persistIfDirty`). No behaviour of its own beyond the log port.
 */

import type { Extracted } from '../model.js';
import type { ExtractedLogPersistence } from './localSnapshotStore.js';
import type { RowMemory } from './rowMemory.js';
import { compactedLogLine, parseLog } from './snapshotFormat.js';

export class ExtractLogWriter {
    /** Entries not yet appended. */
    private pending = new Map<string, Extracted[]>();
    /** The log's last line is torn (no newline), so the next append starts on a fresh line. */
    private torn = false;

    /** `isLive(hash)`: the body is still in the cache (not evicted), so a failed append may put it back in line. */
    constructor(private readonly port: ExtractedLogPersistence, private readonly isLive: (hash: string) => boolean = () => true) {}

    read(): Promise<string | undefined> { return this.port.read(); }

    /** Queues an entry for the next {@link flush}. */
    queue(hash: string, entries: Extracted[]): void { this.pending.set(hash, entries); }

    /** An evicted body that was never appended is not appended at all. */
    unqueue(hash: string): void { this.pending.delete(hash); }

    /**
     * Fills `mem` by REPLAYING the log's lines in order, exactly as the puts happened (last line per hash wins; a put claims only the
     * entries it added or changed, and evicts what is then neither current nor previous for any source, extractRefs.ts), then compacts
     * the log when dead bytes (superseded lines, evicted bodies, torn lines) exceed live bytes. A compacted line carries the body's
     * reference state (`c`/`p`, snapshotFormat.ts), so the replay of a compacted log RESTORES the state instead of re-deriving it
     * from a put order compaction no longer has.
     */
    async load(text: string | undefined, mem: RowMemory): Promise<void> {
        const parsed = parseLog(text, (h, e, state) => { if (state) mem.restoreBody(h, e, state); else mem.putBody(h, e); });
        const kept = [...parsed.liveLines].filter(([h]) => mem.extracted.has(h));
        const liveBytes = kept.reduce((n, [, l]) => n + l.length, 0);
        // A body that names no source (no reference state) keeps its line byte for byte; any other carries its state.
        const live = kept.map(([h, line]) => {
            const state = mem.bodyState(h);
            return state.current.length + state.previous.length === 0 ? line : compactedLogLine(h, mem.extracted.get(h)!, state);
        });
        const deadBytes = Math.max(0, (text?.length ?? 0) - liveBytes);
        this.torn = !!text && !text.endsWith('\n');
        // Needs the port's `replace`; a failure keeps the uncompacted log (still correct, only larger) and is reported to `onError`.
        // Another window appending between this read and the replace loses those lines from the cache — a cache miss, re-extracted next review.
        if (!this.port.replace || deadBytes <= liveBytes) return;
        try {
            await this.port.replace(live.join(''));
            this.torn = false;
        } catch (err) {
            this.onError?.(err);
        }
    }

    /** Told about a compaction that failed (the load itself succeeds). */
    onError: ((err: unknown) => void) | undefined;

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
            // An entry evicted while the append was in flight is not put back: appended later, it would supersede its successor.
            for (const [h, e] of batch) if (!this.pending.has(h) && this.isLive(h)) this.pending.set(h, e);
            throw err;
        }
    }
}
