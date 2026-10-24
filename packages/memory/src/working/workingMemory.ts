/**
 * Short-term memory for one run: what it has read, searched and changed so far.
 *
 * ## Why the brief exists
 *
 * A model's reasoning is stripped from the transcript it reads next (models copy
 * reasoning markup instead of calling tools), and a bounded working window compacts old
 * tool results away. So by turn forty a run cannot see what it concluded at turn ten,
 * and re-derives the whole architecture in every turn's reasoning — measured: 8–10k
 * completion tokens a turn for two tool calls each, the same files re-read a dozen
 * times. {@link WorkingMemory.brief} is the run's own record, handed to EVERY turn on
 * that request only (never into the transcript, so it neither grows history nor breaks
 * prompt caching), so the model reasons from "here is what I have" instead of from
 * scratch.
 *
 * Record SUCCESSFUL observations only: a memory claiming a file was written when the
 * write failed grounds every later turn — and run — on a lie.
 */

import { rankByCount } from './rank.js';
import type { Observation, WorkingMemorySnapshot } from './types.js';

/**
 * Below this many observations the run's own results are still in its window, so a
 * restatement only costs tokens. Past it, compaction starts dropping them.
 */
const BRIEF_MIN_OBSERVATIONS = 10;
const BRIEF_MAX_FILES = 25;
const BRIEF_MAX_SEARCHES = 15;

export class WorkingMemory {
  private task: string | undefined;
  private readonly created: string[] = [];
  private readonly edited: string[] = [];
  private readonly deleted: string[] = [];
  private readonly read: string[] = [];
  private readonly searched: string[] = [];
  private observations = 0;

  /** What the run was asked — the consolidation names it, so a map is a lesson, not a list. */
  setTask(task: string): void {
    const asked = task.replace(/\s+/g, ' ').trim();
    if (asked) this.task = asked;
  }

  observe(o: Observation): void {
    this.observations++;
    if (o.kind === 'searched') {
      if (o.query.trim()) this.searched.push(o.query.trim());
      return;
    }
    const p = o.path.trim();
    if (!p) return;
    this[o.kind].push(p);
  }

  /** Observations recorded so far (searches and file events alike). */
  get size(): number {
    return this.observations;
  }

  get snapshot(): WorkingMemorySnapshot {
    return {
      ...(this.task ? { task: this.task } : {}),
      created: this.created,
      edited: this.edited,
      deleted: this.deleted,
      read: this.read,
      searched: this.searched,
    };
  }

  /**
   * What this run has ALREADY established, to put in front of its next turn — or `null`
   * until it has done enough for a restatement to be worth its tokens. `observations`
   * lets a host count its own activity (every tool call, not only memory-relevant ones).
   * Hosts deliver it inside the per-request volatile tail (`/handoff` `volatileTail`), so
   * it never becomes the cached prefix nor "the latest user directive".
   */
  brief(opts: { observations?: number } = {}): string | null {
    if ((opts.observations ?? this.observations) < BRIEF_MIN_OBSERVATIONS) return null;
    const changed = [...new Set([...this.created, ...this.edited])];
    const changedSet = new Set(changed);
    const deleted = [...new Set(this.deleted)];
    const reads = rankByCount(this.read).filter(([p]) => !changedSet.has(p));
    const searches = rankByCount(this.searched);
    if (changed.length === 0 && deleted.length === 0 && reads.length === 0 && searches.length === 0) return null;
    const lines = [
      '## What this run has already established',
      'Your own record so far, kept by the runtime — not a new request. Build on it instead of re-deriving it: earlier results may have been compressed out of your context, but what you concluded from them still holds. Re-read a file only when it may have changed since (a command rewrote it) or you need a part you have not seen; re-running a search to verify an edit is fine. If you already know where the change goes, make it.',
    ];
    if (changed.length) lines.push('', 'Changed:', ...changed.slice(0, BRIEF_MAX_FILES).map((p) => `- ${p}`));
    if (deleted.length) lines.push('', 'Deleted:', ...deleted.slice(0, BRIEF_MAX_FILES).map((p) => `- ${p}`));
    if (reads.length) {
      lines.push('', 'Read:', ...reads.slice(0, BRIEF_MAX_FILES).map(([p, n]) => `- ${p}${n > 1 ? ` (${n}×)` : ''}`));
      if (reads.length > BRIEF_MAX_FILES) lines.push(`- … ${reads.length - BRIEF_MAX_FILES} more`);
    }
    if (searches.length) {
      lines.push('', 'Searched:', ...searches.slice(0, BRIEF_MAX_SEARCHES).map(([s, n]) => `- ${s}${n > 1 ? ` (${n}×)` : ''}`));
    }
    return lines.join('\n');
  }
}
