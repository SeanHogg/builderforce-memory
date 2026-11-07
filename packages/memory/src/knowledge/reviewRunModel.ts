/**
 * knowledge/reviewRunModel.ts — the vocabulary of a watchable review (PRD 41 A8): phases, the
 * per-file step events a review emits, and the failures it records instead of throwing.
 */

/** `mapped` (PRD 41 A10): the mechanical pass is over and the map is complete; judging, if any is owed, follows. */
export type ReviewPhase = 'plan' | 'extract' | 'resolve' | 'judge' | 'compare' | 'write' | 'index' | 'mapped' | 'done';

export type ReviewStepOutcome = 'ok' | 'cached' | 'skipped' | 'failed';

/**
 * One file in one phase. `cached` = reused prior knowledge or an extract-cache hit; `skipped` = binary
 * or too large; `failed` carries `reason`. `lane` is the worker/lane id (0..width-1) where known.
 */
export interface ReviewStep {
    path: string;
    phase: ReviewPhase;
    ms: number;
    outcome: ReviewStepOutcome;
    reason?: string;
    lane?: number;
}

/** A file a review could not finish: its prior knowledge and manifest hash are kept, so the next review retries it. */
export interface ReviewFailure {
    path: string;
    phase: ReviewPhase;
    reason: string;
}

/** Lane ids 0..width-1 for in-flight work: `take` the lowest free one, `free` it when the work settles. */
export class Lanes {
    private readonly busy = new Set<number>();
    private readonly width: number;

    constructor(width: number) { this.width = width; }

    take(): number {
        // More work in flight than lanes means the stage that bounds it is broken: say so, never share a lane.
        if (this.busy.size >= this.width) throw new Error(`Lanes: invariant violated — all ${this.width} lanes are busy`);
        let id = 0;
        while (this.busy.has(id)) id += 1;
        this.busy.add(id);
        return id;
    }

    free(id: number): void { this.busy.delete(id); }
}

/** A thrown value as a short, single-line reason. */
export const reasonOf = (error: unknown): string => {
    const text = error instanceof Error ? error.message : String(error);
    return text.split(/\r?\n/)[0]!.slice(0, 300) || 'unknown error';
};
