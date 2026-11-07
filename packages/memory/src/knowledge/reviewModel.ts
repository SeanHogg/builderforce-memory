/**
 * knowledge/reviewModel.ts — the review's port and reports (PRD 38 §4.1, §4.3): where files
 * come from, the confirm chip's plan, progress and the final report.
 */

import type { IngestManifest, SourceDocument } from '../ingest/types.js';
import type { SimilarPair } from '../retrieval/knowledge-index/types.js';
import type { JudgementPort, KnowledgePack, KnowledgeStore } from './model.js';
import type { PlanDetail } from './plan.js';
import type { ExtractPool } from './extractJob.js';
import type { ReviewFailure, ReviewPhase, ReviewStep } from './reviewRunModel.js';

/** Where files come from: one adapter per host. */
export interface DirectorySource {
    rootId: string;
    title: string;
    list(): AsyncIterable<{ path: string; hash: string; size: number }>;
    read(path: string): Promise<SourceDocument>;
    watch?(cb: (paths: string[]) => void): () => void;
}

export interface ReviewPlan {
    rootId: string;
    files: number;
    byPack: Record<string, number>;
    unchanged: number;
    changed: string[];
    removed: string[];
    skipped: Array<{ path: string; reason: 'ignored' | 'binary' | 'too-large' | 'no-pack' }>;
    /** ≈ judgement calls for changed files (sections of judged kinds). */
    estimatedModelCalls: number;
}

export interface ReviewProgress {
    phase: ReviewPhase;
    done: number;
    total: number;
    path?: string;
}

export interface ReviewReport {
    rootId: string;
    digest: string;
    extracted: number;
    judged: number;
    /** Sections and compare pairs left for *Continue* (budget ran out, or a model call failed). */
    unjudged: number;
    invalidJudgements: number;
    factsWritten: number;
    relationsWritten: number;
    staleMarked: number;
    /**
     * Files whose read threw: skipped, not retired — their facts, anchors and manifest hash are
     * untouched, so the next review tries them again. A file that is truly gone shows up in
     * the next fresh plan's `removed`.
     */
    readErrors: number;
    /** Files whose extraction threw: handled like a read error (knowledge and manifest hash kept, retried next review). */
    extractErrors: number;
    /** Every file this run could not finish (read, extract, judge), with why. */
    failures: ReviewFailure[];
    aborted: boolean;
    durationMs: number;
}

export interface ReviewArgs {
    source: DirectorySource;
    store: KnowledgeStore;
    packs: KnowledgePack[];
    manifest: IngestManifest;
    judgement?: JudgementPort;
    /**
     * Near-duplicate claim pairs to compare / possible-same entities; from the knowledge index.
     * Called with the pending claim refs (sorted) and, as an optional second argument, each
     * ref's claim text (`subject predicate object`, the form the compare uses) — so a host
     * need not re-read every fact to recover what the review already holds. A port that takes
     * only `ids` keeps working.
     */
    similar?: (ids: string[], texts?: ReadonlyMap<string, string>) => Promise<SimilarPair[]>;
    budget: { modelCalls: number; ms: number };
    /** *Continue*: the plan the person confirmed. Reused without a re-walk; `changed` is re-read from the manifest. */
    plan?: PlanDetail;
    /** *Re-review all*: every file is reviewed whatever its hash (see `PlanOptions.force`). Ignored with `plan`. */
    force?: boolean;
    signal?: AbortSignal;
    /**
     * Files read + extracted ahead of judging (1..32, default 8). With an `extractPool` this is the pool's width.
     * Writes always stay in file order.
     */
    concurrency?: number;
    /**
     * Files judged at once (1..16, default 4, never wider than the read-ahead). One ordered writer applies the
     * results in plan order, so the stored bytes equal those of `concurrency: 1, judgeConcurrency: 1`.
     */
    judgeConcurrency?: number;
    /**
     * The most model calls in flight across ALL files of the review (default 6): `judgeConcurrency` parallelizes
     * files without multiplying the provider's load. A rate-limit answer halves it (minimum 1); it grows back by 1
     * per 20 successful calls.
     */
    maxModelCalls?: number;
    /**
     * Extracts a changed file whose extract is not cached — e.g. a `worker_threads` pool. Default: in-thread.
     * With a pool the main thread never parses; the pool returns a file's sections when it will be judged.
     * A result without the file's exact `sourceRef` fails that file's extraction (prior knowledge kept).
     */
    extractPool?: ExtractPool;
    /** One event per file per phase, as it happens (PRD 41 A8). */
    onStep?(step: ReviewStep): void;
    onProgress?(p: ReviewProgress): void;
}
