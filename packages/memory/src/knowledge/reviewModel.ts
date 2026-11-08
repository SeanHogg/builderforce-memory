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
    /** `signal` aborts the read when the review gives up on it (a per-job timeout, a stop); a source that cannot cancel may ignore it. */
    read(path: string, signal?: AbortSignal): Promise<SourceDocument>;
    watch?(cb: (paths: string[]) => void): () => void;
    /**
     * The function that produced the `hash` of `list()` entries, over a text file's content. When given, the review hashes the
     * text it read (once) and a file whose bytes no longer match the planned hash fails with "changed since plan" and stays owed.
     * Optional: a source whose hash is not a pure function of the text (or that cannot say) is not checked in-thread.
     * The `path` lets a host whose read() transforms bytes (document text) return the listing hash for that path.
     * A host that declares `hashOf(text)` still works: the extra argument is ignored.
     */
    hashOf?(text: string, path: string): string;
    /**
     * The read deadline for `path`, in ms, when it differs from the default (5 s). A finite number above 0 is used for that path's read;
     * anything else keeps the default. For slow transforming reads such as document extraction.
     */
    readTimeoutMs?(path: string): number | undefined;
}

export interface ReviewPlan {
    rootId: string;
    files: number;
    byPack: Record<string, number>;
    /**
     * Files in `files` that are assets (binary, never read): stored as a file node with `mediaType` and `size`, not counted in `byPack`.
     * `files` includes them (the plan diffs them like any file), so a host's ETA and "files to read" figures subtract `assets`
     * (absent on a plan from an older host or built by hand: read it as 0).
     */
    assets?: number;
    unchanged: number;
    changed: string[];
    removed: string[];
    skipped: Array<{ path: string; reason: 'ignored' | 'too-large' | 'no-pack' }>;
    /** ≈ judgement calls for changed files (sections of judged kinds). */
    estimatedModelCalls: number;
    /** The file budget of the judge pass this run will run after the map is complete (PRD 41 A10); set by the host, so the confirm chip states it. */
    judgeFiles?: number;
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
    /**
     * Files still owed model judgement after this run (PRD 41 A10): in a `mechanical` run every file of a judged pack
     * with a section to judge; otherwise the files a budget cut or a failed call left. Plan order.
     */
    awaitingJudge: string[];
    /** How many files are awaiting judgement: the ONLY form that leaves memory — hosts persist and send this, never the path list. */
    awaitingJudgeCount?: number;
    /** Pass 2 (`only`): files of the slice judged to completion. */
    judgedFiles?: number;
    /** Judging was stopped for this run after consecutive provider failures (A10 rule 9); the files stay owed. */
    judgeStopped?: string;
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
    /**
     * Pass 1 of a two-pass review (PRD 41 A10): read, extract, write — no model call. A file whose pack asks for
     * judgement and has sections to judge stays OWED (manifest hash unchanged) and is listed in `report.awaitingJudge`;
     * the map is complete without it. A later run without `mechanical` (pass 2) judges it.
     */
    mechanical?: boolean;
    /** Restrict the run to these changed paths (pass 2: the prioritized, budgeted slice). Other changed files are untouched and stay owed. */
    only?: readonly string[];
    /**
     * With an `extractPool`: the pool reads each file itself (PRD 41 A10 rule 8) — the job carries the path and the planned hash,
     * the main thread neither reads nor copies the text, and a binary file comes back as `binary`. The pool must support `readByPool` jobs.
     */
    deferRead?: boolean;
    /** Files read and extracted ahead of the writer (default: the pool's width times 4, 32..64, with a pool; else `concurrency`). Wider than the pool, so workers never wait for the writer. */
    readAhead?: number;
    /** Per-job ceilings in ms for the extract path (PRD 41 A10): reads 5000, extract (pool dispatch included) 10000. A job over its ceiling is a failed step with reason `timeout`. */
    timeouts?: { readMs?: number; extractMs?: number };
    /** One event per file per phase, as it happens (PRD 41 A8). */
    onStep?(step: ReviewStep): void;
    onProgress?(p: ReviewProgress): void;
}
