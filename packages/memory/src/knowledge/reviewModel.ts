/**
 * knowledge/reviewModel.ts — the review's port and reports (PRD 38 §4.1, §4.3): where files
 * come from, the confirm chip's plan, progress and the final report.
 */

import type { IngestManifest, SourceDocument } from '../ingest/types.js';
import type { SimilarPair } from '../retrieval/knowledge-index/types.js';
import type { JudgementPort, KnowledgePack, KnowledgeStore } from './model.js';
import type { PlanDetail } from './plan.js';

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
    phase: 'plan' | 'extract' | 'resolve' | 'judge' | 'write' | 'index' | 'done';
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
    onProgress?(p: ReviewProgress): void;
}
