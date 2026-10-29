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
    /** Sections left extracted-only because the budget ran out. */
    unjudged: number;
    invalidJudgements: number;
    factsWritten: number;
    relationsWritten: number;
    staleMarked: number;
    aborted: boolean;
    durationMs: number;
}

export interface ReviewArgs {
    source: DirectorySource;
    store: KnowledgeStore;
    packs: KnowledgePack[];
    manifest: IngestManifest;
    judgement?: JudgementPort;
    /** Near-duplicate claim pairs to compare / possible-same entities; from the knowledge index. */
    similar?: (ids: string[]) => Promise<SimilarPair[]>;
    budget: { modelCalls: number; ms: number };
    /** *Continue*: the plan the person confirmed. Reused without a re-walk; `changed` is re-read from the manifest. */
    plan?: PlanDetail;
    signal?: AbortSignal;
    onProgress?(p: ReviewProgress): void;
}
