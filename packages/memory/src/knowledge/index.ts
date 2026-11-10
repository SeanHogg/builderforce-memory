/**
 * `@seanhogg/builderforce-memory/knowledge` — the knowledge graph: facts connected to each
 * other and to the sources they came from (PRD 38, Builderforce.ai/docs/specs/38-prd-knowledge-graph.md).
 *
 * Pure TypeScript: runs in the browser, a Worker and Node. Hosts supply ports only — a
 * DirectorySource, a KnowledgeStore, a JudgementPort — never graph logic.
 */

export type {
    NodeRef, Origin, StoredOrigin, KNode, EdgeVerdict, KEdge, Extracted, ExtractContext, Finding, FindingRule,
    PartitionRule, PartitionRegistry, JudgementSpec, JudgementRequest, JudgedClaim, JudgedEntity, JudgementResult,
    JudgementPort, AnalyzerPort, AnalyzerProgress, KnowledgeParser, KnowledgePack, StoredFact, StoredRelation, Scope, ScopeRecord,
    KnowledgeStore, DirectorySource, LiveOverlay, KnowledgeGraph, ReviewPlan, ReviewProgress, ReviewReport,
    ReviewArgs, BuildGraphInput, BuildGraphOptions, TraceStep, Trace, GraphOutline, OutlineFile, OutlineCommunity,
    GraphSystem, GraphApp, GraphReadSkip,
} from './model.js';
export {
    buildOutline, outlineOf, outlineRef, OUTLINE_MAX_FILES, OUTLINE_MAX_LINKS, OUTLINE_BADGE_KEYS, OUTLINE_VERDICTS, OUTLINE_FLAG_CYCLE,
} from './outline.js';
export { EXTERNAL_SERVICE_IDS, type ExternalServiceId } from './externals.js';

export { buildGraph, graphDigest, oneLineLabel, DEFAULT_MAX_NODES, MAX_NODES_CAP } from './build.js';
export { placeInSystem, crossesSystems, type SystemPlacement } from './systems.js';
export { truncateGraph, TRUNCATION_HARD_CAP, UNANCHORED_AREA } from './truncate.js';
export type { TruncateOptions } from './truncate.js';
export {
    detectCommunities, applyCommunities, communityTable, placementOf, COMMUNITY_EDGE_KINDS, COMMUNITY_LOUVAIN_MAX_NODES,
    COMMUNITY_MAX_PASSES, COMMUNITY_MAX_ITERATIONS, COMMUNITY_LP_ITERATIONS,
} from './communities.js';
export type { CommunityMembership, CommunityMethod, CommunityResult, NodePlacement } from './communities.js';
export { pageRank, byImportance, areaImportance, PAGERANK_DAMPING, PAGERANK_ITERATIONS } from './importance.js';
export {
    loadPartitions, partitionOf, partitionOrder, verdictOf, globToRegExp, matchGlob, TOP_LEVEL_FOLDER,
} from './partitions.js';
export {
    traceGraph, toMermaid, DEFAULT_TRACE_EDGE_KINDS, DEFAULT_TRACE_DEPTH, DEFAULT_TRACE_STEPS, UNPARTITIONED_LANE,
} from './trace.js';
export type { LanedTrace, TraceOptions } from './trace.js';
export {
    computeFindings, genericFindings, finding, findingId, partitionSeverity, requestKeyFor, components, CLONE_GROUP_BADGE, MAX_FINDING_NODES,
} from './findings.js';
export { GOD_MODULE_LIMITS, GOD_MODULE_KIND, godModuleFindings } from './godModule.js';
export type { GodModuleLimits } from './godModule.js';
export {
    resolveEntities, knownKeysOf, possibleSameFindings, normalizeKey, entityName, entityKey,
} from './resolve.js';
export type { ResolvedEntity } from './resolve.js';
export { toCards } from './cards.js';
export { planReview, planDetail, refreshPlan, AVG_SECTION_BYTES } from './plan.js';
export type { PlanOptions, PlannedFile, PlanDetail } from './plan.js';
export type { ExtractorState } from './owedFiles.js';
export { contentHash, CONTENT_HASH_RULE } from './contentHash.js';
export { merkleTree, changedPaths } from './merkle.js';
export type { MerkleTree, MerkleEntry, MerkleKind } from './merkle.js';
export { reviewDirectory, assertStoredBatch } from './review.js';
export { orderForJudging } from './judgePriority.js';
export type { JudgeSignals } from './judgePriority.js';
export { extractFile, extractFileDetailed, knownKeysOfEntities } from './extractFile.js';
export type { ExtractFileArgs, FileExtraction } from './extractFile.js';
export { ExtractFailed, extractJob, jobOf } from './extractJob.js';
export type { ExtractJob, ExtractJobResult, ExtractPool } from './extractJob.js';
export { putExtract, putNoExtract } from './extractFile.js';
export { createModelLimiter, isRateLimit } from './modelLimiter.js';
export type { ModelLimiter } from './modelLimiter.js';
export { reasonOf } from './reviewRunModel.js';
export type { ReviewFailure, ReviewPhase, ReviewStep, ReviewStepOutcome } from './reviewRunModel.js';
export { startRun, applyRunEvent, runRate, runEta, slowestSteps, failuresOf, failureTotal, REVIEW_RUN_STEP_RING, REVIEW_RUN_FAILURE_CHUNK } from './reviewRun.js';
export type { ReviewFailureChunks, ReviewRun, ReviewRunEvent, ReviewRunKind, ReviewRunLane, ReviewRunOutcome, ReviewRunPhase } from './reviewRun.js';
export {
    validateExtract, validateCompare, clampQuote, clampClaim, MAX_QUOTE_LINES, MAX_QUOTE_CHARS, MAX_CLAIM_TEXT_CHARS,
    DEFAULT_REVIEW_MODEL_CALLS, DEFAULT_REVIEW_JUDGE_CONCURRENCY, REVIEW_FACT_SOURCE,
} from './judge.js';
export {
    composePacks, ignoreMatcher, ignoreSourcesOf, isAssetPath, isBinaryPath, isBinaryText, isDocumentPath, mediaTypeOfPath,
    resolveDocument, resolveRelativePath,
    DEFAULT_MAX_FILE_BYTES, EXTRACTOR_VERSION, filesManifestKey, extractorManifestKey, judgedManifestKey, compareManifestKey,
} from './sources.js';
export {
    srcRef, splitSourceRef, FALLBACK_PACK_ID, pathOfRef, fileRefOf, factRef, factKeyOf, slugOf, canonicalJson, compareStr, nameResolver, aboutTargets,
} from './refs.js';
export { DOCS_PACK, DOCS_PACK_ID, brokenLinkRule, orphanDocumentRule, contradictionRule } from './packs/docs.js';
export { frontMatterOf, SECTION_INDEX_BADGE } from '../ingest/docsExtract.js';
export { LocalSnapshotStore, localSnapshotStore, canonicalDump } from './stores/localSnapshotStore.js';
export type {
    SnapshotPersistence, KnowledgeSnapshot, LocalSnapshotStoreOptions, ExtractedLogPersistence, RowPersistence, RowChange,
} from './stores/localSnapshotStore.js';

// The knowledge index (search, expand, near-duplicate pairs) ships on the same subpath.
export * from '../retrieval/knowledge-index/index.js';
