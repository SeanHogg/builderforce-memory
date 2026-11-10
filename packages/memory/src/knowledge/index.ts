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
    JudgementPort, AnalyzerPort, AnalyzerProgress, KnowledgeParser, KnowledgePack, StoredFact, StoredRelation, Scope, ScopeRecord, MeasuredWith,
    KnowledgeStore, DirectorySource, LiveOverlay, KnowledgeGraph, ReviewPlan, ReviewProgress, ReviewReport,
    ReviewArgs, BuildGraphInput, BuildGraphOptions, FindingsProvider, TraceStep, Trace, GraphOutline, OutlineFile, OutlineCommunity,
    GraphSystem, GraphApp, GraphReadSkip, CallSite,
} from './model.js';
export {
    SYMBOL_EDGE_KINDS, MAX_SYMBOL_EDGES_PER_SYMBOL, MAX_CALL_SITES_PER_EDGE, isSymbolEdgeKind, compareCallSites, dedupeSymbolEdges,
    type SymbolEdgeKind,
} from './symbolEdges.js';
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
export {
    coChangePartners, encodeCoChange, decodeCoChange, coChangeRecallAtK, applyCoChangeHidden, COCHANGE_MIN_SUPPORT, COCHANGE_MIN_LIFT,
    COCHANGE_MAX_COMMIT_FILES, COCHANGE_MAX_PARTNERS, COCHANGE_MAX_COMMITS, COCHANGE_BADGE, COCHANGE_HIDDEN_BADGE,
} from './coChange.js';
export type { CoChangeBadgeEntry, CoChangeCommit, CoChangeOptions, CoChangePartner, CoChangeRecall } from './coChange.js';
export { transitionsFromWeights, transitionsFromSequences, stationTransitions, transitionOf, mostLikelyWalk } from './transitions.js';
export type { StationTransition, TransitionRow, TransitionSource, TransitionTable } from './transitions.js';
export { pageRank, byImportance, areaImportance, PAGERANK_DAMPING, PAGERANK_ITERATIONS } from './importance.js';
export {
    loadPartitions, partitionOf, partitionOrder, verdictOf, globToRegExp, matchGlob, TOP_LEVEL_FOLDER,
} from './partitions.js';
export {
    traceGraph, toMermaid, DEFAULT_TRACE_EDGE_KINDS, DEFAULT_TRACE_DEPTH, DEFAULT_TRACE_STEPS, UNPARTITIONED_LANE,
} from './trace.js';
export type { LanedTrace, TraceOptions } from './trace.js';
export {
    computeFindings, computeFindingsWith, genericFindings, finding, findingId, partitionSeverity, requestKeyFor, components, CLONE_GROUP_BADGE, MAX_FINDING_NODES,
    GENERIC_FINDING_RULES, KNOWLEDGE_FINDING_RULES,
} from './findings.js';
// PRD 41 §B1/§B3/§B5: FileFacts v3, the composite extractor version, FactRule + the incremental evaluator, the fact graph.
export {
    FACT_KINDS, FACT_KIND_FIELDS, MATCHER_STRENGTH_ORDER, canonicalFileFacts, readFileFacts, isFileFactsV3, projectFileFacts,
} from './facts/fileFacts.js';
export { factKindHashes, factsHash } from './facts/factsHash.js';
export type {
    FactKind, FileFacts, SymbolFact, ImportFact, ExportFact, CallFact, ComplexityFact, EntryPointKind, EntryPointFact, CapabilityKind,
    CapabilityImportFact, MatcherStrength, MatcherUse, TestModifier, TestCaseFact, TestFacts,
} from './facts/fileFacts.js';
export { FACTS_EXTRACTOR_VERSION, factsExtractorTag, isFactsCurrent, extractorStampMeta } from './facts/extractorVersion.js';
export type { ExtractorFamily, ExtractorStamp } from './facts/extractorVersion.js';
export { defineRule, DEFAULT_RULE_HOPS, MAX_RULE_HOPS } from './rules/factRule.js';
export type { FactRule, RuleInput, RuleScope } from './rules/factRule.js';
export { evaluateRules, importNeighbours, derivedRowKey } from './rules/evaluateRules.js';
export type { RuleWorld, DerivedRow, DerivedRows, RuleTraceEntry, RuleEvaluation } from './rules/evaluateRules.js';
export { emptyFactGraph, updateFactGraph, buildFactGraph, factGraphDump, FACT_GRAPH_DUMP_VERSION } from './rules/factGraph.js';
export type { FactGraph, FactManifestEntry, FactChange, FactGraphOptions, FactGraphUpdate } from './rules/factGraph.js';
export { knowledgeFacts, knowledgeInputHash, runKnowledgeRules, packOfRef, KNOWLEDGE_INPUT } from './rules/knowledgeInput.js';
export type { KnowledgeFacts, KnowledgeRuleExtra, KnowledgeRule } from './rules/knowledgeInput.js';
// PRD 41 §B4: the fact store port (pure; hosts implement it) + §C5.1: router mounts and `use` guard semantics.
export { saveFactGraph, loadFactGraph, factStoreDump, sweepFactStore, memoryFactStore, FACT_STORE_LOAD_ATTEMPTS } from './rules/factStore.js';
export type {
    Awaitable, FactStorePort, FactStoreVersion, FactStoreManifest, FactManifestDelta, FactStoreSaveResult, StoredFactGraph,
    LoadFactGraphOptions, MemoryFactStore,
} from './rules/factStore.js';
export { factsJson, factsWireForm } from './facts/factsHash.js';
export { normalizeMountPrefix } from './facts/fileFacts.js';
export type { MountFact, UseFact } from './facts/fileFacts.js';
export { effectiveGuards, rootRouters, routePathCovers, joinRoutePath, MAX_MOUNT_DEPTH } from './facts/guards.js';
export type { ResolvedMount, EffectiveEntryPoint, EffectiveGuardsOptions, EffectiveGuardsResult, RootRouter, MountTarget } from './facts/guards.js';
export { refAnchor, type FindingAnchor } from './findingBuilder.js';
export {
    identityOf, defaultAnchorOf, composeAnchor, anchorWithSnippet, anchorKindOfRef, kindOfAnchor, normalizeSnippet, DEFAULT_RULE_VERSION,
} from './findingIdentity.js';
export type { AnchorKind, FindingIdentity } from './findingIdentity.js';
export { matchLineage, LINEAGE_MIN_JACCARD, LINEAGE_MIN_MEMBERS } from './lineage.js';
export type { Lineage, LineageChange, LineageEntry } from './lineage.js';
export {
    comparable, normalizeAnalysis, compareAnalysis, analysisMeta, identityOfIssue, ANALYSIS_ANCHOR_VERSION, ANALYSIS_TOOL_COUNT,
} from './comparability.js';
export type {
    ComparabilityMeta, ComparabilityReason, AnalysisLocation, AnalysisIssue, CodeAnalysisSnapshot, CodeAnalysisComparison,
} from './comparability.js';
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
export { contentHash, CONTENT_HASH_RULE, BASELINE_CONTENT_HASH_RULE, contentHashRuleOf } from './contentHash.js';
export { merkleTree, changedPaths, canonicalListing } from './merkle.js';
export type { MerkleTree, MerkleEntry, MerkleKind, ListingSkipReason } from './merkle.js';
export { normalizePath, isRootFilePath } from './refs.js';
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
    composePacks, ignoreMatcher, ignoreSourcesOf, isAssetPath, isBinaryBytes, isBinaryPath, isBinaryText, isDocumentPath, mediaTypeOfPath,
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
