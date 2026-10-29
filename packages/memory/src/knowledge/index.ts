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
    JudgementPort, AnalyzerPort, KnowledgeParser, KnowledgePack, StoredFact, StoredRelation, Scope, ScopeRecord,
    KnowledgeStore, DirectorySource, LiveOverlay, KnowledgeGraph, ReviewPlan, ReviewProgress, ReviewReport,
    ReviewArgs, BuildGraphInput, BuildGraphOptions, TraceStep, Trace,
} from './model.js';

export { buildGraph, graphDigest, oneLineLabel, DEFAULT_MAX_NODES, MAX_NODES_CAP } from './build.js';
export { truncateGraph, TRUNCATION_HARD_CAP, UNANCHORED_AREA } from './truncate.js';
export type { TruncateOptions } from './truncate.js';
export { pageRank, byImportance, areaImportance, PAGERANK_DAMPING, PAGERANK_ITERATIONS } from './importance.js';
export {
    loadPartitions, partitionOf, partitionOrder, verdictOf, globToRegExp, matchGlob, TOP_LEVEL_FOLDER,
} from './partitions.js';
export {
    traceGraph, toMermaid, DEFAULT_TRACE_EDGE_KINDS, DEFAULT_TRACE_DEPTH, DEFAULT_TRACE_STEPS, UNPARTITIONED_LANE,
} from './trace.js';
export type { LanedTrace, TraceOptions } from './trace.js';
export {
    computeFindings, genericFindings, finding, findingId, requestKeyFor, components, CLONE_GROUP_BADGE, MAX_FINDING_NODES,
} from './findings.js';
export {
    resolveEntities, knownKeysOf, possibleSameFindings, normalizeKey, entityName, entityKey,
} from './resolve.js';
export type { ResolvedEntity } from './resolve.js';
export { toCards } from './cards.js';
export { planReview, planDetail, refreshPlan, AVG_SECTION_BYTES } from './plan.js';
export type { PlanOptions, PlannedFile, PlanDetail } from './plan.js';
export { reviewDirectory, assertStoredBatch } from './review.js';
export { extractFile, extractFileDetailed, knownKeysOfEntities } from './extractFile.js';
export type { ExtractFileArgs, FileExtraction } from './extractFile.js';
export {
    validateExtract, validateCompare, clampQuote, MAX_QUOTE_LINES, MAX_QUOTE_CHARS, DEFAULT_REVIEW_MODEL_CALLS, REVIEW_FACT_SOURCE,
} from './judge.js';
export {
    composePacks, ignoreMatcher, ignoreSourcesOf, isBinaryPath, isBinaryText, isDocumentPath, mediaTypeOfPath,
    resolveDocument, resolveRelativePath,
    DEFAULT_MAX_FILE_BYTES, EXTRACTOR_VERSION, filesManifestKey, extractorManifestKey, judgedManifestKey, compareManifestKey,
} from './sources.js';
export {
    srcRef, pathOfRef, fileRefOf, factRef, factKeyOf, slugOf, canonicalJson, compareStr, nameResolver, aboutTargets,
} from './refs.js';
export { DOCS_PACK, DOCS_PACK_ID, brokenLinkRule, orphanDocumentRule, contradictionRule } from './packs/docs.js';
export { frontMatterOf, SECTION_INDEX_BADGE } from '../ingest/docsExtract.js';
export { LocalSnapshotStore, localSnapshotStore } from './stores/localSnapshotStore.js';
export type { SnapshotPersistence, KnowledgeSnapshot, LocalSnapshotStoreOptions } from './stores/localSnapshotStore.js';

// The knowledge index (search, expand, near-duplicate pairs) ships on the same subpath.
export * from '../retrieval/knowledge-index/index.js';
