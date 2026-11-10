/**
 * src/contract/types.contract.test.ts — the COMPILE-TIME contract of the shapes consumers depend on
 * (agent-tools `packGraph.ts` / `portablePack.ts`, the VS Code run tests, the API worker).
 *
 * There is nothing to run: the checks are types, and a failing one is a `tsc` error on a line whose alias
 * NAMES what changed (e.g. `KnowledgeSnapshot_extracted_is_optional`). The package build excludes test files,
 * so `npm run check:contract-types` type-checks this file through `tsconfig.contract.json`; CI runs it after
 * the build. A failure means a consumer-visible change: update the check deliberately, note it in the
 * CHANGELOG ("Breaking" when a consumer must change) and bump the consumers.
 *
 * No type-test dependency: `Equal` / `Expect` below are the usual two-line helpers.
 */

import type { ParsedSection, SourceDocument } from '../ingest/types.js';
import type { changedPaths, contentHash, CONTENT_HASH_RULE, merkleTree, MerkleEntry, MerkleTree, PlanDetail, ReviewPlan } from '../knowledge/index.js';
import type { BASELINE_CONTENT_HASH_RULE, canonicalListing, contentHashRuleOf, ListingSkipReason } from '../knowledge/index.js';
import type {
    DirectorySource, Extracted, ExtractJob, ExtractJobResult, GraphApp, GraphOutline, GraphReadSkip, GraphSystem, KEdge, KNode, KnowledgeGraph, KnowledgeStore,
    OutlineFile, OutlineCommunity, NodePlacement, ReviewFailure, ReviewFailureChunks, ReviewPhase, ReviewReport, ReviewRun, ReviewRunPhase, ReviewStep, ScopeRecord,
    StoredFact, StoredRelation, ExtractedLogPersistence, KnowledgeSnapshot, LocalSnapshotStoreOptions, SnapshotPersistence,
    RowChange, RowPersistence, CallSite, SymbolEdgeKind,
} from '../knowledge/index.js';
import type { dedupeSymbolEdges } from '../knowledge/index.js';
import type { PlannedFile } from '../knowledge/index.js';
import type { ExtractFileArgs, PlanOptions, ReviewArgs, refreshPlan } from '../knowledge/index.js';
import type { IngestManifest } from '../ingest/types.js';
import type {
    AnalysisIssue, AnchorKind, CodeAnalysisSnapshot, ComparabilityMeta, Finding, FindingIdentity, LineageEntry,
    comparable, compareAnalysis, identityOf, matchLineage, isBinaryBytes,
} from '../knowledge/index.js';
import { placementOf } from '../knowledge/index.js';
import type { MeasuredWith } from '../knowledge/index.js';
import type {
    applyCoChangeHidden, coChangePartners, coChangeRecallAtK, decodeCoChange, encodeCoChange, mostLikelyWalk, stationTransitions, transitionOf,
    transitionsFromSequences, transitionsFromWeights, CoChangeBadgeEntry, CoChangeCommit, CoChangeOptions, CoChangePartner, CoChangeRecall,
    NodeRef, StationTransition, TransitionRow, TransitionSource, TransitionTable,
} from '../knowledge/index.js';
import type { betaMoments, logisticProbability, onlineLogisticStep, rateInterval, sigmoid, wilsonInterval } from '../stats/index.js';
import type { meanAndSd, MeanAndSd } from '../stats/index.js';
import type {
    CallFact, CapabilityImportFact, DerivedRows, EntryPointFact, EntryPointKind, ExportFact, ExtractorStamp, FactChange, FactGraph, FactGraphOptions,
    FactGraphUpdate, FactKind, FactRule, FileFacts, RuleEvaluation, RuleScope, RuleWorld, TestCaseFact, TestFacts,
    evaluateRules, factGraphDump, factsExtractorTag, updateFactGraph,
} from '../knowledge/index.js';
import type {
    Awaitable, DerivedRow, EffectiveEntryPoint, EffectiveGuardsOptions, EffectiveGuardsResult, FactManifestDelta, FactManifestEntry,
    FactStoreManifest, FactStorePort, FactStoreSaveResult, FactStoreVersion, LoadFactGraphOptions, MemoryFactStore, MountFact, ResolvedMount,
    StoredFactGraph, UseFact, RootRouter, MountTarget, BuildGraphOptions, FindingsProvider, KnowledgeFacts, computeFindingsWith,
    effectiveGuards, factStoreDump, loadFactGraph, memoryFactStore, rootRouters, saveFactGraph, sweepFactStore,
} from '../knowledge/index.js';
import { canonicalDump, ExtractFailed, failuresOf, failureTotal, LocalSnapshotStore, localSnapshotStore, type ExtractorState } from '../knowledge/index.js';

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false;
type Expect<T extends true> = T;
/** True when `K` of `T` may be omitted. */
type IsOptional<T, K extends keyof T> = Record<never, never> extends Pick<T, K> ? true : false;

/* ── KnowledgeSnapshot ─────────────────────────────────────────────────────────────────────────────────────── */
// `extracted` became optional (the extract cache moved to the log); consumers read it ONLY through `extractedEntries()`.
export type KnowledgeSnapshot_keys = Expect<Equal<keyof KnowledgeSnapshot, 'version' | 'scopes' | 'facts' | 'relations' | 'extracted' | 'recheck'>>;
export type KnowledgeSnapshot_extracted_is_optional = Expect<Equal<IsOptional<KnowledgeSnapshot, 'extracted'>, true>>;
export type KnowledgeSnapshot_extracted_shape = Expect<Equal<KnowledgeSnapshot['extracted'], Record<string, Extracted[]> | undefined>>;
export type KnowledgeSnapshot_rows_are_required = Expect<Equal<
    Pick<KnowledgeSnapshot, 'version' | 'scopes' | 'facts' | 'relations'>,
    { version: 1; scopes: ScopeRecord[]; facts: Record<string, StoredFact[]>; relations: Record<string, StoredRelation[]> }
>>;
export const snapshotWithoutExtracted = { version: 1, scopes: [], facts: {}, relations: {} } satisfies KnowledgeSnapshot;
export const readsThroughTheEntriesPort = (store: LocalSnapshotStore): Promise<Record<string, Extracted[]>> => store.extractedEntries();

/* ── ScopeRecord.measured (PRD 41 §C9): recorded when the review runs, carried by the pack's MEAS section ───────────── */
export type MeasuredWith_shape = Expect<Equal<MeasuredWith, { extractorVersion: string; toolVersions: Record<string, string> }>>;
export type ScopeRecord_measured_is_optional = Expect<Equal<IsOptional<ScopeRecord, 'measured'>, true>>;
export type ScopeRecord_measured_shape = Expect<Equal<ScopeRecord['measured'], MeasuredWith | undefined>>;
export type ReviewArgs_measured = Expect<Equal<Pick<import('../knowledge/index.js').ReviewArgs, 'measured'>, { measured?: MeasuredWith }>>;

/* ── ReviewReport ──────────────────────────────────────────────────────────────────────────────────────────── */
export type ReviewReport_judging_fields = Expect<Equal<
    Pick<ReviewReport, 'awaitingJudge' | 'awaitingJudgeCount' | 'judgedFiles' | 'judgeStopped' | 'failures'>,
    {
        awaitingJudge: string[]; awaitingJudgeCount?: number; judgedFiles?: number; judgeStopped?: string;
        failures: ReviewFailure[];
    }
>>;
export type ReviewReport_counters = Expect<Equal<
    Pick<ReviewReport, 'extracted' | 'judged' | 'unjudged' | 'readErrors' | 'extractErrors' | 'aborted' | 'digest' | 'durationMs'>,
    { extracted: number; judged: number; unjudged: number; readErrors: number; extractErrors: number; aborted: boolean; digest: string; durationMs: number }
>>;
export type ReviewFailure_shape = Expect<Equal<ReviewFailure, { path: string; phase: ReviewPhase; reason: string; kind?: 'file' | 'analyzer' }>>;

/* ── ReviewRun: failures are CHUNKS, read through failuresOf / failureTotal ─────────────────────────────────── */
export type ReviewRun_failures_are_chunks = Expect<Equal<ReviewRun['failures'], readonly (readonly ReviewFailure[])[]>>;
export type ReviewFailureChunks_shape = Expect<Equal<ReviewFailureChunks, readonly (readonly ReviewFailure[])[]>>;
export type ReviewRun_failures_is_not_a_flat_list = Expect<Equal<ReviewFailure[] extends ReviewRun['failures'] ? true : false, false>>;
export type failuresOf_signature = Expect<Equal<typeof failuresOf, (run: Pick<ReviewRun, 'failures'>) => ReviewFailure[]>>;
export type failureTotal_signature = Expect<Equal<typeof failureTotal, (run: Pick<ReviewRun, 'failures'>) => number>>;
export type ReviewRun_fields = Expect<Equal<
    Pick<ReviewRun, 'id' | 'root' | 'startedAt' | 'updatedAt' | 'phases' | 'steps' | 'endedAt' | 'outcome' | 'report'>,
    {
        id: string; root: string; startedAt: number; updatedAt: number; phases: ReviewRunPhase[]; steps: ReviewStep[];
        endedAt?: number; outcome?: 'done' | 'stopped' | 'failed'; report?: ReviewReport;
    }
>>;

/* ── ReviewStep.parts, ReviewPhase ─────────────────────────────────────────────────────────────────────────── */
export type ReviewStep_parts = Expect<Equal<ReviewStep['parts'], { readMs?: number; lookupMs?: number; poolMs?: number } | undefined>>;
export type ReviewStep_outcome = Expect<Equal<ReviewStep['outcome'], 'ok' | 'cached' | 'skipped' | 'failed'>>;
export type ReviewStep_fields = Expect<Equal<
    Pick<ReviewStep, 'path' | 'phase' | 'ms' | 'reason' | 'lane'>,
    { path: string; phase: ReviewPhase; ms: number; reason?: string; lane?: number }
>>;
// `mapped` is the end of the mechanical pass: a view that switches on phases must know it.
export type ReviewPhase_values = Expect<Equal<ReviewPhase, 'plan' | 'extract' | 'resolve' | 'judge' | 'compare' | 'write' | 'analyse' | 'index' | 'mapped' | 'done'>>;

/* ── ExtractJob: a union of "the review read it" (doc) and "the pool reads it" (readByPool) ──────────────────── */
const aDocument: SourceDocument = { id: 'a.md', tenantId: 't', content: { kind: 'text', text: '# A\n' } };
const someSections: ParsedSection[] = [{ text: '# A\n' }];
const anExtract: Extracted = { sourceRef: 'src:r::a.md', sourceHash: 'h', nodes: [], edges: [], keys: [] };
const base = { rootId: 'r', path: 'a.md', hash: 'h', knownKeys: [] as ReadonlyArray<readonly [string, KNode['ref']]> };
export const jobCarryingTheDocument = { ...base, doc: aDocument } satisfies ExtractJob;
export const jobTheWorkerReads = { ...base, readByPool: { path: 'a.md', hash: 'h' } } satisfies ExtractJob;
export const jobWithSectionFlags = { ...base, doc: aDocument, needSections: true, sectionsOnly: true } satisfies ExtractJob;
// @ts-expect-error ExtractJob: a job carries `doc` OR `readByPool`, never both
export const jobWithBoth: ExtractJob = { ...base, doc: aDocument, readByPool: { path: 'a.md', hash: 'h' } };
// @ts-expect-error ExtractJob: a job carries one of `doc` / `readByPool`
export const jobWithNeither: ExtractJob = { ...base };
// readByPool gained an OPTIONAL `readMs` (the source's per-path read deadline, for a worker to use instead of its constant); a worker that ignores it is unaffected.
export type ExtractJob_readByPool_shape = Expect<Equal<NonNullable<Extract<ExtractJob, { readByPool: unknown }>['readByPool']>, { path: string; hash: string; readMs?: number }>>;
export const jobTheWorkerReadsWithDeadline = { ...base, readByPool: { path: 'a.md', hash: 'h', readMs: 30_000 } } satisfies ExtractJob;
// PlannedFile.oversized: an asset only because a claimed document is over the size cap; optional, so a hand-built PlannedFile stays valid.
export type PlannedFile_oversized = Expect<Equal<PlannedFile['oversized'], true | undefined>>;
export const resultShape = { binary: true, extracted: [anExtract], sections: someSections } satisfies ExtractJobResult;
// `extractorKey`: the key the worker ACTUALLY extracted under (it echoes its own, never the job's); a mismatch is dropped, not cached.
export type ExtractJobResult_fields = Expect<Equal<ExtractJobResult, { binary?: true; extracted: Extracted[]; sections?: ParsedSection[]; extractorKey?: string }>>;
export type ExtractJob_extractorKey = Expect<Equal<Pick<ExtractJob, 'extractorKey'>, { extractorKey?: string }>>;
// ExtractFailed: the marker of a PROVEN-final extraction failure (the parser or extractor threw); every other failure is transient.
export type ExtractFailed_is_an_Error = Expect<Equal<InstanceType<typeof ExtractFailed> extends Error ? true : false, true>>;
export type ExtractFailed_constructor = Expect<Equal<ConstructorParameters<typeof ExtractFailed>, [message: string, options?: { cause?: unknown }]>>;
// ExtractorState.failures: consecutive transient failures by path at one hash; optional so a hand-built state stays valid.
export type ExtractorState_failures = Expect<Equal<ExtractorState['failures'], ReadonlyMap<string, { hash: string; count: number }> | undefined>>;
// ExtractorState.oversized: claimed documents over the size cap, planned as assets; optional so a hand-built state stays valid.
export type ExtractorState_oversized = Expect<Equal<ExtractorState['oversized'], ReadonlySet<string> | undefined>>;
// ExtractorState.contentHashRule: the content-hash rule the entry recorded (absent: none recorded, which never owes a file); optional so a hand-built state stays valid.
export type ExtractorState_contentHashRule = Expect<Equal<ExtractorState['contentHashRule'], string | undefined>>;
// The caller-supplied extractor key (agent-tools: the tsconfig path-alias digest): optional everywhere, so a host passing none is unchanged.
export type ExtractorState_extractorKey = Expect<Equal<Pick<ExtractorState, 'extractorKey' | 'extractorKeyNow'>, { extractorKey?: string; extractorKeyNow?: string }>>;
export type Extracted_extractorKey = Expect<Equal<Pick<Extracted, 'extractorKey'>, { extractorKey?: string }>>;
export type ExtractFileArgs_extractorKey = Expect<Equal<Pick<ExtractFileArgs, 'extractorKey'>, { extractorKey?: string }>>;
export type PlanOptions_extractorKey = Expect<Equal<Pick<PlanOptions, 'extractorKey'>, { extractorKey?: string }>>;
export type ReviewArgs_extractorKey = Expect<Equal<Pick<ReviewArgs, 'extractorKey'>, { extractorKey?: string }>>;
export type refreshPlan_signature = Expect<Equal<
    Parameters<typeof refreshPlan>, [detail: PlanDetail, manifest: IngestManifest, extractorKey?: string]
>>;

/* ── Content identity and directory Merkle hashes (PRD 41 §B2 steps 0 and 4) ───────────────────────────────── */
export type contentHash_signature = Expect<Equal<typeof contentHash, (bytes: Uint8Array) => string>>;
export type CONTENT_HASH_RULE_type = Expect<Equal<typeof CONTENT_HASH_RULE, 'eol-lf@1'>>;
export type MerkleEntry_shape = Expect<Equal<MerkleEntry, { name: string; kind: 'file' | 'dir'; hash: string }>>;
export type MerkleTree_shape = Expect<Equal<MerkleTree, {
    root: string; dirs: ReadonlyMap<string, string>; entries: ReadonlyMap<string, ReadonlyArray<MerkleEntry>>;
}>>;
export type merkleTree_signature = Expect<Equal<typeof merkleTree, (files: ReadonlyArray<{ path: string; hash: string }>) => MerkleTree>>;
export type changedPaths_signature = Expect<Equal<typeof changedPaths, (before: MerkleTree, after: MerkleTree) => string[]>>;
export type ReviewPlan_merkleRoot = Expect<Equal<ReviewPlan['merkleRoot'], string | undefined>>;
export type PlanDetail_merkle = Expect<Equal<PlanDetail['merkle'], MerkleTree | undefined>>;
export type BASELINE_CONTENT_HASH_RULE_type = Expect<Equal<typeof BASELINE_CONTENT_HASH_RULE, 'eol-lf@1'>>;
export type contentHashRuleOf_signature = Expect<Equal<typeof contentHashRuleOf, (recorded: string | undefined) => string>>;
// The plan never throws over a listing: invalid and colliding paths are skipped with a reason; a re-spelled path keeps the source's spelling for reads.
export type ReviewPlan_skipped_reason = Expect<Equal<
    ReviewPlan['skipped'][number]['reason'], 'ignored' | 'too-large' | 'no-pack' | 'invalid-path' | 'duplicate-path'
>>;
export type PlannedFile_listedPath = Expect<Equal<PlannedFile['listedPath'], string | undefined>>;
export type canonicalListing_skipped = Expect<Equal<
    ReturnType<typeof canonicalListing<{ path: string }>>['skipped'], Array<{ path: string; reason: ListingSkipReason }>
>>;
export type ListingSkipReason_values = Expect<Equal<ListingSkipReason, 'invalid-path' | 'duplicate-path'>>;

/* ── DirectorySource: optional per-path hooks ──────────────────────────────────────────────────────────────── */
// hashOf takes the path (a host whose read() transforms bytes returns the listing hash for it); a one-argument hashOf still satisfies it.
export type DirectorySource_hashOf = Expect<Equal<DirectorySource['hashOf'], ((text: string, path: string) => string) | undefined>>;
export type DirectorySource_hashOf_legacy_one_arg = Expect<Equal<((text: string) => string) extends NonNullable<DirectorySource['hashOf']> ? true : false, true>>;
export type DirectorySource_readTimeoutMs = Expect<Equal<DirectorySource['readTimeoutMs'], ((path: string) => number | undefined) | undefined>>;

/* ── KnowledgeStore: the port hosts implement ──────────────────────────────────────────────────────────────── */
export type KnowledgeStore_methods = Expect<Equal<
    keyof KnowledgeStore,
    'scopes' | 'facts' | 'relations' | 'write' | 'markStale' | 'getExtracted' | 'putExtracted' | 'preloadExtracted' | 'markRecheck' | 'recheckHashes' | 'clearRecheck' | 'hasRecheck' | 'batched'
>>;
export type KnowledgeStore_optional_methods = Expect<Equal<
    [IsOptional<KnowledgeStore, 'getExtracted'>, IsOptional<KnowledgeStore, 'putExtracted'>, IsOptional<KnowledgeStore, 'preloadExtracted'>, IsOptional<KnowledgeStore, 'batched'>,
        IsOptional<KnowledgeStore, 'markRecheck'>, IsOptional<KnowledgeStore, 'recheckHashes'>, IsOptional<KnowledgeStore, 'clearRecheck'>, IsOptional<KnowledgeStore, 'hasRecheck'>],
    [true, true, true, true, true, true, true, true]
>>;
export type KnowledgeStore_required_methods = Expect<Equal<
    [IsOptional<KnowledgeStore, 'scopes'>, IsOptional<KnowledgeStore, 'facts'>, IsOptional<KnowledgeStore, 'relations'>, IsOptional<KnowledgeStore, 'write'>, IsOptional<KnowledgeStore, 'markStale'>],
    [false, false, false, false, false]
>>;
export type KnowledgeStore_signatures = Expect<Equal<
    Pick<KnowledgeStore, 'scopes' | 'facts' | 'relations' | 'write' | 'markStale' | 'getExtracted' | 'putExtracted' | 'preloadExtracted' | 'markRecheck' | 'recheckHashes' | 'clearRecheck' | 'hasRecheck'>,
    {
        scopes(): Promise<ScopeRecord[]>;
        facts(scope: { rootId: string }): Promise<StoredFact[]>;
        relations(scope: { rootId: string }): Promise<StoredRelation[]>;
        write(batch: { scope: ScopeRecord; facts: StoredFact[]; relations: StoredRelation[] }): Promise<void>;
        markStale(scope: { rootId: string }, sourceRef: KNode['ref'], currentHash: string): Promise<void>;
        getExtracted?(hash: string): Promise<Extracted[] | undefined>;
        putExtracted?(hash: string, extracted: Extracted[]): Promise<void>;
        preloadExtracted?(hashes: readonly string[]): Promise<void>;
        markRecheck?(hashes: readonly string[]): Promise<void>;
        recheckHashes?(): Promise<string[]>;
        clearRecheck?(hashes: readonly string[]): Promise<void>;
        hasRecheck?(hash: string): Promise<boolean>;
    }
>>;
// A host store with only the required methods is still a KnowledgeStore (the optional ones stay optional).
export const minimalHostStore = {
    async scopes() { return [] as ScopeRecord[]; },
    async facts() { return [] as StoredFact[]; },
    async relations() { return [] as StoredRelation[]; },
    async write() {},
    async markStale() {},
} satisfies KnowledgeStore;
export const batchedHostStore = {
    ...minimalHostStore,
    batched: <T>(work: (flush: () => Promise<void>) => Promise<T>): Promise<T> => work(async () => {}),
} satisfies KnowledgeStore;

/* ── LocalSnapshotStore: the shipped KnowledgeStore ────────────────────────────────────────────────────────── */
export type LocalSnapshotStore_is_a_KnowledgeStore = Expect<Equal<LocalSnapshotStore extends KnowledgeStore ? true : false, true>>;
export type LocalSnapshotStore_public_methods = Expect<Equal<
    keyof LocalSnapshotStore,
    'batched' | 'scopes' | 'facts' | 'relations' | 'write' | 'markStale' | 'getExtracted' | 'extractedEntries' | 'preloadExtracted' | 'putExtracted' | 'markRecheck' | 'recheckHashes' | 'clearRecheck' | 'hasRecheck'
>>;
export type LocalSnapshotStore_extractedEntries = Expect<Equal<LocalSnapshotStore['extractedEntries'], () => Promise<Record<string, Extracted[]>>>>;
export type LocalSnapshotStore_constructor = Expect<Equal<
    ConstructorParameters<typeof LocalSnapshotStore>, [persistence: SnapshotPersistence, opts?: LocalSnapshotStoreOptions]
>>;
// Two overloads: the snapshot-persisted one (the last, so `Parameters` reads it) and the row-persisted one (`undefined` + `opts.rows`).
export type localSnapshotStore_factory = Expect<Equal<Parameters<typeof localSnapshotStore>, [persistence: SnapshotPersistence, opts?: LocalSnapshotStoreOptions]>>;
export type localSnapshotStore_factory_returns = Expect<Equal<ReturnType<typeof localSnapshotStore>, LocalSnapshotStore>>;
export const rowPersistedStore = (rows: RowPersistence): LocalSnapshotStore => localSnapshotStore(undefined, { rows });
export const rowPersistedStoreByConstructor = (rows: RowPersistence): LocalSnapshotStore => new LocalSnapshotStore(undefined, { rows });
export const dumpsAStore = (store: LocalSnapshotStore): Promise<string> => canonicalDump(store);
export type LocalSnapshotStoreOptions_keys = Expect<Equal<keyof LocalSnapshotStoreOptions, 'extractedLog' | 'rows' | 'now' | 'checkpointMs' | 'nowMs'>>;
export type RowChange_shape = Expect<Equal<RowChange, {
    upserts: {
        scopes: ScopeRecord[]; facts: Array<[rootId: string, fact: StoredFact]>; relations: Array<[rootId: string, relation: StoredRelation]>;
        extracted: Array<[hash: string, entries: Extracted[]]>; recheck: string[];
    };
    deletes: {
        scopes: string[]; facts: Array<[rootId: string, key: string]>;
        relations: Array<[rootId: string, from: string, to: string, kind: string]>;
        extracted: string[]; recheck: string[];
    };
}>>;
export type RowPersistence_ports = Expect<Equal<RowPersistence, {
    load(): Promise<KnowledgeSnapshot | undefined>;
    apply(change: RowChange): Promise<{ generation: number; reloaded?: KnowledgeSnapshot }>;
    loadAllExtracted?(): Promise<Record<string, Extracted[]>>;
    loadExtracted?(hash: string): Promise<Extracted[] | undefined>;
    extractedSources?(): Promise<Record<string, string[]>>;
    changedElsewhere?(): Promise<boolean>;
}>>;
export type SnapshotPersistence_ports = Expect<Equal<
    SnapshotPersistence,
    { read(): Promise<string | undefined>; write(snapshot: string): Promise<void>; keepAside?(text: string): Promise<void> }
>>;
export type ExtractedLogPersistence_ports = Expect<Equal<
    ExtractedLogPersistence,
    { read(): Promise<string | undefined>; append(text: string): Promise<void>; replace?(text: string): Promise<void> }
>>;

/* ── GraphSystem / KNode system fields ─────────────────────────────────────────────────────────────────────── */
export type KNode_system_fields = Expect<Equal<
    Pick<KNode, 'system' | 'app' | 'copyOf'>,
    { system?: string; app?: string; copyOf?: string }
>>;
export type GraphSystem_fields = Expect<Equal<
    GraphSystem,
    {
        path: string; name: string; files: number; assets?: number; apps: GraphApp[]; truncated?: boolean; copyOf?: string; copySimilarity?: number;
        copyReason?: 'no-git' | 'name-mismatch' | 'later-name'; graphRead?: { read: number; skipped: GraphReadSkip[] };
    }
>>;
export type GraphApp_fields = Expect<Equal<GraphApp, { path: string; name: string }>>;
export type GraphReadSkip_fields = Expect<Equal<GraphReadSkip, { reason: string; count: number }>>;
export type KnowledgeGraph_systems = Expect<Equal<NonNullable<KnowledgeGraph['meta']>['systems'], GraphSystem[] | undefined>>;
export type GraphOutline_systems = Expect<Equal<
    Pick<GraphOutline, 'systems' | 'apps' | 'systemInfo'>,
    { systems?: string[]; apps?: string[]; systemInfo?: GraphSystem[] }
>>;
export type OutlineFile_system_indexes = Expect<Equal<Pick<OutlineFile, 'system' | 'app' | 'copyOf'>, { system?: number; app?: number; copyOf?: number }>>;
// PRD 38 P1: communities. `communityId` indexes the outline's `communities` table; ids are per build.
export type KnowledgeGraph_stats_modularity = Expect<Equal<KnowledgeGraph['stats']['modularity'], number | undefined>>;
export type KnowledgeGraph_stats_communityMethod = Expect<Equal<KnowledgeGraph['stats']['communityMethod'], 'louvain' | 'label-propagation' | undefined>>;
export type GraphOutline_communities = Expect<Equal<GraphOutline['communities'], OutlineCommunity[] | undefined>>;
export type OutlineCommunity_fields = Expect<Equal<OutlineCommunity, { id: number; label: string; size: number }>>;
export type NodePlacement_shape = Expect<Equal<NodePlacement, { kind: 'partition'; id: string } | { kind: 'community'; id: number }>>;
export type placementOf_signature = Expect<Equal<ReturnType<typeof placementOf>, NodePlacement | undefined>>;
// PRD 38 P4: co-change partners. The `coChange` badge is a string read ONLY through decodeCoChange.
export type CoChangeCommit_fields = Expect<Equal<CoChangeCommit, { at?: string; refs: readonly NodeRef[]; fileCount?: number }>>;
export type CoChangeOptions_fields = Expect<Equal<
    CoChangeOptions, { minSupport?: number; minLift?: number; maxCommitFiles?: number; maxPartners?: number; maxCommits?: number }
>>;
export type CoChangePartner_fields = Expect<Equal<
    CoChangePartner, { ref: NodeRef; together: number; ofSelf: number; confidence: number; lift: number; low: number; high: number }
>>;
export type CoChangeBadgeEntry_fields = Expect<Equal<CoChangeBadgeEntry, { ref: NodeRef; together: number; ofSelf: number; low: number; high: number }>>;
export type CoChangeRecall_shape = Expect<Equal<CoChangeRecall, { recall: number; cases: number; skipped: number } | { status: 'insufficient-data' }>>;
export type coChangePartners_signature = Expect<Equal<
    typeof coChangePartners, (commits: ReadonlyArray<CoChangeCommit>, opts?: CoChangeOptions) => Map<NodeRef, CoChangePartner[]>
>>;
export type encodeCoChange_signature = Expect<Equal<typeof encodeCoChange, (partners: readonly CoChangePartner[]) => string>>;
export type decodeCoChange_signature = Expect<Equal<typeof decodeCoChange, (value: unknown) => CoChangeBadgeEntry[]>>;
export type coChangeRecallAtK_signature = Expect<Equal<
    typeof coChangeRecallAtK, (commits: ReadonlyArray<CoChangeCommit>, k?: number, heldOutShare?: number, opts?: CoChangeOptions) => CoChangeRecall
>>;
export type applyCoChangeHidden_signature = Expect<Equal<typeof applyCoChangeHidden, (nodes: Iterable<KNode>, edges: Iterable<KEdge>) => void>>;
// The one interval of a shown rate (PRD 41 D3), shared by co-change confidence and trace hops.
export type rateInterval_signature = Expect<Equal<typeof rateInterval, (k: number, n: number) => { low: number; high: number }>>;
// PRD 41 D13: the one sigmoid / logistic head, Wilson interval and Beta moments that Builderforce's copies migrate onto.
export type sigmoid_signature = Expect<Equal<typeof sigmoid, (z: number) => number>>;
export type logisticProbability_signature = Expect<Equal<
    typeof logisticProbability, (weights: readonly number[], bias: number, x: readonly number[]) => number
>>;
export type onlineLogisticStep_signature = Expect<Equal<
    typeof onlineLogisticStep,
    (weights: readonly number[], bias: number, x: readonly number[], label: 0 | 1, learningRate: number) => { weights: number[]; bias: number }
>>;
export type wilsonInterval_signature = Expect<Equal<
    typeof wilsonInterval, (successes: number, n: number, z?: number) => { low: number; high: number }
>>;
export type betaMoments_signature = Expect<Equal<typeof betaMoments, (alpha: number, beta: number) => { mean: number; variance: number }>>;
// PRD 41 D13: the one mean / sample sd (Welford) for predicted measures; run-delta's naturalness model migrates onto it.
export type meanAndSd_signature = Expect<Equal<typeof meanAndSd, (values: readonly number[]) => MeanAndSd>>;
export type MeanAndSd_fields = Expect<Equal<MeanAndSd, { n: number; mean: number; sd: number; nonFinite: number }>>;
// PRD 38 P7: station transitions (the view groups stations; memory owns the formula). Reference rows carry no interval.
export type TransitionSource_values = Expect<Equal<TransitionSource, 'traces' | 'references'>>;
export type StationTransition_fields = Expect<Equal<StationTransition, { to: string; p: number; k: number; n: number; low?: number; high?: number }>>;
export type TransitionRow_fields = Expect<Equal<TransitionRow, { source: TransitionSource; transitions: readonly StationTransition[] }>>;
export type TransitionTable_shape = Expect<Equal<TransitionTable, ReadonlyMap<string, TransitionRow>>>;
type WeightedEdges = ReadonlyArray<{ from: string; to: string; weight: number }>;
export type transitionsFromWeights_signature = Expect<Equal<typeof transitionsFromWeights, (edges: WeightedEdges) => TransitionTable>>;
export type transitionsFromSequences_signature = Expect<Equal<
    typeof transitionsFromSequences, (sequences: ReadonlyArray<readonly string[]>) => TransitionTable
>>;
export type stationTransitions_signature = Expect<Equal<
    typeof stationTransitions, (input: { sequences?: ReadonlyArray<readonly string[]>; weights: WeightedEdges }) => TransitionTable
>>;
export type transitionOf_signature = Expect<Equal<
    typeof transitionOf, (table: TransitionTable, from: string, to: string) => StationTransition | undefined
>>;
export type mostLikelyWalk_signature = Expect<Equal<
    typeof mostLikelyWalk, (table: TransitionTable, start: string, maxHops: number) => { stops: string[]; hops: StationTransition[] }
>>;
export type KEdge_verdict = Expect<Equal<KEdge['verdict'], 'allowed' | 'same' | 'skips' | 'upward' | undefined>>;

/* ── Symbol-level edges (PRD 43 P0b): `calls` edges carry their call sites ─────────────────────────────────── */
export type KEdge_sites = Expect<Equal<KEdge['sites'], CallSite[] | undefined>>;
export type KEdge_sites_is_optional = Expect<Equal<IsOptional<KEdge, 'sites'>, true>>;
export type CallSite_shape = Expect<Equal<CallSite, { path: string; line: number; column: number; order: number }>>;
export type SymbolEdgeKind_values = Expect<Equal<SymbolEdgeKind, 'extends' | 'implements' | 'composes' | 'uses' | 'calls'>>;
export type dedupeSymbolEdges_signature = Expect<Equal<
    typeof dedupeSymbolEdges, (edges: readonly KEdge[], opts?: { maxPerSymbol?: number; maxSites?: number }) => KEdge[]
>>;

/* ── Finding identity v2, lineage, comparability (PRD 41 §C3/§C9) ─────────────────────────────────────────── */
export type AnchorKind_values = Expect<Equal<AnchorKind, 'symbol' | 'route' | 'package' | 'file' | 'group'>>;
export type Finding_anchor_is_optional = Expect<Equal<IsOptional<Finding, 'anchor'>, true>>;
export type Finding_anchorKind = Expect<Equal<Finding['anchorKind'], AnchorKind | undefined>>;
export type FindingIdentity_shape = Expect<Equal<FindingIdentity, { id: string; anchor: string; anchorKind: AnchorKind; members: string[] }>>;
export type identityOf_signature = Expect<Equal<typeof identityOf, (f: Finding, ruleVersion?: number) => FindingIdentity>>;
export type LineageEntry_shape = Expect<Equal<LineageEntry, {
    lineage: 'introduced' | 'resolved' | 'changed' | 'persisting'; change?: 'grew' | 'shrank' | 'moved'; before?: string; after?: string;
}>>;
export type matchLineage_signature = Expect<Equal<
    typeof matchLineage, (before: readonly FindingIdentity[], after: readonly FindingIdentity[]) => LineageEntry[]
>>;
export type ComparabilityMeta_shape = Expect<Equal<ComparabilityMeta, {
    extractor?: string; extractorVersion?: string | number; contentHashRule?: string; tools?: Record<string, string>;
    scope?: string; repo?: string; branch?: string; truncated?: boolean; failed?: string[];
}>>;
export type comparable_signature = Expect<Equal<
    typeof comparable, (before: ComparabilityMeta, after: ComparabilityMeta) => { comparable: boolean; reasons: string[] }
>>;
export type AnalysisIssue_anchor_is_optional = Expect<Equal<IsOptional<AnalysisIssue, 'anchor'>, true>>;
export type compareAnalysis_result = Expect<Equal<
    ReturnType<typeof compareAnalysis>,
    { comparable: boolean; reasons: string[]; added: AnalysisIssue[]; unchanged: AnalysisIssue[]; resolved: AnalysisIssue[] }
>>;
export type CodeAnalysisSnapshot_version = Expect<Equal<CodeAnalysisSnapshot['version'], 1>>;
export type CodeAnalysisSnapshot_truncated = Expect<Equal<CodeAnalysisSnapshot['truncated'], true | undefined>>;
export type CodeAnalysisSnapshot_truncated_is_optional = Expect<Equal<IsOptional<CodeAnalysisSnapshot, 'truncated'>, true>>;
export type isBinaryBytes_signature = Expect<Equal<typeof isBinaryBytes, (bytes: Uint8Array) => boolean>>;

/* ── FileFacts v3, FactRule, the fact graph (PRD 41 §B1/§B3/§B5) ──────────────────────────────────────────── */
// The Phase 2 extractors (agent-tools, tree-sitter) emit exactly this shape; run-delta rules read it.
export type FactKind_values = Expect<Equal<FactKind, 'body' | 'content' | 'structure'>>;
// 2026-11-10 (additive, §C5.1): + optional `mounts` and `uses`.
export type FileFacts_keys = Expect<Equal<keyof FileFacts,
    'factsVersion' | 'extractor' | 'language' | 'contentHash' | 'symbols' | 'imports' | 'exports' | 'entryPoints'
    | 'calls' | 'complexity' | 'capabilityImports' | 'testFacts' | 'mounts' | 'uses'>>;
export type FileFacts_testFacts_is_optional = Expect<Equal<IsOptional<FileFacts, 'testFacts'>, true>>;
// 2026-11-10 (additive, §C5.1): + optional registration `order` and `router`.
export type EntryPointFact_shape = Expect<Equal<EntryPointFact, {
    kind: EntryPointKind; key: string; symbol: string; guards: string[]; inputs: string[]; chain: string[]; order?: number; router?: string;
}>>;
export type ExportFact_keys = Expect<Equal<keyof ExportFact, 'name' | 'kind' | 'signatureHash' | 'from'>>;
export type CallFact_keys = Expect<Equal<keyof CallFact, 'callee' | 'enclosing' | 'awaited' | 'inLoop' | 'count'>>;
export type CapabilityImportFact_keys = Expect<Equal<keyof CapabilityImportFact, 'capability' | 'module' | 'symbol' | 'algorithm'>>;
export type TestFacts_keys = Expect<Equal<keyof TestFacts, 'framework' | 'tests' | 'snapshotFiles'>>;
export type TestCaseFact_keys = Expect<Equal<keyof TestCaseFact, 'name' | 'assertions' | 'matchers' | 'modifier' | 'timeoutMs' | 'expectedHash'>>;
export type FactRule_keys = Expect<Equal<keyof FactRule, 'id' | 'version' | 'reads' | 'scope' | 'hops' | 'readsGuardChain' | 'extraInputs' | 'derive'>>;
export type RuleScope_values = Expect<Equal<RuleScope, 'file' | 'neighbourhood' | 'global'>>;
export type factsExtractorTag_signature = Expect<Equal<typeof factsExtractorTag, (stamp: ExtractorStamp, language?: string) => string>>;
export type evaluateRules_signature = Expect<Equal<
    typeof evaluateRules, <X>(rules: readonly FactRule<X>[], world: RuleWorld<X>, previous?: DerivedRows) => RuleEvaluation
>>;
export type updateFactGraph_signature = Expect<Equal<
    typeof updateFactGraph, <X>(graph: FactGraph, changes: readonly FactChange[], rules: readonly FactRule<X>[], opts?: FactGraphOptions<X>) => FactGraphUpdate
>>;
export type factGraphDump_signature = Expect<Equal<typeof factGraphDump, (graph: FactGraph) => string>>;

/* ── FactStorePort (PRD 41 §B4) and router mounts (§C5.1) ──────────────────────────────────────────────────── */
// Node hosts implement the port on the G03 SQLite store, the Worker behind a Durable Object (or one R2 object with an
// `onlyIf` etag); memory owns the shape and the save / load / sweep.
export type FileFacts_mounts_is_optional = Expect<Equal<IsOptional<FileFacts, 'mounts'>, true>>;
export type FileFacts_uses_is_optional = Expect<Equal<IsOptional<FileFacts, 'uses'>, true>>;
export type MountFact_keys = Expect<Equal<keyof MountFact, 'prefix' | 'target' | 'order' | 'router' | 'guards'>>;
export type MountFact_required = Expect<Equal<Pick<MountFact, 'prefix' | 'target' | 'order'>, { prefix: string; target: string; order: number }>>;
export type UseFact_keys = Expect<Equal<keyof UseFact, 'path' | 'match' | 'guards' | 'order' | 'router'>>;
export type UseFact_match = Expect<Equal<UseFact['match'], 'prefix' | 'exact' | undefined>>;
export type FactStoreVersion_shape = Expect<Equal<FactStoreVersion, { epoch: string; generation: number }>>;
export type FactStorePort_keys = Expect<Equal<keyof FactStorePort,
    'getObjects' | 'putObjects' | 'readVersion' | 'readManifest' | 'writeManifest' | 'derivedRows' | 'sweepObjects' | 'reset'>>;
export type FactStorePort_writeManifest = Expect<Equal<FactStorePort['writeManifest'],
    (manifest: FactManifestDelta, derivedPuts: readonly DerivedRow[], derivedDeletes: readonly string[]) => Awaitable<void>>>;
export type FactStorePort_getObjects = Expect<Equal<FactStorePort['getObjects'], (hashes: readonly string[]) => Awaitable<ReadonlyMap<string, string>>>>;
export type FactStorePort_putObjects = Expect<Equal<FactStorePort['putObjects'], (objects: ReadonlyMap<string, string>) => Awaitable<void>>>;
export type FactStorePort_derivedRows = Expect<Equal<FactStorePort['derivedRows'], () => Awaitable<ReadonlyMap<string, DerivedRow>>>>;
export type FactStorePort_sweepObjects = Expect<Equal<FactStorePort['sweepObjects'], (expected: FactStoreVersion) => Awaitable<number>>>;
export type FactStoreManifest_keys = Expect<Equal<keyof FactStoreManifest, 'epoch' | 'generation' | 'files'>>;
export type FactStoreManifest_files = Expect<Equal<FactStoreManifest['files'], ReadonlyMap<string, FactManifestEntry>>>;
export type FactManifestDelta_keys = Expect<Equal<keyof FactManifestDelta, 'epoch' | 'generation' | 'puts' | 'deletes'>>;
export type StoredFactGraph_keys = Expect<Equal<keyof StoredFactGraph, 'epoch' | 'generation' | 'graph'>>;
export type FactStoreSaveResult_keys = Expect<Equal<keyof FactStoreSaveResult,
    'epoch' | 'generation' | 'graph' | 'objects' | 'puts' | 'deletes' | 'derivedPuts' | 'derivedDeletes'>>;
export type LoadFactGraphOptions_shape = Expect<Equal<LoadFactGraphOptions, { verify?: 'full' | 'manifest' }>>;
export type saveFactGraph_signature = Expect<Equal<
    typeof saveFactGraph, (port: FactStorePort, graph: FactGraph, previous?: StoredFactGraph) => Promise<FactStoreSaveResult>
>>;
export type loadFactGraph_signature = Expect<Equal<typeof loadFactGraph, (port: FactStorePort, opts?: LoadFactGraphOptions) => Promise<StoredFactGraph>>>;
export type factStoreDump_signature = Expect<Equal<typeof factStoreDump, (port: FactStorePort) => Promise<string>>>;
export type sweepFactStore_signature = Expect<Equal<typeof sweepFactStore, (port: FactStorePort) => Promise<number>>>;
export type memoryFactStore_signature = Expect<Equal<typeof memoryFactStore, () => MemoryFactStore>>;
export type effectiveGuards_signature = Expect<Equal<
    typeof effectiveGuards, (fileFacts: FileFacts, resolvedMounts: readonly ResolvedMount[], opts?: EffectiveGuardsOptions) => EffectiveGuardsResult
>>;
export type EffectiveGuardsResult_shape = Expect<Equal<EffectiveGuardsResult, { entryPoints: EffectiveEntryPoint[]; truncated: number; unresolved: number }>>;
export type EffectiveGuardsOptions_shape = Expect<Equal<EffectiveGuardsOptions, { path?: string; router?: string; maxDepth?: number }>>;
export type rootRouters_signature = Expect<Equal<
    typeof rootRouters, (files: ReadonlyMap<string, FileFacts>, resolveMount: (from: string, mount: MountFact) => MountTarget | undefined) => RootRouter[]
>>;
/* ── buildGraph findings provider (PRD 41 §B5 host seam) ────────────────────────────────────────────────────── */
export type BuildGraphOptions_findings_is_optional = Expect<Equal<IsOptional<BuildGraphOptions, 'findings'>, true>>;
export type BuildGraphOptions_findings = Expect<Equal<BuildGraphOptions['findings'], FindingsProvider | undefined>>;
export type FindingsProvider_keys = Expect<Equal<keyof FindingsProvider, 'kinds' | 'findings'>>;
export type FindingsProvider_findings = Expect<Equal<
    FindingsProvider['findings'], (input: { graph: KnowledgeGraph; knowledge: KnowledgeFacts }) => Finding[]
>>;
export type computeFindingsWith_last_param = Expect<Equal<Parameters<typeof computeFindingsWith>[4], FindingsProvider>>;
export type RootRouter_shape = Expect<Equal<RootRouter, { path: string; router: string }>>;
export type MountTarget_shape = Expect<Equal<MountTarget, { path: string; router?: string }>>;
export type ResolvedMount_shape = Expect<Equal<ResolvedMount, {
    target: string; path: string; facts: FileFacts; router?: string; mounts?: readonly ResolvedMount[];
}>>;
export type EffectiveEntryPoint_extra_keys = Expect<Equal<Exclude<keyof EffectiveEntryPoint, keyof EntryPointFact>, 'file' | 'via' | 'guardsInferred'>>;

describe('type contract', () => {
    it('is enforced by `npm run check:contract-types` (tsc over tsconfig.contract.json), not at run time', () => {
        // The values above only exist so the `satisfies` / `@ts-expect-error` lines are real code; this proves the file loads.
        expect(snapshotWithoutExtracted.version).toBe(1);
        expect(typeof readsThroughTheEntriesPort).toBe('function');
        expect(typeof jobWithBoth).toBe('object');
        expect(Object.keys(batchedHostStore)).toContain('batched');
    });
});
