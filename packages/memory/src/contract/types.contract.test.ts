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
import type {
    DirectorySource, Extracted, ExtractJob, ExtractJobResult, GraphApp, GraphOutline, GraphReadSkip, GraphSystem, KEdge, KNode, KnowledgeGraph, KnowledgeStore,
    OutlineFile, OutlineCommunity, NodePlacement, ReviewFailure, ReviewFailureChunks, ReviewPhase, ReviewReport, ReviewRun, ReviewRunPhase, ReviewStep, ScopeRecord,
    StoredFact, StoredRelation, ExtractedLogPersistence, KnowledgeSnapshot, LocalSnapshotStoreOptions, SnapshotPersistence,
    RowChange, RowPersistence,
} from '../knowledge/index.js';
import type { PlannedFile } from '../knowledge/index.js';
import { placementOf } from '../knowledge/index.js';
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
export type ExtractJobResult_fields = Expect<Equal<ExtractJobResult, { binary?: true; extracted: Extracted[]; sections?: ParsedSection[] }>>;
// ExtractFailed: the marker of a PROVEN-final extraction failure (the parser or extractor threw); every other failure is transient.
export type ExtractFailed_is_an_Error = Expect<Equal<InstanceType<typeof ExtractFailed> extends Error ? true : false, true>>;
export type ExtractFailed_constructor = Expect<Equal<ConstructorParameters<typeof ExtractFailed>, [message: string, options?: { cause?: unknown }]>>;
// ExtractorState.failures: consecutive transient failures by path at one hash; optional so a hand-built state stays valid.
export type ExtractorState_failures = Expect<Equal<ExtractorState['failures'], ReadonlyMap<string, { hash: string; count: number }> | undefined>>;
// ExtractorState.oversized: claimed documents over the size cap, planned as assets; optional so a hand-built state stays valid.
export type ExtractorState_oversized = Expect<Equal<ExtractorState['oversized'], ReadonlySet<string> | undefined>>;
// ExtractorState.contentHashRule: the content-hash rule the entry recorded (absent: the one its version implies); optional so a hand-built state stays valid.
export type ExtractorState_contentHashRule = Expect<Equal<ExtractorState['contentHashRule'], string | undefined>>;

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
export type KEdge_verdict = Expect<Equal<KEdge['verdict'], 'allowed' | 'same' | 'skips' | 'upward' | undefined>>;

describe('type contract', () => {
    it('is enforced by `npm run check:contract-types` (tsc over tsconfig.contract.json), not at run time', () => {
        // The values above only exist so the `satisfies` / `@ts-expect-error` lines are real code; this proves the file loads.
        expect(snapshotWithoutExtracted.version).toBe(1);
        expect(typeof readsThroughTheEntriesPort).toBe('function');
        expect(typeof jobWithBoth).toBe('object');
        expect(Object.keys(batchedHostStore)).toContain('batched');
    });
});
