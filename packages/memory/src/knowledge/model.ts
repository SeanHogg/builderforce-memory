/**
 * The knowledge graph model — memory's structure for KNOWLEDGE: facts connected to each
 * other and to the sources they came from (PRD: Builderforce.ai/docs/specs/38-prd-knowledge-graph.md).
 *
 * Generic: it knows sources, entities, facts and relations — never imports or headings.
 * Packs (docs builtin, code registered by the host) supply that meaning as DATA.
 *
 * Store judgement, compute perception: `extracted` nodes/edges are recomputed from sources
 * (cached by content hash) and are NEVER written to a KnowledgeStore; `model`/`human`/`agent`
 * ones are stored with the content hash they were judged from and go stale when it changes.
 */

import type { SourceDocument, ParsedSection } from '../ingest/types.js';
import type { IndexDoc, SimilarPair } from '../retrieval/knowledge-index/types.js';
import type { Finding, FindingRule } from './findingModel.js';

/** The finding types live beside the rules (findingModel.ts); re-exported so one import serves. */
export type { Finding, FindingRule } from './findingModel.js';

/** `src:${rootId}::${path}` · `ent:${pack}:${normalizedName}` · `fact:${key}` (claims are `fact:claim:${hash}`). */
export type NodeRef = `src:${string}` | `ent:${string}` | `fact:${string}`;

export type Origin = 'extracted' | 'model' | 'human' | 'agent';
export type StoredOrigin = Exclude<Origin, 'extracted'>;

export interface KNode {
    ref: NodeRef;
    /** Pack-defined: 'file' | 'section' | 'symbol' | 'area' | 'entity' | 'claim' | 'fact' … */
    kind: string;
    label: string;
    pack: string;
    partition?: string;
    area?: NodeRef;
    /** PageRank over the joined graph, normalized 0–1 (areas: sum of members, renormalized). */
    importance: number;
    origin: Origin;
    sourceHash?: string;
    stale?: boolean;
    /** Pack data, e.g. { added, removed }, { unused: 'file' }, { line: 12 }. */
    badges?: Record<string, number | string>;
    /** The system (a repository or project folder of the root) a file belongs to; '' when the root is one system. File nodes only. */
    system?: string;
    /** The workspace package (app) of the system the file belongs to, as a root-relative folder. */
    app?: string;
    /** On a file of a duplicated copy: the path of the system it is a copy of, so a view can fold it. */
    copyOf?: string;
}

export type EdgeVerdict = 'allowed' | 'same' | 'skips' | 'upward';

export interface KEdge {
    from: NodeRef;
    to: NodeRef;
    /** 'references' | 'links' | 'contains' | 'mentions' | 'duplicate' | 'near-duplicate'
     *  | 'contradicts' | 'agrees' | 'supersedes' | 'about' | 'co-changed' — a new kind is a value. */
    kind: string;
    weight: number;
    origin: Origin;
    confidence?: number;
    verdict?: EdgeVerdict;
    cycle?: boolean;
    /** A stored relation whose source changed since it was judged (§5.2): shown, never treated as current. */
    stale?: boolean;
    /**
     * Extracted edges only: the `[[Name]]` this link was written with. `buildGraph` re-points
     * the edge to the one present file of that name (else it keeps its beside-the-file
     * target) — so the cached extract never depends on which other files exist. Never emitted.
     */
    byName?: string;
}

/** Output of a pack's deterministic extractor (or analyzer) for one source. Never stored. */
export interface Extracted {
    sourceRef: NodeRef;
    sourceHash: string;
    /** The extractor version (`EXTRACTOR_VERSION`) that produced it; a cached entry of another version is ignored. */
    extractorVersion?: number;
    nodes: KNode[];
    edges: KEdge[];
    /** Exact keys for entity resolution and the index's exact tier. */
    keys: Array<{ ref: NodeRef; key: string }>;
}

export interface ExtractContext {
    rootId: string;
    /** Normalized entity keys known so far (for `mentions`). */
    knownKeys: ReadonlyMap<string, NodeRef>;
    /** Resolves a relative link/import from `fromPath` to a root-relative path, or undefined. */
    resolvePath(fromPath: string, target: string): string | undefined;
}

export interface PartitionRule {
    id: string;
    /** 0 = top. */
    order: number;
    /** Globs over the source path; first matching rule wins. */
    match: string[];
}

export interface PartitionRegistry {
    partitions: PartitionRule[];
    /** from-partition → partitions it may use. Absent ⇒ no verdicts. */
    allow?: Record<string, string[]>;
}

/** What the judgement model is asked for one section kind; output is schema-validated. */
export interface JudgementSpec {
    /** Node kinds whose text is judged (e.g. ['section']). */
    nodeKinds: string[];
    /** Ask for claims[] / entities[] per node; and contradicts|agrees for near-duplicate claim pairs. */
    claims: boolean;
    entities: boolean;
    contradictions: boolean;
}

export interface JudgementRequest {
    kind: 'extract' | 'compare';
    /** extract: one node's text; compare: two claims. */
    items: Array<{ ref: NodeRef; text: string; sourceHash: string }>;
    signal?: AbortSignal;
}

export interface JudgedClaim { subject: string; predicate: string; object: string; quote: string }
export interface JudgedEntity { name: string; type: string }

export interface JudgementResult {
    extract?: Array<{ ref: NodeRef; claims: JudgedClaim[]; entities: JudgedEntity[] }>;
    compare?: Array<{ a: NodeRef; b: NodeRef; relation: 'contradicts' | 'agrees' | 'unrelated'; confidence: number }>;
    /** Items dropped for invalid output. */
    invalid: number;
}

/** A routed model call. Hosts implement it (VS Code/on-prem: local engines → gateway; web: POST …/knowledge/judge). */
export interface JudgementPort {
    judge(req: JudgementRequest): Promise<JudgementResult>;
}

/** Progress an analyzer reports while it runs (files listed/read, tools finished). */
export interface AnalyzerProgress { done: number; total: number; path?: string }

/** Host-run tools for a pack (code: dependency-cruiser/jscpd/knip). Returns extra extracted edges/badges. `onProgress` is optional to call and to honour. */
export interface AnalyzerPort {
    analyse(rootId: string, signal?: AbortSignal, onProgress?: (p: AnalyzerProgress) => void): Promise<Extracted[]>;
}

/** A host-supplied parser that also extracts graph structure. `extract` is the optional ingest extension. */
export interface KnowledgeParser {
    id: string;
    accepts(source: SourceDocument): boolean;
    parse(source: SourceDocument): ParsedSection[];
    extract?(source: SourceDocument, sections: ParsedSection[], ctx: ExtractContext): Extracted;
}

export interface KnowledgePack {
    /** 'docs' | 'code' */
    id: string;
    parsers: KnowledgeParser[];
    partitions?: PartitionRegistry;
    findings?: FindingRule[];
    judgement?: JudgementSpec;
    analyzers?: AnalyzerPort;
}

export interface StoredFact {
    /** project_facts.key — judgement nodes use `ent:<pack>:<name>` / `claim:<hash>` namespaces. */
    key: string;
    content: string;
    source: string;
    importance: number;
    origin: StoredOrigin;
    fileAnchors: Array<{ path: string; line?: number; contentHash?: string }>;
    updatedAt?: string;
    createdBy?: string | null;   // users.id is varchar(36)
}

export interface StoredRelation {
    from: NodeRef;
    to: NodeRef;
    kind: string;
    weight: number;
    origin: StoredOrigin;
    confidence?: number;
    sourceHash?: string;
    staleAt?: string | null;
}

export interface Scope {
    rootId: string;
}

export interface ScopeRecord {
    rootId: string;
    title: string;
    packs: string[];
    lastDigest?: string;
    lastReviewedAt?: string;
}

/** Where knowledge is persisted: the platform API adapter or the local snapshot adapter. */
export interface KnowledgeStore {
    scopes(): Promise<ScopeRecord[]>;
    facts(scope: Scope): Promise<StoredFact[]>;
    relations(scope: Scope): Promise<StoredRelation[]>;
    write(batch: { scope: ScopeRecord; facts: StoredFact[]; relations: StoredRelation[] }): Promise<void>;
    /** Marks model facts/relations judged from `sourceRef` at a hash other than `currentHash` stale. */
    markStale(scope: Scope, sourceRef: NodeRef, currentHash: string): Promise<void>;
    /** Extracted output cache, by content hash (never knowledge — perception). */
    getExtracted?(hash: string): Promise<Extracted[] | undefined>;
    putExtracted?(hash: string, extracted: Extracted[]): Promise<void>;
    /**
     * A hint that `getExtracted` is about to be asked for these hashes, so a remote store can fetch
     * them in bulk; afterwards `getExtracted` for a preloaded hash is answered without a round trip.
     * Only a hint: `getExtracted` stays correct without it, and a failure propagates like any store failure.
     */
    preloadExtracted?(hashes: readonly string[]): Promise<void>;
    /**
     * The re-check channel between a replica and the review, through the one store both already share: a replica that
     * is asked for the body of a hash it holds none of calls `markRecheck(hashes)`; the next review reads ONLY the
     * manifest files with those hashes (binary → an explicit `[]`, text → extracted), then `clearRecheck`s them. A store
     * that cannot persist the set keeps it in memory. Optional like the cache methods; a store without them has no channel.
     */
    markRecheck?(hashes: readonly string[]): Promise<void>;
    recheckHashes?(): Promise<string[]>;
    clearRecheck?(hashes: readonly string[]): Promise<void>;
    /** Whether a hash is on the re-check set (a cheap membership test for a store that observes every put). */
    hasRecheck?(hash: string): Promise<boolean>;
    /**
     * Runs `work` as one unit of many mutations. Inside it the store may hold persistence back
     * (a whole-snapshot store would otherwise rewrite everything per mutation); `flush` makes
     * everything so far durable; when `work` settles, resolved or rejected, the store is durable.
     * Optional: a store without it persists per mutation.
     */
    batched?<T>(work: (flush: () => Promise<void>) => Promise<T>): Promise<T>;
}

export interface LiveOverlay {
    changes: Array<{ path: string; status: string; added: number; removed: number }>;
    /** Files touched together in one run → co-changed edges. */
    runFiles?: string[];
}

/** A workspace package of a system. */
export interface GraphApp { path: string; name: string }

/** Why a file the repo graph could have read was not read. */
export interface GraphReadSkip { reason: string; count: number }

/**
 * One system of a review root (a repository, or a folder whose manifest names a project): the unit
 * the reference graph, layer verdicts and partition violations are computed within (PRD 41 A9).
 * `path` is root-relative, '' when the root itself is the system.
 */
export interface GraphSystem {
    path: string;
    name: string;
    /** File nodes the graph holds under it, not counting assets (binary files, see `assets`). */
    files: number;
    /** Asset file nodes (images, fonts, archives …) under it; absent when it has none. Not part of `files`. */
    assets?: number;
    apps: GraphApp[];
    /** The repo graph did not read every file of it (see `graphRead.skipped`). */
    truncated?: boolean;
    /** Set on a duplicated copy: the path of the system it copies. */
    copyOf?: string;
    copySimilarity?: number;
    copyReason?: 'no-git' | 'name-mismatch' | 'later-name';
    /** The reference graph's reading of this system: files read, and files left out by reason. */
    graphRead?: { read: number; skipped: GraphReadSkip[] };
}

export interface KnowledgeGraph {
    scope: { rootId: string; packs: string[]; digest: string };
    nodes: KNode[];
    edges: KEdge[];
    findings: Finding[];
    /** Present only when the root held systems to report (see {@link GraphSystem}). */
    meta?: { systems?: GraphSystem[] };
    stats: {
        sources: number; nodes: number; edges: number; stale: number; truncated: boolean;
        /** Q of the file communities (PRD 38 P1), 6 decimals; absent on graphs stored before it existed. */
        modularity?: number;
        /** Which algorithm produced the communities (label propagation above COMMUNITY_LOUVAIN_MAX_NODES connected files). */
        communityMethod?: 'louvain' | 'label-propagation';
    };
    /**
     * Attached only when the graph was truncated (`stats.truncated`): the whole graph's files,
     * so an overview never depends on the cut. Consumers read it through `outlineOf`, never directly.
     */
    outline?: GraphOutline;
}

/**
 * The whole graph's files and file-to-file references, in a compact form. Read it with
 * `outlineOf(graph)` (the one entry point: the attached outline, or one built from an untruncated
 * graph) and resolve a file's ref with `outlineRef`.
 */
export interface GraphOutline {
    /** Common ref prefix of every file (e.g. `src:<root>::`); a file's ref = prefix + path. '' when files share none. */
    prefix: string;
    /** Distinct area refs, sorted; `OutlineFile.area` indexes into it. */
    areas: NodeRef[];
    /** Distinct partitions, sorted; `OutlineFile.partition` indexes into it. */
    partitions: string[];
    /** Distinct system paths of the files (and the systems they copy), sorted; `OutlineFile.system` / `copyOf` index into it. Absent when no file has a system. */
    systems?: string[];
    /** Distinct app paths, sorted; `OutlineFile.app` indexes into it. Absent when no file has an app. */
    apps?: string[];
    /** The root's systems with their file counts, copies and graph reads — `KnowledgeGraph.meta.systems`, carried so the outline stands alone. */
    systemInfo?: GraphSystem[];
    /**
     * The communities the files' `communityId` badge indexes (PRD 38 P1), sorted by id. Absent when no file has one.
     * Ids are per build, not stable across builds.
     */
    communities?: OutlineCommunity[];
    /** File nodes of the full graph, sorted by ref. */
    files: OutlineFile[];
    /**
     * File-to-file `references` of the full graph, as [from, to, verdict, flags]: from/to index `files`;
     * verdict is a code into OUTLINE_VERDICTS (the worst verdict of the merged edges); flags bit 1 = cycle.
     * Section/symbol endpoints are rolled up to their file. Sorted by [from, to].
     */
    links: Array<[number, number, number, number]>;
    /** True when files or links were capped (see OUTLINE_MAX_*). */
    capped: boolean;
}

/** One community of the file reference graph: `id` is the `communityId` badge; `label` the members' common directory (else the top member's path). */
export interface OutlineCommunity {
    id: number;
    /** Unique within the table. */
    label: string;
    /** Members counted in THIS outline: smaller than the full community when the outline is capped (`capped`). */
    size: number;
}

export interface OutlineFile {
    /** The ref without the outline's prefix. */
    path: string;
    /** Index into `GraphOutline.areas`, when the file has an area. */
    area?: number;
    /** Index into `GraphOutline.partitions`, when the file has a partition. */
    partition?: number;
    /** Index into `GraphOutline.systems`, when the file has a system ('' is a system too). */
    system?: number;
    /** Index into `GraphOutline.apps`, when the file has an app. */
    app?: number;
    /** Index into `GraphOutline.systems` of the system this file's system copies. */
    copyOf?: number;
    importance: number;
    /** The first 12 characters of the node's sourceHash. */
    hash?: string;
    /** Only the overview's badge keys (see OUTLINE_BADGE_KEYS), when present. */
    badges?: Record<string, number | string>;
}

export interface BuildGraphInput {
    rootId: string;
    extracted: Extracted[];
    facts: StoredFact[];
    relations: StoredRelation[];
    packs: KnowledgePack[];
    live?: LiveOverlay;
    nearDuplicates?: SimilarPair[];
    /** The root's systems (file counts are computed by the build): each file node is stamped with its system, app and copyOf. */
    systems?: GraphSystem[];
}

export interface BuildGraphOptions {
    /** Default 400; cap 2000. */
    maxNodes?: number;
    focus?: NodeRef[];
}

export interface TraceStep {
    from: NodeRef;
    to: NodeRef;
    depth: number;
    kind: string;
    verdict?: EdgeVerdict;
    cycle?: boolean;
    back?: boolean;
}

export interface Trace {
    from: NodeRef;
    /** Partitions present, in registry order (lanes). */
    lanes: string[];
    steps: TraceStep[];
    truncated: boolean;
}

/** Re-exported for card consumers. */
export type { IndexDoc };
/** The review's ports and reports live beside it (reviewModel.ts); re-exported so one import serves. */
export type {
    DirectorySource, ReviewPlan, ReviewProgress, ReviewReport, ReviewArgs,
} from './reviewModel.js';
