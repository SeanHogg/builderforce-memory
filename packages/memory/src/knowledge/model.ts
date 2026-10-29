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
}

/** Output of a pack's deterministic extractor (or analyzer) for one source. Never stored. */
export interface Extracted {
    sourceRef: NodeRef;
    sourceHash: string;
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
    /** Resolves a bare `[[Name]]` to a root-relative path by file name anywhere in the root (the review's listing). */
    resolveName?(name: string): string | undefined;
}

/** A finding a person can act on. `kind` is pack-defined; `requestKey` names an i18n template. */
export interface Finding {
    id: string;
    kind: string;
    pack: string;
    nodes: NodeRef[];
    severity: 'info' | 'warn' | 'error';
    /** Numbers/strings for the label and the Fix-with-agent template. */
    params: Record<string, string | number>;
    /** i18n key of the fixed Fix-with-agent request (e.g. 'knowledgeGraph.fix.cloneGroup'). */
    requestKey?: string;
}

export interface FindingRule {
    id: string;
    find(graph: KnowledgeGraph): Finding[];
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

/** Host-run tools for a pack (code: dependency-cruiser/jscpd/knip). Returns extra extracted edges/badges. */
export interface AnalyzerPort {
    analyse(rootId: string, signal?: AbortSignal): Promise<Extracted[]>;
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
}

export interface LiveOverlay {
    changes: Array<{ path: string; status: string; added: number; removed: number }>;
    /** Files touched together in one run → co-changed edges. */
    runFiles?: string[];
}

export interface KnowledgeGraph {
    scope: { rootId: string; packs: string[]; digest: string };
    nodes: KNode[];
    edges: KEdge[];
    findings: Finding[];
    stats: { sources: number; nodes: number; edges: number; stale: number; truncated: boolean };
}

export interface BuildGraphInput {
    rootId: string;
    extracted: Extracted[];
    facts: StoredFact[];
    relations: StoredRelation[];
    packs: KnowledgePack[];
    live?: LiveOverlay;
    nearDuplicates?: SimilarPair[];
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
