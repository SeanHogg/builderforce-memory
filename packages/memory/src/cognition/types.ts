/**
 * Evermind — Write-Through Cognition types.
 *
 * The cognition layer is what lets the model's KNOWLEDGE stay current without a
 * reconciliation step: every incoming fact is evaluated against EVIDENCE and the
 * incumbent belief, then committed write-through (replace-on-write by a STABLE
 * subject key) instead of being appended under a fresh per-run key. These types
 * are intentionally store- and surface-agnostic so the same logic runs in the
 * IDE, on-prem, cloud, and the browser.
 */

/** Outcome of evaluating a candidate fact against its incumbent + evidence. */
export type Verdict =
    /** No incumbent on this subject — stored as a new belief. */
    | 'augment'
    /** Incumbent identical — recency/confidence refreshed, nothing replaced. */
    | 'confirm'
    /** Incumbent conflicted and evidence favoured the new fact — replaced. */
    | 'supersede'
    /** Incumbent conflicted but evidence did NOT favour the new fact — dropped. */
    | 'reject';

/**
 * Minimal write-through fact store the cognition layer needs. `MemoryStore`
 * satisfies this structurally (no adapter required) — kept narrow so cloud
 * (Postgres) / browser (IndexedDB) backends can satisfy it too.
 */
export interface CognitionFactStore {
    remember(
        key: string,
        content: string,
        opts?: { tags?: string[]; importance?: number; ttlMs?: number },
    ): Promise<void>;
    recall(key: string): Promise<{ content: string } | undefined>;
    forget(key: string): Promise<void>;
}

/** A fact asserted about a subject, keyed by a STABLE canonical key. */
export interface Claim {
    /**
     * STABLE canonical key naming the SUBJECT of the fact (e.g. `pkg:ssm-stack`).
     * This is the anti-drift fix: logically-superseding facts collide on this key
     * and replace, instead of accumulating under per-run keys.
     */
    subjectKey: string;
    /** The asserted fact content. */
    content: string;
    tags?: string[];
    importance?: number;
    /**
     * When true, a brand-new subject is only stored if the gatherer's evidence
     * supports it (guards against recording unverified first-observations).
     */
    requireEvidence?: boolean;
}

/** The verdict of an evidence probe: does ground truth favour the new claim? */
export interface EvidenceResult {
    supportsNew: boolean;
    /** Audit-readable lines describing what was checked and found. */
    notes: string[];
}

export interface EvidenceContext {
    claim: Claim;
    /** Incumbent belief content for this subject, when one exists. */
    incumbent?: string;
}

/**
 * Gathers ground-truth evidence for a claim. Injected by the caller because the
 * evidence source is surface-specific (IDE file tools, cloud DB, HTTP probe…).
 */
export type EvidenceGatherer = (ctx: EvidenceContext) => Promise<EvidenceResult>;

/** Result of committing a claim through the cognition pipeline. */
export interface CommitResult {
    verdict: Verdict;
    subjectKey: string;
    /** The belief now held for this subject (incumbent's content on reject). */
    content: string;
    /** Prior content, present only when `verdict === 'supersede'`. */
    superseded?: string;
    /** Evidence audit trail. */
    evidence: string[];
    /** Knowledge-generation token after this commit (bumps on augment/supersede). */
    version: number;
}
