/**
 * Evermind — Write-Through Cognition engine.
 *
 * `commit()` is the model-knowledge analogue of a write-through cache write with
 * a conflict resolver: Canonicalize (stable subject key) → Recall incumbent →
 * Evaluate evidence → Reconcile (augment | confirm | supersede | reject) →
 * write-through. `recall()` is the read side, served through a version-token
 * cache that invalidates for free whenever knowledge changes.
 *
 * This is the layer the append-only knowledge loop skipped: it derives a STABLE
 * subject key (not a per-run id) and replaces-on-write, so beliefs never drift
 * into a manual reconciliation step.
 */

import type {
    Claim,
    CognitionFactStore,
    CommitResult,
    EvidenceGatherer,
    EvidenceResult,
    Verdict,
} from './types.js';

export interface EvermindCognitionOptions {
    store: CognitionFactStore;
    /** Default evidence gatherer used when `commit()` is not given one. */
    gather?: EvidenceGatherer;
    /**
     * Optional embedding-capable runtime (e.g. SSMRuntime) forwarded to the
     * store's `recallSimilar` so recall sharpens as the model adapts.
     */
    runtime?: unknown;
}

/** Subset of a store that can rank facts by semantic similarity. */
interface SimilarityCapableStore {
    recallSimilar(query: string, topK: number, runtime?: unknown): Promise<Array<{ content: string }>>;
}

function hasRecallSimilar(store: unknown): store is SimilarityCapableStore {
    return typeof (store as SimilarityCapableStore).recallSimilar === 'function';
}

export class EvermindCognition {
    private readonly _store: CognitionFactStore;
    private readonly _defaultGather?: EvidenceGatherer;
    private readonly _runtime?: unknown;

    /**
     * Knowledge-generation token. Bumped on every change to the fact set
     * (augment / supersede). Doubles as the recall-cache namespace so a write
     * invalidates cached reads for free — the "invalidate on write" rule applied
     * to model knowledge.
     */
    private _version = 0;

    /**
     * L1 recall cache, namespaced by `_version`. After any knowledge change the
     * version moves, so prior-generation entries are never read again (and are
     * cleared to bound growth) — not an ad-hoc TTL map; a version-token cache.
     */
    private readonly _recallCache = new Map<string, string[]>();

    constructor(opts: EvermindCognitionOptions) {
        this._store = opts.store;
        this._defaultGather = opts.gather;
        this._runtime = opts.runtime;
    }

    /** Current knowledge-generation token. */
    get version(): number {
        return this._version;
    }

    /**
     * Commit a candidate fact write-through. Evidence decides conflicts; a
     * supersede replaces the incumbent under the same stable key and invalidates
     * recall. Never appends a competing belief for the same subject.
     */
    async commit(claim: Claim, gather: EvidenceGatherer | undefined = this._defaultGather): Promise<CommitResult> {
        const incumbent = await this._store.recall(claim.subjectKey);
        const audit: string[] = [];

        // ── Brand-new subject ────────────────────────────────────────────────
        if (!incumbent) {
            if (gather && claim.requireEvidence) {
                const e = await gather({ claim });
                audit.push(...e.notes);
                if (!e.supportsNew) {
                    return this._result('reject', claim.subjectKey, claim.content, undefined, audit);
                }
            }
            await this._write(claim, claim.importance ?? 0.6);
            this._bumpVersion();
            return this._result('augment', claim.subjectKey, claim.content, undefined, audit);
        }

        // ── Identical incumbent — confirm (refresh confidence, no replace) ────
        if (incumbent.content === claim.content) {
            await this._write(claim, Math.min(1, (claim.importance ?? 0.6) + 0.1));
            return this._result('confirm', claim.subjectKey, claim.content, undefined, audit);
        }

        // ── Conflict — evidence decides ──────────────────────────────────────
        const e: EvidenceResult = gather
            ? await gather({ claim, incumbent: incumbent.content })
            : { supportsNew: true, notes: ['no evidence gatherer supplied; trusting newer observation'] };
        audit.push(...e.notes);

        if (e.supportsNew) {
            await this._write(claim, Math.max(claim.importance ?? 0.6, 0.9));
            this._bumpVersion();
            return this._result('supersede', claim.subjectKey, claim.content, incumbent.content, audit);
        }
        return this._result('reject', claim.subjectKey, incumbent.content, undefined, audit);
    }

    /**
     * Write-through recall: the top-K most relevant facts for `query`, served
     * from a version-namespaced cache that invalidates on the next knowledge
     * change. Falls back to an empty set when the store can't rank similarity.
     */
    async recall(query: string, topK = 5): Promise<string[]> {
        const cacheKey = `${this._version}:${topK}:${query}`;
        const cached = this._recallCache.get(cacheKey);
        if (cached) return cached;

        const facts = hasRecallSimilar(this._store)
            ? (await this._store.recallSimilar(query, topK, this._runtime)).map((e) => e.content)
            : [];

        this._recallCache.set(cacheKey, facts);
        return facts;
    }

    // ── internals ────────────────────────────────────────────────────────────

    private async _write(claim: Claim, importance: number): Promise<void> {
        await this._store.remember(claim.subjectKey, claim.content, { tags: claim.tags, importance });
    }

    private _bumpVersion(): void {
        this._version++;
        this._recallCache.clear();
    }

    private _result(
        verdict: Verdict,
        subjectKey: string,
        content: string,
        superseded: string | undefined,
        evidence: string[],
    ): CommitResult {
        return { verdict, subjectKey, content, superseded, evidence, version: this._version };
    }
}
