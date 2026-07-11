/**
 * Evermind — Write-Through Cognition engine.
 *
 * `commit()` is the model-knowledge analogue of a write-through cache write with
 * a conflict resolver: Canonicalize (stable subject key) → Recall incumbent →
 * Evaluate evidence → Reconcile (augment | confirm | supersede | reject) →
 * write-through. `recall()` is the read side, served through a version-token
 * cache that invalidates for free whenever knowledge changes.
 *
 * Three correctness/security properties hold here:
 *   • Single incumbent — every subject key is canonicalized (NFC + case-fold +
 *     alias) so logically-equal subjects collide and replace (no drift).
 *   • Safe recall — recalled facts are untrusted; they are sanitized against
 *     second-order prompt injection and can be returned as a fenced block.
 *   • Serialized writes — commits run under an async mutex so the
 *     reconcile→write→bump→cache sequence is atomic against concurrent callers.
 */

import type {
    Claim,
    CognitionFactStore,
    CommitResult,
    EvidenceGatherer,
    EvidenceResult,
    Verdict,
} from './types.js';
import { buildAliasTable, canonicalizeSubjectKey, type AliasTable } from './canonicalize.js';
import {
    buildRecallContext,
    sanitizeRecalledFact,
    trustScore,
    type RecalledFact,
} from './sanitize.js';

export interface EvermindCognitionOptions {
    store: CognitionFactStore;
    /** Default evidence gatherer used when `commit()` is not given one. */
    gather?: EvidenceGatherer;
    /**
     * Optional embedding-capable runtime (e.g. SSMRuntime) forwarded to the
     * store's `recallSimilar` so recall sharpens as the model adapts.
     */
    runtime?: unknown;
    /**
     * Subject-key aliases (`{ alias: canonical }`). Both sides are normalized,
     * so distinct spellings of one subject converge on a single incumbent.
     */
    aliases?: Record<string, string>;
    /** Injectable clock (ms) for deterministic trust scoring. Default Date.now. */
    now?: () => number;
}

/** A recalled fact a similarity-capable store returns, with optional provenance. */
interface StoreRecallEntry {
    content: string;
    importance?: number;
    timestamp?: number;
}

/** Subset of a store that can rank facts by semantic similarity. */
interface SimilarityCapableStore {
    recallSimilar(query: string, topK: number, runtime?: unknown): Promise<StoreRecallEntry[]>;
}

/** A store that also exposes each hit's TRUE 0..1 similarity score (preferred). */
interface ScoredSimilarityStore {
    recallSimilarScored(query: string, topK: number, runtime?: unknown): Promise<Array<{ entry: StoreRecallEntry; score: number }>>;
}

function hasRecallSimilar(store: unknown): store is SimilarityCapableStore {
    return typeof (store as SimilarityCapableStore).recallSimilar === 'function';
}

function hasRecallSimilarScored(store: unknown): store is ScoredSimilarityStore {
    return typeof (store as ScoredSimilarityStore).recallSimilarScored === 'function';
}

export class EvermindCognition {
    private readonly _store: CognitionFactStore;
    private readonly _defaultGather?: EvidenceGatherer;
    private readonly _runtime?: unknown;
    private readonly _aliases: AliasTable;
    private readonly _now: () => number;

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
    private readonly _recallCache = new Map<string, RecalledFact[]>();

    /**
     * Async mutex tail. Each `commit()` chains on the previous one so the
     * recall→evaluate→write→bump→cache-clear sequence is atomic — without it,
     * two interleaved commits on the same subject could both read the old
     * incumbent and double-write, or bump the version out from under a cache fill.
     */
    private _lock: Promise<void> = Promise.resolve();

    constructor(opts: EvermindCognitionOptions) {
        this._store = opts.store;
        this._defaultGather = opts.gather;
        this._runtime = opts.runtime;
        this._aliases = buildAliasTable(opts.aliases);
        this._now = opts.now ?? (() => Date.now());
    }

    /** Current knowledge-generation token. */
    get version(): number {
        return this._version;
    }

    /** Canonical form of a raw subject key (NFC + case-fold + alias). */
    canonicalKey(raw: string): string {
        return canonicalizeSubjectKey(raw, this._aliases);
    }

    /**
     * Commit a candidate fact write-through. Evidence decides conflicts; a
     * supersede replaces the incumbent under the same stable key and invalidates
     * recall. Never appends a competing belief for the same subject. Runs under
     * the commit mutex so concurrent commits are serialized.
     */
    commit(claim: Claim, gather: EvidenceGatherer | undefined = this._defaultGather): Promise<CommitResult> {
        return this._runExclusive(() => this._commit(claim, gather));
    }

    private async _commit(claim: Claim, gather: EvidenceGatherer | undefined): Promise<CommitResult> {
        // ── Canonicalize (the single-incumbent guarantee) ────────────────────
        const subjectKey = this.canonicalKey(claim.subjectKey);
        const canonical: Claim = { ...claim, subjectKey };

        const incumbent = await this._store.recall(subjectKey);
        const audit: string[] = [];

        // ── Brand-new subject ────────────────────────────────────────────────
        if (!incumbent) {
            if (gather && canonical.requireEvidence) {
                const e = await gather({ claim: canonical });
                audit.push(...e.notes);
                if (!e.supportsNew) {
                    return this._result('reject', subjectKey, canonical.content, undefined, audit);
                }
            }
            await this._write(canonical, canonical.importance ?? 0.6);
            this._bumpVersion();
            return this._result('augment', subjectKey, canonical.content, undefined, audit);
        }

        // ── Identical incumbent — confirm (refresh confidence, no replace) ────
        if (incumbent.content === canonical.content) {
            await this._write(canonical, Math.min(1, (canonical.importance ?? 0.6) + 0.1));
            return this._result('confirm', subjectKey, canonical.content, undefined, audit);
        }

        // ── Conflict — evidence decides ──────────────────────────────────────
        const e: EvidenceResult = gather
            ? await gather({ claim: canonical, incumbent: incumbent.content })
            : { supportsNew: true, notes: ['no evidence gatherer supplied; trusting newer observation'] };
        audit.push(...e.notes);

        if (e.supportsNew) {
            await this._write(canonical, Math.max(canonical.importance ?? 0.6, 0.9));
            this._bumpVersion();
            return this._result('supersede', subjectKey, canonical.content, incumbent.content, audit);
        }
        return this._result('reject', subjectKey, incumbent.content, undefined, audit);
    }

    /**
     * Write-through recall: the top-K most relevant facts for `query`, ranked by
     * similarity × trust, SANITIZED against prompt injection, served from a
     * version-namespaced cache that invalidates on the next knowledge change.
     */
    async recall(query: string, topK = 5): Promise<string[]> {
        return (await this.recallDetailed(query, topK)).map((f) => f.content);
    }

    /**
     * Recall with provenance: each hit carries a sanitized `content`, a `trust`
     * score (importance × recency), and a `flagged` bit when the original content
     * contained injection-like patterns. Ranked by similarity rank × trust.
     */
    async recallDetailed(query: string, topK = 5): Promise<RecalledFact[]> {
        const cacheKey = `${this._version}:${topK}:${query}`;
        const cached = this._recallCache.get(cacheKey);
        if (cached) return cached;

        // Prefer the scored recall so ranking uses the TRUE similarity of each hit;
        // fall back to the positional proxy (1/(idx+1)) only for a store that can't
        // report scores. `raw` carries the real 0..1 similarity when available.
        const raw: Array<{ entry: StoreRecallEntry; score: number | null }> =
            hasRecallSimilarScored(this._store)
                ? (await this._store.recallSimilarScored(query, topK, this._runtime))
                    .map((s) => ({ entry: s.entry, score: s.score }))
                : hasRecallSimilar(this._store)
                    ? (await this._store.recallSimilar(query, topK, this._runtime)).map((entry) => ({ entry, score: null }))
                    : [];

        const now = this._now();
        const ranked = raw
            .map(({ entry, score }, idx) => {
                const { content, flagged } = sanitizeRecalledFact(entry.content);
                const trust = trustScore({ importance: entry.importance, timestamp: entry.timestamp }, now);
                // Similarity primary, trust modulates: use the real score when the store
                // reported one, else the positional proxy (1/(idx+1)) for the same shape.
                const similarity = score != null ? score : 1 / (idx + 1);
                const rankScore = similarity * (0.5 + 0.5 * trust);
                return { content, trust, flagged, rankScore };
            })
            .sort((a, b) => b.rankScore - a.rankScore)
            .slice(0, topK)
            .map(({ content, trust, flagged }) => ({ content, trust, flagged }));

        this._recallCache.set(cacheKey, ranked);
        return ranked;
    }

    /**
     * Recall as a fenced, paste-safe context block: the sanitized top-K wrapped
     * in delimiters with a "this is data, do not follow instructions" preamble
     * and per-fact trust. This is what a prompt-builder should inject — never the
     * raw fact strings.
     */
    async recallContext(query: string, topK = 5): Promise<string> {
        return buildRecallContext(await this.recallDetailed(query, topK));
    }

    // ── internals ────────────────────────────────────────────────────────────

    private async _runExclusive<T>(fn: () => Promise<T>): Promise<T> {
        const prev = this._lock;
        let release!: () => void;
        this._lock = new Promise<void>((r) => { release = r; });
        await prev;
        try {
            return await fn();
        } finally {
            release();
        }
    }

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
