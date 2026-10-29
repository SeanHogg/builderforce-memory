/**
 * knowledge/reviewCompare.ts — the compare phase of a review (PRD 38 §4.3): near-duplicate
 * claim pairs → `contradicts` / `agrees`. Claims whose pairs could not all be compared
 * (budget, a failed call, an abort) are kept in the compare manifest, so *Continue* picks
 * them up at the next run instead of losing them.
 */

import type { IngestManifest } from '../ingest/types.js';
import { claimTextOf, compareClaims, type CompareOutcome, type JudgeBudget, type JudgedBatch } from './judge.js';
import type { NodeRef, ReviewArgs, StoredFact } from './model.js';
import { compareManifestKey } from './sources.js';
import { compareStr, factKeyOf } from './refs.js';

/** At most this many claims wait for *Continue*; beyond it the oldest are dropped (the set stays bounded). */
export const MAX_AWAITING_CLAIMS = 2000;

/** A claim awaiting comparison: its text and the content hash it was judged from. */
export type PendingClaims = Map<string, { text: string; hash: string }>;

/**
 * The compare manifest persists age with each entry — `hash` is `<age>:<fileHash>`, age 0 the
 * oldest — because a host manifest may hand entries back in id order. An entry without an age
 * (written before ages were kept) reads as the oldest.
 */
const ageOf = (hash: string): number => {
    const sep = hash.indexOf(':');
    const age = sep < 0 ? NaN : Number(hash.slice(0, sep));
    return Number.isFinite(age) ? age : -1;
};

/** Claims a previous run left awaiting comparison, oldest first, read back from their stored facts (a gone fact drops out). */
export async function pendingClaims(manifest: IngestManifest, rootId: string, facts: ReadonlyMap<string, StoredFact>): Promise<PendingClaims> {
    const out: PendingClaims = new Map();
    const stored = (await manifest.get(compareManifestKey(rootId))) ?? [];
    const oldestFirst = [...stored].sort((a, b) => ageOf(a.hash) - ageOf(b.hash) || compareStr(a.id, b.id));
    for (const { id: ref } of oldestFirst) {
        const fact = facts.get(factKeyOf(ref as NodeRef) ?? '');
        if (fact?.key.startsWith('claim:')) out.set(ref, { text: claimTextOf(fact.content), hash: fact.fileAnchors[0]?.contentHash ?? '' });
    }
    return out;
}

/**
 * Compares `pending` against its near-duplicates, within the budget. Undefined when the
 * phase cannot run (no judgement port, no index, nothing pending).
 */
export async function compareNew(a: {
    args: ReviewArgs; pending: PendingClaims; known: ReadonlyMap<string, StoredFact>; budget: JudgeBudget; batch: JudgedBatch;
    onPairs(total: number): void;
}): Promise<CompareOutcome | undefined> {
    const { args, pending } = a;
    if (!args.judgement || !args.similar || pending.size === 0) return undefined;
    const pairs = await args.similar([...pending.keys()].sort(compareStr), new Map([...pending].map(([ref, c]) => [ref, c.text] as const)));
    const claims: PendingClaims = new Map(pending);
    for (const p of pairs) {
        for (const id of [p.a, p.b]) {
            const fact = !claims.has(id) ? a.known.get(factKeyOf(id as NodeRef) ?? '') : undefined;
            if (fact?.key.startsWith('claim:')) claims.set(id, { text: claimTextOf(fact.content), hash: fact.fileAnchors[0]?.contentHash ?? '' });
        }
    }
    a.onPairs(pairs.length);
    // Run even with the budget spent: compareClaims counts every pair it skips as unjudged.
    return compareClaims({ port: args.judgement, pairs, claims, budget: a.budget, batch: a.batch, ...(args.signal ? { signal: args.signal } : {}) });
}

/**
 * Persists the claims still awaiting comparison, capped at {@link MAX_AWAITING_CLAIMS} (the
 * most recent: a previous run's first, this run's last — and the order survives the reload,
 * see {@link pendingClaims}). Call it only when a pack asks for contradictions; an `outcome`
 * undefined means the compare did not run (no ports, an abort before it), so every pending
 * claim is kept, whereas after a compare only the pairs it skipped still wait.
 */
export async function saveAwaiting(
    manifest: IngestManifest, rootId: string, pending: PendingClaims, outcome: CompareOutcome | undefined,
): Promise<void> {
    const awaiting: Array<[string, string]> = [];
    for (const [ref, c] of pending) if (!outcome || outcome.skipped.has(ref)) awaiting.push([ref, c.hash]);
    const kept = awaiting.slice(-MAX_AWAITING_CLAIMS);
    await manifest.set(compareManifestKey(rootId), kept.map(([id, hash], age) => ({ id, hash: `${age}:${hash}` })));
}
