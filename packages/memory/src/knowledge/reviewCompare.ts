/**
 * knowledge/reviewCompare.ts — the compare phase of a review (PRD 38 §4.3): near-duplicate
 * claim pairs → `contradicts` / `agrees`. Claims whose pairs could not all be compared
 * (budget, a failed call, an abort) are kept in the compare manifest, so *Continue* picks
 * them up at the next run instead of losing them.
 */

import type { IngestManifest } from '../ingest/types.js';
import { claimTextOf, compareClaims, type CompareOutcome, type JudgeBudget, type JudgedBatch } from './judge.js';
import type { NodeRef, ReviewArgs, StoredFact } from './model.js';
import { compareManifestKey, manifestMap, saveManifestMap } from './sources.js';
import { compareStr, factKeyOf } from './refs.js';

/** At most this many claims wait for *Continue*; beyond it the oldest are dropped (the set stays bounded). */
export const MAX_AWAITING_CLAIMS = 2000;

/** A claim awaiting comparison: its text and the content hash it was judged from. */
export type PendingClaims = Map<string, { text: string; hash: string }>;

/** Claims a previous run left awaiting comparison, read back from their stored facts (a gone fact drops out). */
export async function pendingClaims(manifest: IngestManifest, rootId: string, facts: ReadonlyMap<string, StoredFact>): Promise<PendingClaims> {
    const out: PendingClaims = new Map();
    for (const ref of (await manifestMap(manifest, compareManifestKey(rootId))).keys()) {
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
    const pairs = await args.similar([...pending.keys()].sort(compareStr));
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
 * most recent: a previous run's first, this run's last). With `canCompare` false — no
 * judgement port, no `similar` port, or no pack asking for contradictions — the phase cannot
 * run at all, so nothing waits: it is not the same as an abort, which keeps every claim.
 */
export async function saveAwaiting(
    manifest: IngestManifest, rootId: string, pending: PendingClaims, outcome: CompareOutcome | undefined, canCompare: boolean,
): Promise<void> {
    const awaiting = new Map<string, string>();
    if (canCompare) {
        for (const [ref, c] of pending) if (!outcome || outcome.skipped.has(ref)) awaiting.set(ref, c.hash);
    }
    const kept = awaiting.size > MAX_AWAITING_CLAIMS ? new Map([...awaiting].slice(-MAX_AWAITING_CLAIMS)) : awaiting;
    await saveManifestMap(manifest, compareManifestKey(rootId), kept);
}
