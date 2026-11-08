/**
 * knowledge/backfillNoExtract.ts — settles the files a root holds with no stored body, so a replica can tell
 * "nothing is extracted for this file" from "not extracted yet".
 *
 * A review stores a file node for an ASSET (a binary file, assetFile.ts), but a root reviewed before that existed names such
 * files in its manifest with no body, and no later review visits them again. Two cases, neither a sweep:
 *
 *   - ASSET by extension (planned with `asset`): no file is read. A manifest entry at the file's current bytes with no stored body
 *     queues the file, so the review writes its node; an older hash of the path with no body gets an explicit `[]`.
 *   - binary by CONTENT (a NUL) or any other body missing: unknowable without reading. A replica that holds a file back, or is told by
 *     the server that a body is missing, calls `store.markRecheck(hashes)`; the unchanged planned files with those hashes are queued
 *     for the review (`changed`), which stores the body or the asset node. A hash only a SKIPPED prior entry carries is never extracted
 *     by any review: `[]`. Hashes settled HERE are cleared here. A queued one is cleared by the review only once its body is stored or
 *     a FINAL failure is recorded; a transient failure or a stop keeps the request.
 *
 * A healthy root has no body missing and an empty recheck set: it reads nothing and costs only in-memory lookups.
 */

import type { KnowledgeStore } from './model.js';
import { putNoExtract } from './extractFile.js';
import type { PlanDetail } from './plan.js';

export interface BackfillResult {
    /** Hashes stored as an explicit `[]`. */
    recorded: number;
    /** Files queued for extraction (marked `changed` in the plan). */
    queued: string[];
}

/** The hashes the backfill will look up (for the review's ONE bulk preload): the planned assets' manifest entries, and the re-check set. */
export function backfillCandidates(detail: PlanDetail, asked: readonly string[]): string[] {
    const assets = detail.files.filter((f) => f.asset).map((f) => detail.prior.get(f.path)).filter((h): h is string => !!h);
    return [...new Set([...assets, ...asked])];
}

export async function backfillNoExtract(args: { store: KnowledgeStore; detail: PlanDetail; asked: readonly string[]; only?: readonly string[] | undefined; signal?: AbortSignal | undefined }): Promise<BackfillResult> {
    const { store, detail, signal } = args;
    const asked = new Set(args.asked);
    const out: BackfillResult = { recorded: 0, queued: [] };
    if (!store.getExtracted || !store.putExtracted) return out;

    for (const f of detail.files) {
        if (!f.asset) continue;
        if (signal?.aborted) return out;
        const prior = detail.prior.get(f.path);
        if (!prior || (await store.getExtracted(prior)) !== undefined) continue;
        if (prior === f.hash) {
            // The manifest holds the asset at its current bytes but the store holds no body for it: the review writes its node.
            if (!f.changed && (!args.only || args.only.includes(f.path))) { f.changed = true; out.queued.push(f.path); }
        } else {
            await putNoExtract(store, prior); // an older hash of the path: nothing more will be extracted under it
            out.recorded += 1;
        }
    }
    if (asked.size === 0) return out;

    const skippedHashes = new Set(detail.plan.skipped.map((s) => detail.prior.get(s.path)).filter((h): h is string => !!h));
    const planned = new Map<string, typeof detail.files>();
    for (const f of detail.files) planned.set(f.hash, [...(planned.get(f.hash) ?? []), f]);
    const done = new Set<string>();
    for (const hash of asked) {
        if (signal?.aborted) break;
        if ((await store.getExtracted(hash)) !== undefined) { done.add(hash); continue; }
        const files = planned.get(hash);
        if (!files) {
            // Only a skipped prior entry (ignored, too large, no pack) carries it: no review will ever extract it. Otherwise it is gone.
            if (skippedHashes.has(hash)) { await putNoExtract(store, hash); out.recorded += 1; }
            done.add(hash);
            continue;
        }
        const todo = files.filter((f) => !f.changed && detail.prior.get(f.path) === f.hash);
        if (todo.length === 0) continue; // a changed file is handled by this review, which clears the request once its work is done (`settledRecheck`)
        // Pass 2 works only on `only`: a file it will not process is not queued here, and its hash stays asked for the run that will.
        if (args.only && !todo.some((t) => args.only!.includes(t.path))) continue;
        // The review decides what the file is (text → its body, a NUL → its asset node, an unreadable file → its one failure path) and stores
        // it; the request is asked for until then (see `settledRecheck`). No read here: a file found binary needs a node, not a bare `[]`.
        for (const t of todo) { t.changed = true; out.queued.push(t.path); }
    }
    if (done.size > 0 && !signal?.aborted) await store.clearRecheck?.([...done]);
    return out;
}
