/**
 * knowledge/backfillNoExtract.ts — settles the files a root holds with no stored body, so a replica can tell
 * "nothing is extracted for this file" (binary: an explicit `[]`) from "not extracted yet".
 *
 * A review stores `[]` for a file it finds binary (`putNoExtract`), but a root reviewed before that existed names such
 * files in its manifest with no body, and no later review visits them again. Two cases, neither a sweep:
 *
 *   - binary by EXTENSION: the plan SKIPS the file (reason `binary`) yet the manifest may still name it. The extension is the
 *     whole test, so no file is read; only a hash with no stored body is touched. (A planned file with a binary-looking
 *     extension is a pdf/office document a host extractor handles: never `[]` by extension.)
 *   - binary by CONTENT (a NUL): unknowable without reading. A replica that holds a file back, or is told by the server that a
 *     body is missing, calls `store.markRecheck(hashes)`; the review reads ONLY the unchanged planned files with those hashes.
 *     Binary → `[]`. Anything else (text, a document) → queued for extraction this run (`changed`). A hash only a SKIPPED
 *     prior entry carries is never extracted by any review: `[]`. Hashes settled HERE are cleared here; a queued one is cleared by the review
 *     once its body (or the failure) is stored, so an aborted run keeps the request.
 *
 * A healthy root has no body missing and an empty recheck set: it reads nothing and costs only in-memory lookups.
 */

import type { DirectorySource, KnowledgeStore } from './model.js';
import { putNoExtract } from './extractFile.js';
import type { PlanDetail } from './plan.js';
import { isBinaryText } from './sources.js';

export interface BackfillResult {
    /** Hashes stored as an explicit `[]`. */
    recorded: number;
    /** Files queued for extraction (marked `changed` in the plan). */
    queued: string[];
}

/** The hashes the backfill will look up (for the review's ONE bulk preload): binary-by-extension manifest entries, and the re-check set. */
export function backfillCandidates(detail: PlanDetail, asked: readonly string[]): string[] {
    const skippedBinary = detail.plan.skipped.filter((s) => s.reason === 'binary').map((s) => detail.prior.get(s.path)).filter((h): h is string => !!h);
    return [...new Set([...skippedBinary, ...asked])];
}

export async function backfillNoExtract(args: { store: KnowledgeStore; source: DirectorySource; detail: PlanDetail; asked: readonly string[]; only?: readonly string[] | undefined; signal?: AbortSignal | undefined }): Promise<BackfillResult> {
    const { store, source, detail, signal } = args;
    const asked = new Set(args.asked);
    const out: BackfillResult = { recorded: 0, queued: [] };
    if (!store.getExtracted || !store.putExtracted) return out;

    const skippedBinary = detail.plan.skipped.filter((s) => s.reason === 'binary').map((s) => detail.prior.get(s.path)).filter((h): h is string => !!h);

    for (const hash of skippedBinary) {
        if (signal?.aborted) return out;
        if ((await store.getExtracted(hash)) !== undefined) continue;
        await putNoExtract(store, hash);
        out.recorded += 1;
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
        if (todo.length === 0) { done.add(hash); continue; } // a changed file is handled by this review
        // Pass 2 works only on `only`: a file it will not process is not queued here, and its hash stays asked for the run that will.
        if (args.only && !todo.some((t) => args.only!.includes(t.path))) continue;
        const f = todo[0]!;
        let binary = false;
        try {
            const doc = await source.read(f.path);
            binary = doc.content.kind === 'text' && isBinaryText(doc.content.text);
        } catch {
            continue; // unreadable says nothing about the file: asked again next time
        }
        if (binary) { await putNoExtract(store, hash); out.recorded += 1; done.add(hash); }
        else for (const t of todo) { t.changed = true; out.queued.push(t.path); } // asked for until the review has stored its body (see `settledRecheck`)
    }
    if (done.size > 0 && !signal?.aborted) await store.clearRecheck?.([...done]);
    return out;
}
