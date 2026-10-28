/**
 * knowledge/plan.ts — `planReview`: the confirm chip's numbers (PRD 38 §4.3).
 *
 * A walk plus a manifest diff by hash, nothing more: a file whose hash is unchanged is
 * never read. The only reads are the ignore files (`.gitignore`, `.builderforce/ignore`)
 * when the listing has them. Binary files (by extension), oversized files and files no
 * pack accepts are skipped with a reason.
 */

import type { IngestManifest } from '../ingest/types.js';
import type { DirectorySource, KnowledgePack, ReviewPlan } from './model.js';
import {
    composePacks, DEFAULT_MAX_FILE_BYTES, filesManifestKey, IGNORE_FILES, ignoreMatcher, isBinaryPath,
    manifestMap, stubSource,
} from './sources.js';
import { compareStr, sortBy } from './refs.js';

/** ≈ bytes per judged section, for the estimate (the plan does not read changed files). */
export const AVG_SECTION_BYTES = 1500;
/** ≈ one compare call per this many judged sections when contradictions are asked for. */
const SECTIONS_PER_COMPARE = 10;

export interface PlanOptions {
    /** Per-file size cap in bytes. Default 2 MB. */
    maxFileBytes?: number;
}

export interface PlannedFile {
    path: string;
    hash: string;
    size: number;
    pack: string;
    changed: boolean;
}

export interface PlanDetail {
    plan: ReviewPlan;
    /** Every included file (changed and unchanged), sorted by path. */
    files: PlannedFile[];
    /** The manifest's prior `path → hash`. */
    prior: Map<string, string>;
}

export async function planReview(
    source: DirectorySource,
    manifest: IngestManifest,
    packs: KnowledgePack[],
    opts: PlanOptions = {},
): Promise<ReviewPlan> {
    return (await planDetail(source, manifest, packs, opts)).plan;
}

export async function planDetail(
    source: DirectorySource,
    manifest: IngestManifest,
    packs: readonly KnowledgePack[],
    opts: PlanOptions = {},
): Promise<PlanDetail> {
    const maxBytes = opts.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;
    const listed: Array<{ path: string; hash: string; size: number }> = [];
    for await (const entry of source.list()) listed.push({ ...entry, path: entry.path.replace(/\\/g, '/').replace(/^\.?\//, '') });
    const entries = sortBy(listed, (e) => e.path);

    const ignoreTexts: string[] = [];
    for (const name of IGNORE_FILES) {
        if (!entries.some((e) => e.path === name)) continue;
        const doc = await source.read(name);
        if (doc.content.kind === 'text') ignoreTexts.push(doc.content.text);
    }
    const ignored = ignoreMatcher(ignoreTexts);
    const composed = composePacks(packs);
    const prior = await manifestMap(manifest, filesManifestKey(source.rootId));
    const packById = new Map(packs.map((p) => [p.id, p] as const));

    const files: PlannedFile[] = [];
    const skipped: ReviewPlan['skipped'] = [];
    const byPack: Record<string, number> = {};
    let estimatedSections = 0;
    let compareSections = 0;
    for (const e of entries) {
        if (ignored(e.path)) { skipped.push({ path: e.path, reason: 'ignored' }); continue; }
        if (isBinaryPath(e.path)) { skipped.push({ path: e.path, reason: 'binary' }); continue; }
        if (e.size > maxBytes) { skipped.push({ path: e.path, reason: 'too-large' }); continue; }
        const hit = composed.resolve(stubSource(e.path));
        if (!hit) { skipped.push({ path: e.path, reason: 'no-pack' }); continue; }
        const changed = prior.get(e.path) !== e.hash;
        files.push({ path: e.path, hash: e.hash, size: e.size, pack: hit.pack.id, changed });
        byPack[hit.pack.id] = (byPack[hit.pack.id] ?? 0) + 1;
        const spec = packById.get(hit.pack.id)?.judgement;
        if (changed && spec && (spec.claims || spec.entities)) {
            const sections = Math.max(1, Math.ceil(e.size / AVG_SECTION_BYTES));
            estimatedSections += sections;
            if (spec.contradictions) compareSections += sections;
        }
    }

    const present = new Set(entries.map((e) => e.path));
    const removed = [...prior.keys()].filter((p) => !present.has(p)).sort(compareStr);
    const changed = files.filter((f) => f.changed).map((f) => f.path);
    const sortedByPack = Object.fromEntries(Object.keys(byPack).sort(compareStr).map((k) => [k, byPack[k]!]));
    return {
        plan: {
            rootId: source.rootId,
            files: files.length,
            byPack: sortedByPack,
            unchanged: files.length - changed.length,
            changed,
            removed,
            skipped,
            estimatedModelCalls: estimatedSections + Math.ceil(compareSections / SECTIONS_PER_COMPARE),
        },
        files,
        prior,
    };
}
