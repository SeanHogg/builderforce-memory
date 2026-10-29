/**
 * knowledge/plan.ts — `planReview`: the confirm chip's numbers (PRD 38 §4.3).
 *
 * A walk plus a manifest diff by hash, nothing more: a file whose hash is unchanged is
 * never read. The only reads are the ignore files (`.builderforce/ignore` and every
 * `.gitignore`, a nested one scoped to its folder) when the listing has them. Binary files
 * (by extension — pdf/office only when a host parser claims their media type), oversized
 * files and files no pack accepts are skipped with a reason.
 */

import type { IngestManifest } from '../ingest/types.js';
import type { DirectorySource, KnowledgePack, ReviewPlan } from './model.js';
import {
    composePacks, DEFAULT_MAX_FILE_BYTES, EXTRACTOR_VERSION, filesManifestKey, ignoreMatcher, ignoreSourcesOf, isBinaryPath,
    isDocumentPath, manifestMap, resolveDocument, storedExtractorVersion, stubSource,
} from './sources.js';
import { compareStr, sortBy } from './refs.js';

/** ≈ bytes per judged section, for the estimate (the plan does not read changed files). */
export const AVG_SECTION_BYTES = 1500;
/** ≈ one compare call per this many judged sections when contradictions are asked for. */
const SECTIONS_PER_COMPARE = 10;

export interface PlanOptions {
    /** Per-file size cap in bytes. Default 2 MB. */
    maxFileBytes?: number;
    /**
     * *Re-review all*: every file is `changed`, whatever its hash — but `removed` still comes
     * from the real prior manifest, and the estimate still counts only files whose content
     * changed (sections whose text was judged before are re-anchored, not judged again).
     * A forced run cut short is forced again, not continued: *Continue* re-derives what is
     * left from content hashes.
     */
    force?: boolean;
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

    const ignoreTexts: Array<{ text: string; base: string }> = [];
    for (const { path, base } of ignoreSourcesOf(entries.map((e) => e.path))) {
        const doc = await source.read(path);
        if (doc.content.kind === 'text') ignoreTexts.push({ text: doc.content.text, base });
    }
    const ignored = ignoreMatcher(ignoreTexts);
    const composed = composePacks(packs);
    const prior = await manifestMap(manifest, filesManifestKey(source.rootId));
    // Another extractor version than the one that reviewed these files: all are re-extracted
    // once, but `prior` stays the real manifest (removals) and the estimate counts content changes only.
    const everything = opts.force === true || (await storedExtractorVersion(manifest, source.rootId)) !== EXTRACTOR_VERSION;
    const packById = new Map(packs.map((p) => [p.id, p] as const));

    const files: PlannedFile[] = [];
    const skipped: ReviewPlan['skipped'] = [];
    const byPack: Record<string, number> = {};
    let estimatedSections = 0;
    let compareSections = 0;
    for (const e of entries) {
        if (ignored(e.path)) { skipped.push({ path: e.path, reason: 'ignored' }); continue; }
        const document = isDocumentPath(e.path) ? resolveDocument(composed, e.path) : undefined;
        if (isBinaryPath(e.path) && !document) { skipped.push({ path: e.path, reason: 'binary' }); continue; }
        if (e.size > maxBytes) { skipped.push({ path: e.path, reason: 'too-large' }); continue; }
        const hit = document ?? composed.resolve(stubSource(e.path));
        if (!hit) { skipped.push({ path: e.path, reason: 'no-pack' }); continue; }
        const contentChanged = prior.get(e.path) !== e.hash;
        const changed = everything || contentChanged;
        files.push({ path: e.path, hash: e.hash, size: e.size, pack: hit.pack.id, changed });
        byPack[hit.pack.id] = (byPack[hit.pack.id] ?? 0) + 1;
        const spec = packById.get(hit.pack.id)?.judgement;
        if (contentChanged && spec && (spec.claims || spec.entities)) {
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

/**
 * *Continue*: the confirmed plan without a re-walk. Only `changed` is re-read — against the
 * manifest as the last run left it — so files finished since are not reviewed again. (While
 * the stored extractor version is not today's, every file still counts as changed.)
 */
export async function refreshPlan(detail: PlanDetail, manifest: IngestManifest): Promise<PlanDetail> {
    const prior = await manifestMap(manifest, filesManifestKey(detail.plan.rootId));
    // The version entry is written only by a run that finished, so an upgrade cut short stays "everything".
    const everything = (await storedExtractorVersion(manifest, detail.plan.rootId)) !== EXTRACTOR_VERSION;
    const files = detail.files.map((f) => ({ ...f, changed: everything || prior.get(f.path) !== f.hash }));
    const changed = files.filter((f) => f.changed).map((f) => f.path);
    const removed = detail.plan.removed.filter((p) => prior.has(p));
    return {
        plan: { ...detail.plan, unchanged: files.length - changed.length, changed, removed },
        files,
        prior,
    };
}
