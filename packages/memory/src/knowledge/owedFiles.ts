/**
 * knowledge/owedFiles.ts — the extractor entry of a root's manifest (PRD 38 §4.3): the
 * extractor version the files were last reviewed with, and the files still OWED a
 * re-extraction under it. Planning reads it, every run writes it — aborted or not.
 *
 * A file is owed when the stored version is not {@link EXTRACTOR_VERSION} (every file, once),
 * or when an earlier run left it unfinished (a read error, an abort before it, the budget, a
 * failed judge call). Owed is independent of the file's content: a content-changed file left
 * unfinished is recorded too — under any version — so its debt survives the bytes reverting
 * to the manifest hash. A path the plan no longer lists (ignored, claimed by no pack, a run
 * that aborted before retiring it) stays owed while the files manifest still holds it; once
 * retired it drops out.
 *
 * Stored under {@link extractorManifestKey} as fingerprints: `{ id: 'version', hash: '<version>' }`
 * plus `{ id: 'documents', hash: '<pdf,docx|->' }` (the document types the packs claimed, see {@link withDocuments})
 * and one `{ id: 'owed:<path>', hash: '<calls>' }` per owed file, where `<calls>` is the
 * decimal number of model calls the file is still expected to cost (the plan's estimate):
 *   • a file the run reached but did not finish: the sections the budget or a failed call left
 *     unjudged; for a content-changed file, every section it had to judge plus what it was
 *     already owed (a revert to the manifest bytes re-judges them all);
 *   • a file not reached (an abort) or unreadable: what it was already owed at, {@link owedCalls}
 *     — the recorded count, plus the one new section a re-extraction may add while the stored
 *     version is not today's. A read error learns nothing about the file, so a permanently
 *     unreadable file is the only file re-read and its known debt is neither erased nor invented.
 * An old `[{ id: 'version', … }]` value is `{ version, owed: none }`; a missing or empty value
 * is an upgrade — every file owed one call.
 */

import type { IngestManifest } from '../ingest/types.js';
import { compareStr } from './refs.js';
import { EXTRACTOR_VERSION, extOf, extractorManifestKey, isBinaryPath, isDocumentPath } from './sources.js';

const VERSION_ID = 'version';
const DOCUMENTS_ID = 'documents';
/** No document type claimed — also what an entry with no `documents` record means, so a root that never had a host parser stores nothing extra. */
const NO_DOCUMENTS = '-';
const OWED_PREFIX = 'owed:';
const FAIL_PREFIX = 'fail:';
/** What a version upgrade costs per file: the one new section a re-extraction may add (an upper bound). */
const UPGRADE_CALLS = 1;

export interface ExtractorState {
    /** The version the files were last reviewed with; undefined when never recorded. */
    version: number | undefined;
    /** Paths a previous run left unfinished under that version → model calls each still costs. */
    owed: ReadonlyMap<string, number>;
    /** Consecutive transient failures of a path at one content hash (stored beside its owed entry); a success or a new hash resets it. Optional: a caller that builds a state by hand need not know it. */
    failures?: ReadonlyMap<string, { hash: string; count: number }>;
    /** The document extensions the packs claimed when the files were last reviewed (see {@link claimedDocumentExts}); undefined when never recorded. */
    documents?: string;
    /** The ones the packs claim now: set by the plan (and carried to *Continue*), saved by the run. */
    documentsNow?: string;
    /** Document extensions whose asset-ness changed between `documents` and `documentsNow`: their files are owed a re-processing. */
    flipped?: ReadonlySet<string>;
    /** Claimed documents over the plan's size cap: planned as assets (never read), so they owe no calls. Set by the plan, carried to *Continue*. */
    oversized?: ReadonlySet<string>;
}

/**
 * Whether a file is an asset depends on the packs (a pdf is an asset until a host parser claims it), but the files manifest records only
 * content hashes. So the extractor entry also records ONE string, the claimed document extensions (`documents:<pdf,docx|->`); the document
 * types whose claim changed since — in either direction — are `flipped`, and every file of those types is owed (its hash may not have moved).
 * No record reads as "none claimed" (and is not written), so a root reviewed before any host parser existed sees the document types a parser
 * claims later as flipped, and a root from before this existed is owed everything by its version anyway.
 *
 * KNOWN LIMIT: two hosts with DIFFERENT pack sets sharing one manifest thrash — each switch flips every pdf/office file and re-extracts it
 * (no model calls: judged sections re-anchor, but the work is real). Every Builderforce host is wired with the same documents pack, so the
 * sets agree; this is deliberately not solved with per-pack-set cache entries.
 */
export function withDocuments(state: ExtractorState, now: string | undefined): ExtractorState {
    if (now === undefined) return state;
    const list = (s: string) => new Set(s === NO_DOCUMENTS ? [] : s.split(','));
    const flipped = new Set<string>();
    if (state.documents !== undefined && state.documents !== now) {
        const was = list(state.documents), is = list(now);
        for (const e of was) if (!is.has(e)) flipped.add(e);
        for (const e of is) if (!was.has(e)) flipped.add(e);
    }
    return { ...state, documentsNow: now, flipped };
}

export async function loadExtractorState(manifest: IngestManifest, rootId: string): Promise<ExtractorState> {
    let version: number | undefined;
    let documents: string | undefined;
    const owed = new Map<string, number>();
    const failures = new Map<string, { hash: string; count: number }>();
    for (const { id, hash } of (await manifest.get(extractorManifestKey(rootId))) ?? []) {
        if (id === VERSION_ID) {
            const v = Number(hash);
            if (Number.isFinite(v)) version = v;
        } else if (id === DOCUMENTS_ID) {
            documents = hash;
        } else if (id.startsWith(FAIL_PREFIX)) {
            const at = hash.indexOf(':');
            const count = Number(hash.slice(0, at));
            if (at > 0 && Number.isInteger(count) && count > 0) failures.set(id.slice(FAIL_PREFIX.length), { hash: hash.slice(at + 1), count });
        } else if (id.startsWith(OWED_PREFIX)) {
            const calls = Number(hash);
            owed.set(id.slice(OWED_PREFIX.length), Number.isFinite(calls) && calls >= 0 ? calls : UPGRADE_CALLS);
        }
    }
    return { version, owed, failures, documents: documents ?? NO_DOCUMENTS };
}

/** The file's document extension when its asset-ness flipped (a host parser arrived or left since the last run). */
const isFlipped = (state: ExtractorState, path: string): boolean => !!state.flipped?.size && state.flipped.has(extOf(path));

/** True when `path` is owed a re-extraction: another extractor version than today's, left unfinished before, or its asset-ness flipped. */
export const isOwed = (state: ExtractorState, path: string): boolean =>
    state.version !== EXTRACTOR_VERSION || state.owed.has(path) || isFlipped(state, path);

/** Whether `path` is an asset under the packs the plan was made with: binary, and not a document some parser claims. */
const isAssetNow = (state: ExtractorState, path: string): boolean => {
    if (!isBinaryPath(path)) return false;
    if (!isDocumentPath(path)) return true;
    if (state.oversized?.has(path)) return true; // claimed, but over the size cap: an asset like an unclaimed one
    const claimed = state.documentsNow;
    return claimed === undefined || claimed === NO_DOCUMENTS || !claimed.split(',').includes(extOf(path));
};

/**
 * Model calls an owed `path` is expected to cost: its recorded count, plus one while an upgrade is pending or its asset-ness flipped.
 * An asset costs none (it is never read or judged); a file that flipped TO a document does.
 */
export const owedCalls = (state: ExtractorState, path: string): number =>
    isAssetNow(state, path) ? 0 : (state.version !== EXTRACTOR_VERSION || isFlipped(state, path) ? UPGRADE_CALLS : 0) + (state.owed.get(path) ?? 0);

/**
 * Paths in the files manifest that the plan does not list yet are owed: ignored, claimed by
 * no pack, or missing from an older plan handed to *Continue*. A removed path is among them
 * until the run retires it; the run saves only those still in the manifest.
 */
export function carriedOwed(state: ExtractorState, prior: ReadonlyMap<string, string>, planned: Iterable<string>): string[] {
    const listed = new Set(planned);
    return [...prior.keys()].filter((p) => !listed.has(p) && isOwed(state, p)).sort(compareStr);
}

/**
 * The owed record a run leaves: its unfinished files that are owed or content-changed (at the
 * cost it measured, else the one it was owed at), and the carried paths the files manifest
 * still holds (`live`). A forced run's unchanged, not-owed files are neither, so it never
 * widens the set.
 */
export function owedAfterRun(a: {
    state: ExtractorState;
    changed: ReadonlyArray<{ path: string; hash: string; owed: boolean }>;
    /** The files manifest as the plan read it: a hash that differs from the file's is a content change. */
    prior: ReadonlyMap<string, string>;
    finished: ReadonlySet<string>;
    carried: readonly string[];
    live: ReadonlyMap<string, string>;
    /** Files whose read or extraction failed for a reason that says nothing final about them (same bytes as the manifest): owed even though their content did not change. */
    unreadable?: ReadonlySet<string>;
    /** Calls a file this run reached but did not finish still costs. */
    measured: ReadonlyMap<string, number>;
}): Map<string, number> {
    const out = new Map<string, number>();
    for (const f of a.changed) {
        if ((f.owed || a.prior.get(f.path) !== f.hash || a.unreadable?.has(f.path)) && !a.finished.has(f.path)) out.set(f.path, a.measured.get(f.path) ?? owedCalls(a.state, f.path));
    }
    for (const p of a.carried) if (a.live.has(p)) out.set(p, owedCalls(a.state, p));
    return out;
}

/** Records today's version and exactly the files still owed after this run. */
export async function saveExtractorState(
    manifest: IngestManifest, rootId: string, owed: ReadonlyMap<string, number>, failures: ReadonlyMap<string, { hash: string; count: number }> = new Map(),
    documents?: string,
): Promise<void> {
    await manifest.set(extractorManifestKey(rootId), [
        { id: VERSION_ID, hash: String(EXTRACTOR_VERSION) },
        ...(documents !== undefined && documents !== NO_DOCUMENTS ? [{ id: DOCUMENTS_ID, hash: documents }] : []),
        ...[...owed].sort((a, b) => compareStr(a[0], b[0])).map(([path, calls]) => ({ id: `${OWED_PREFIX}${path}`, hash: String(calls) })),
        ...[...failures].filter(([path]) => owed.has(path)).sort((a, b) => compareStr(a[0], b[0])).map(([path, f]) => ({ id: `${FAIL_PREFIX}${path}`, hash: `${f.count}:${f.hash}` })),
    ]);
}
