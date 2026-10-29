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
 * plus one `{ id: 'owed:<path>', hash: '<calls>' }` per owed file, where `<calls>` is the
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
import { EXTRACTOR_VERSION, extractorManifestKey } from './sources.js';

const VERSION_ID = 'version';
const OWED_PREFIX = 'owed:';
/** What a version upgrade costs per file: the one new section a re-extraction may add (an upper bound). */
const UPGRADE_CALLS = 1;

export interface ExtractorState {
    /** The version the files were last reviewed with; undefined when never recorded. */
    version: number | undefined;
    /** Paths a previous run left unfinished under that version → model calls each still costs. */
    owed: ReadonlyMap<string, number>;
}

export async function loadExtractorState(manifest: IngestManifest, rootId: string): Promise<ExtractorState> {
    let version: number | undefined;
    const owed = new Map<string, number>();
    for (const { id, hash } of (await manifest.get(extractorManifestKey(rootId))) ?? []) {
        if (id === VERSION_ID) {
            const v = Number(hash);
            if (Number.isFinite(v)) version = v;
        } else if (id.startsWith(OWED_PREFIX)) {
            const calls = Number(hash);
            owed.set(id.slice(OWED_PREFIX.length), Number.isFinite(calls) && calls >= 0 ? calls : UPGRADE_CALLS);
        }
    }
    return { version, owed };
}

/** True when `path` is owed a re-extraction: another extractor version than today's, or left unfinished before. */
export const isOwed = (state: ExtractorState, path: string): boolean =>
    state.version !== EXTRACTOR_VERSION || state.owed.has(path);

/** Model calls an owed `path` is expected to cost: its recorded count, plus one while an upgrade is pending. */
export const owedCalls = (state: ExtractorState, path: string): number =>
    (state.version !== EXTRACTOR_VERSION ? UPGRADE_CALLS : 0) + (state.owed.get(path) ?? 0);

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
    /** Calls a file this run reached but did not finish still costs. */
    measured: ReadonlyMap<string, number>;
}): Map<string, number> {
    const out = new Map<string, number>();
    for (const f of a.changed) {
        if ((f.owed || a.prior.get(f.path) !== f.hash) && !a.finished.has(f.path)) out.set(f.path, a.measured.get(f.path) ?? owedCalls(a.state, f.path));
    }
    for (const p of a.carried) if (a.live.has(p)) out.set(p, owedCalls(a.state, p));
    return out;
}

/** Records today's version and exactly the files still owed after this run. */
export async function saveExtractorState(manifest: IngestManifest, rootId: string, owed: ReadonlyMap<string, number>): Promise<void> {
    await manifest.set(extractorManifestKey(rootId), [
        { id: VERSION_ID, hash: String(EXTRACTOR_VERSION) },
        ...[...owed].sort((a, b) => compareStr(a[0], b[0])).map(([path, calls]) => ({ id: `${OWED_PREFIX}${path}`, hash: String(calls) })),
    ]);
}
