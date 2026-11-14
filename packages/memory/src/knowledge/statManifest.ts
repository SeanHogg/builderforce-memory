/**
 * knowledge/statManifest.ts — the STAT PREFILTER (PRD 41 §B2 steps 1–2): a per-root manifest of
 * `(path, size, mtimeNs, fileId, hash, recordedAtMs)` so a warm plan reuses a stored content hash
 * without reading the file, and only stats — never content — decide it.
 *
 * Rules implemented here, straight from the PRD:
 *   0. Content identity is unchanged by this module: the stored `hash` is the host's content hash
 *      (EOL-normalized for text, raw for binary — see contentHash.ts); this manifest only caches it.
 *   1. Stat tuple match → reuse the stored hash. Stats are BigInt (`bigint: true` from fs.stat):
 *      Windows file ids and ns mtimes lose precision as JS numbers, so {@link StatRecord} carries
 *      them as bigint and the manifest persists them as decimal strings.
 *   2. Racy-file rule (git's index): a file whose mtime is within {@link RACY_WINDOW_MS} of the
 *      manifest's write time, or of `now` at plan time, is re-hashed whatever its stat says.
 *   4. Directory Merkle hashes stay the identity of a subtree (merkle.ts) — this manifest never
 *      enters a digest, a cache key or the plan output (PRD 41 §B8.3: mtime is only a hint).
 *
 * Determinism (§B8): every output here is sorted by path; the only clock use is the injected
 * `now` (default `Date.now`) stamping `recordedAtMs`, which is persisted metadata, never part of
 * any digest. Hosts that need a deterministic plan inject `now`.
 *
 * The plan itself consumes hashes from `DirectorySource.list()`; this module is the helper a
 * host's listing calls per file ({@link createStatPrefilter}) plus the manifest persistence the
 * plan updates after a successful run ({@link saveStatManifest}).
 */

import type { IngestManifest } from '../ingest/types.js';
import { canonicalJson, compareStr, normalizePath } from './refs.js';
import { manifestMap, saveManifestMap } from './sources.js';

/** git's racy-window: an mtime within this many ms of the index write is re-checked next plan. */
export const RACY_WINDOW_MS = 2000;

/** One file's stat tuple, read with `bigint: true`. `fileId` is `ino` on POSIX, the file id on Windows; null when the host cannot say. */
export interface StatRecord {
    size: number;
    mtimeNs: bigint;
    fileId: bigint | null;
}

/** A stat-manifest row: the stat tuple the hash was cached under, the hash, and when the manifest was written (ms). */
export interface StatManifestEntry extends StatRecord {
    hash: string;
    recordedAtMs: number;
}

/** The manifest key under the ingest manifest: one stat manifest per root. */
export const statManifestKey = (rootId: string): string => `stats:${rootId}`;

/** Stat-tuple equality: same size, same mtime to the nanosecond, same file id (a null fileId matches only a null fileId). */
export function statMatches(a: StatRecord, b: StatRecord): boolean {
    return a.size === b.size && a.mtimeNs === b.mtimeNs && a.fileId === b.fileId;
}

/**
 * The racy-file rule (§B2 step 2): true when the file may have been written in the same tick as
 * the manifest — its mtime is within {@link RACY_WINDOW_MS} of the manifest's `recordedAtMs`, or
 * of `nowMs` (a file touched just now whose mtime we are about to record). A racy file is
 * re-hashed on the next plan whatever its stat says.
 */
export function isRacy(stat: StatRecord, recordedAtMs: number, nowMs: number): boolean {
    const mtimeMs = Number(stat.mtimeNs / 1_000_000n);
    return Math.abs(mtimeMs - recordedAtMs) < RACY_WINDOW_MS || Math.abs(mtimeMs - nowMs) < RACY_WINDOW_MS;
}

export interface PrefilterDecision {
    /** Paths whose stored hash is reused (stat matched, not racy), sorted. */
    reused: readonly string[];
    /** Paths that must be hashed (new, stat moved, or racy), sorted. */
    hash: readonly string[];
}

/**
 * The ONE prefilter decision, pure and order-independent: a path reuses its stored hash only when
 * its prior entry exists, the stat tuple matches, and the racy rule does not fire. Everything else
 * is hashed. `nowMs` is the plan's clock.
 */
export function prefilterStats(
    listed: ReadonlyArray<{ path: string; stat: StatRecord }>,
    prior: ReadonlyMap<string, StatManifestEntry>,
    nowMs: number,
): PrefilterDecision {
    const reused: string[] = [];
    const hash: string[] = [];
    for (const { path, stat } of listed) {
        const p = normalizePath(path);
        const entry = prior.get(p);
        if (entry && statMatches(entry, stat) && !isRacy(stat, entry.recordedAtMs, nowMs)) reused.push(p);
        else hash.push(p);
    }
    reused.sort(compareStr);
    hash.sort(compareStr);
    return { reused, hash };
}

/**
 * A per-file hash provider for a host's `DirectorySource.list()`: returns the stored hash when
 * {@link prefilterStats} reuses the path, else `hashPath(path)`. `hashPath` reads the file and
 * hashes it (EOL-normalized, per the host's content-hash rule) — the only path that reads bytes.
 * `now` defaults to `Date.now`; inject it for deterministic plans.
 */
export function createStatPrefilter(
    prior: ReadonlyMap<string, StatManifestEntry>,
    hashPath: (listedPath: string) => Promise<string>,
    now: () => number = () => Date.now(),
): (entry: { path: string; stat: StatRecord }) => Promise<{ path: string; hash: string }> {
    return async ({ path, stat }) => {
        const normalized = normalizePath(path);
        const entry = prior.get(normalized);
        if (entry && statMatches(entry, stat) && !isRacy(stat, entry.recordedAtMs, now())) {
            return { path: normalized, hash: entry.hash };
        }
        return { path: normalized, hash: await hashPath(path) };
    };
}

/* ── persistence (the ingest manifest, one canonical JSON blob per root) ─────────────────────────── */

interface StatRow {
    size: number;
    mtimeNs: string;
    fileId: string | null;
    hash: string;
    recordedAtMs: number;
}

const encode = (entries: ReadonlyMap<string, StatManifestEntry>): string => canonicalJson(
    [...entries.entries()].sort((a, b) => compareStr(a[0], b[0])).map(([path, e]): [string, StatRow] => [path, {
        size: e.size,
        mtimeNs: e.mtimeNs.toString(),
        fileId: e.fileId === null ? null : e.fileId.toString(),
        hash: e.hash,
        recordedAtMs: e.recordedAtMs,
    }]),
);

const decode = (json: string): Map<string, StatManifestEntry> => {
    const rows = JSON.parse(json) as Array<[string, StatRow]>;
    const out = new Map<string, StatManifestEntry>();
    for (const [path, r] of rows ?? []) {
        out.set(path, {
            size: r.size,
            mtimeNs: BigInt(r.mtimeNs),
            fileId: r.fileId === null ? null : BigInt(r.fileId),
            hash: r.hash,
            recordedAtMs: r.recordedAtMs,
        });
    }
    return out;
};

/** The stored stat manifest for `rootId` (empty when none — the first run hashes everything, the existing fallback). */
export async function loadStatManifest(manifest: IngestManifest, rootId: string): Promise<Map<string, StatManifestEntry>> {
    const map = await manifestMap(manifest, statManifestKey(rootId));
    const raw = map.get('entries');
    if (raw === undefined) return new Map();
    try {
        return decode(raw);
    } catch {
        return new Map(); // a corrupt stat cache is a perf fallback only: re-hash, never fail the plan
    }
}

/**
 * Persists the stat manifest for `rootId`: every listed file's hash and the stat tuple it was
 * cached under, stamped with `recordedAtMs` (the manifest write time the racy rule compares to).
 */
export async function saveStatManifest(
    manifest: IngestManifest,
    rootId: string,
    files: ReadonlyArray<{ path: string; hash: string; stat: StatRecord }>,
    recordedAtMs: number,
): Promise<void> {
    const entries = new Map<string, StatManifestEntry>();
    for (const f of files) entries.set(normalizePath(f.path), { ...f.stat, hash: f.hash, recordedAtMs });
    await saveManifestMap(manifest, statManifestKey(rootId), new Map([['entries', encode(entries)]]));
}
