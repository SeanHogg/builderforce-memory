/**
 * Consolidation — what survives a run, as ONE long-term memory per code AREA.
 *
 * A run that spends ninety turns locating the code behind a feature leaves with that
 * knowledge in its transcript and nowhere else, so the next run asked about the same
 * area repeats the whole search. {@link consolidate} turns the run's working memory
 * into a map — the files that mattered, changed first, then reads ranked by how much
 * the run leaned on them — under a key per area (the directory most of them share).
 * Deterministic: built from what the run actually opened, it cannot invent a path.
 *
 * {@link mergeConsolidated} is the WRITE-side rule a keyed store applies, so a small run
 * ADDS to an area's map instead of overwriting the twelve files an earlier one found,
 * and a file this run deleted leaves the map. {@link consolidatedFiles} and
 * {@link withoutConsolidatedFiles} read the stored format back, so a store can anchor a
 * map to its files and drop one that changed since without hiding the rest.
 */

import { CONSOLIDATED_KEY_PREFIX, CONSOLIDATED_ROOT_AREA } from './codemapAreas.js';
import { rankByCount } from './rank.js';
import type { ConsolidateOptions, ConsolidatedMemory, WorkingMemorySnapshot } from './types.js';

/**
 * Fewer distinct files than this — changed, deleted and read together — is a lookup,
 * not a map worth keeping. It holds for EVERY run, a run that changed files included: a
 * one-file fix keyed to its own directory would otherwise write (or, once merged, keep
 * re-dating) a map that tells the next run nothing a search would not.
 */
export const MIN_CONSOLIDATED_FILES = 3;
const MAX_FILES = 12;
const MIN_AREA_SEGMENTS = 2;
const MAX_AREA_SEGMENTS = 4;
const MAX_TASK_CHARS = 200;
const MAX_KEY_CHARS = 255;
/** Dependencies, build output and the knowledge tree itself — never "where the code is". */
const NOISE_SEGMENTS: ReadonlySet<string> = new Set(['node_modules', 'dist', '.next', '.turbo', 'coverage', '.git', '.builderforce']);
const NOISE_FILES: ReadonlySet<string> = new Set(['pnpm-lock.yaml', 'package-lock.json', 'yarn.lock', 'bun.lockb']);

const DRIVE = /^[a-z]:\//i;
/** MSYS / Git-Bash spelling of a drive path: `/c/code/x` ≡ `C:/code/x`. */
const MSYS_DRIVE = /^\/([a-z])\//i;

function canonical(raw: string): string {
  let p = raw.trim().replace(/\\/g, '/');
  const msys = MSYS_DRIVE.exec(p);
  if (msys) p = `${msys[1]}:/${p.slice(3)}`;
  // Collapse `a//b` and `a/./b`; leading `./` too.
  p = p.replace(/\/{2,}/g, '/').replace(/(^|\/)(\.\/)+/g, '$1');
  return p;
}

/**
 * A repo-relative path, or undefined when the path is not a repo file worth mapping:
 * outside the root (or absolute with no root to resolve it against), escaping with
 * `..`, dependencies or build output, or a lockfile. Drive paths compare
 * case-insensitively, as the filesystem does.
 */
function repoPath(raw: string, root?: string): string | undefined {
  let p = canonical(raw);
  if (p.startsWith('/') || DRIVE.test(p)) {
    const base = root ? canonical(root).replace(/\/+$/, '') : '';
    if (!base || base === '/') return undefined;
    const under = DRIVE.test(base) ? p.toLowerCase().startsWith(`${base.toLowerCase()}/`) : p.startsWith(`${base}/`);
    if (!under) return undefined;
    p = p.slice(base.length + 1);
  }
  const segs = p.split('/');
  if (!p || segs.includes('..')) return undefined;
  if (segs.some((s) => NOISE_SEGMENTS.has(s)) || NOISE_FILES.has(segs[segs.length - 1]!)) return undefined;
  return p;
}

function dirSegments(file: string): string[] {
  const parts = file.split('/');
  parts.pop();
  return parts;
}

/**
 * The area a set of files belongs to: the DEEPEST directory (to a bounded depth, so
 * keys stay stable from run to run) holding a majority of them, else the depth-2
 * directory holding the most. Root-level files (DONE.md, package.json — touched by
 * many unrelated runs) never decide it.
 */
function areaOf(files: readonly string[]): string {
  const placed = files.map(dirSegments).filter((d) => d.length > 0);
  if (placed.length === 0) return CONSOLIDATED_ROOT_AREA;
  const prefixes = (depth: number): [string, number][] =>
    rankByCount(placed.filter((d) => d.length >= depth).map((d) => d.slice(0, depth).join('/')));
  for (let depth = MAX_AREA_SEGMENTS; depth >= MIN_AREA_SEGMENTS; depth--) {
    const [top] = prefixes(depth);
    if (top && top[1] * 2 > placed.length) return top[0];
  }
  return prefixes(MIN_AREA_SEGMENTS)[0]?.[0] ?? prefixes(1)[0]![0];
}

/** The key for an area — stable across runs, so consolidations of one area meet in one memory. */
export function consolidatedKey(area: string): string {
  return `${CONSOLIDATED_KEY_PREFIX}${area}`.slice(0, MAX_KEY_CHARS);
}

/**
 * The long-term memory for one finished run, or `null` when the run touched too little
 * to be worth keeping. Files the run deleted are carried as `(deleted)` lines, which
 * {@link mergeConsolidated} applies as removals and never stores.
 */
export function consolidate(snapshot: WorkingMemorySnapshot, opts: ConsolidateOptions = {}): ConsolidatedMemory | null {
  const keep = (raw: string): string | undefined => repoPath(raw, opts.root);
  const kept = (list: readonly string[]): string[] => list.map(keep).filter((p): p is string => !!p);
  const deleted = [...new Set(kept(snapshot.deleted))];
  const deletedSet = new Set(deleted);
  const changed = [...new Set(kept([...snapshot.created, ...snapshot.edited]))].filter((p) => !deletedSet.has(p));
  const changedSet = new Set(changed);
  const reads = rankByCount(kept(snapshot.read)).filter(([p]) => !changedSet.has(p) && !deletedSet.has(p));
  const readCounts = new Map(reads);
  const ranked = [...changed, ...reads.map(([p]) => p)];
  if (ranked.length + deleted.length < (opts.minFiles ?? MIN_CONSOLIDATED_FILES)) return null;
  if (ranked.length === 0 && deleted.length === 0) return null;

  const shown = ranked.slice(0, MAX_FILES);
  const area = areaOf(ranked.length > 0 ? ranked : deleted);
  const at = (opts.at ?? new Date()).toISOString().slice(0, 10);
  const asked = (snapshot.task ?? '').replace(/\s+/g, ' ').trim().slice(0, MAX_TASK_CHARS);
  const lines = [
    `Code map for \`${area}\` (mapped ${at}${asked ? ` by a run asked: "${asked}"` : ''}).`,
    'Open these first instead of searching:',
    ...shown.map((p) => {
      if (changedSet.has(p)) return `- ${p} (changed)`;
      const n = readCounts.get(p) ?? 1;
      return `- ${p}${n > 1 ? ` (read ${n}×)` : ''}`;
    }),
    ...deleted.map((p) => `- ${p} (deleted)`),
  ];
  if (ranked.length > shown.length) lines.push(`(+${ranked.length - shown.length} more file(s) touched)`);
  return { key: consolidatedKey(area), content: lines.join('\n') };
}

/** `- <path> (<note>)` — the path is everything before the trailing parenthesised note. */
const ENTRY = /^- (.+?)(?: \(([^()]*)\))?$/;
const MORE = /^\(\+\d+ more file\(s\) touched\)$/;

interface Entry { path: string; note: string | undefined }

function entriesOf(content: string): Entry[] {
  const out: Entry[] = [];
  for (const line of content.split('\n')) {
    const m = ENTRY.exec(line);
    if (m) out.push({ path: m[1]!, note: m[2] });
  }
  return out;
}

function render(e: Entry): string {
  return `- ${e.path}${e.note ? ` (${e.note})` : ''}`;
}

/**
 * The content to STORE when a consolidated map lands on a key that already holds one:
 * the new run's entries first, then the earlier map's entries it did not touch (an
 * earlier run's `changed` is restated as `changed earlier`, so this run is not credited
 * with edits it never made), minus every file this run deleted, capped. Any other key,
 * or no prior, stores `next` without its `(deleted)` lines.
 *
 * The STORE decides whether `prior` is still live — pass it only when the stored memory
 * has not expired or been superseded, or a retired map comes back to life.
 */
export function mergeConsolidated(key: string, prior: string | null | undefined, next: string): string {
  if (!key.startsWith(CONSOLIDATED_KEY_PREFIX)) return next;
  const nextLines = next.split('\n');
  const header = nextLines.filter((l) => !ENTRY.test(l) && !MORE.test(l));
  const fresh = entriesOf(next);
  const removed = new Set(fresh.filter((e) => e.note === 'deleted').map((e) => e.path));
  const seen = new Set<string>(removed);
  const merged: string[] = [];
  for (const e of fresh) {
    if (seen.has(e.path)) continue;
    seen.add(e.path);
    merged.push(render(e));
  }
  for (const e of prior?.trim() ? entriesOf(prior) : []) {
    if (seen.has(e.path) || e.note === 'deleted') continue;
    seen.add(e.path);
    merged.push(render(e.note === 'changed' ? { ...e, note: 'changed earlier' } : e));
  }
  const shown = merged.slice(0, MAX_FILES);
  const more = merged.length - shown.length;
  return [...header, ...shown, ...(more > 0 ? [`(+${more} more file(s) touched)`] : [])].join('\n');
}

/** The files a stored map lists, in order — never a `(deleted)` line. */
export function consolidatedFiles(content: string): string[] {
  return entriesOf(content).filter((e) => e.note !== 'deleted').map((e) => e.path);
}

/** Paths compare as the store and the filesystem do: slashes normalised, case ignored. */
function samePathKey(p: string): string {
  return canonical(p).toLowerCase();
}

/**
 * The map with `paths` taken out — for a store whose files changed since the map was
 * written: the changed file's line goes, the rest of the map stays. `null` when no file
 * is left to list, so the caller drops the map rather than serving a bare header.
 */
export function withoutConsolidatedFiles(content: string, paths: Iterable<string>): string | null {
  const drop = new Set([...paths].map(samePathKey));
  let left = 0;
  const lines = content.split('\n').filter((line) => {
    const m = ENTRY.exec(line);
    if (!m) return true;
    if (drop.has(samePathKey(m[1]!))) return false;
    left++;
    return true;
  });
  return left > 0 ? lines.join('\n') : null;
}
