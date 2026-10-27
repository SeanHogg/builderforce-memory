/**
 * Code-map RECALL — which `codemap:<area>` memories a run should be handed.
 *
 * {@link consolidate} writes one map per code area. A map only saves a search when the
 * next run on that area is GIVEN it, so the read side lives here beside the write side,
 * and every host (editor, cloud, on-prem) picks maps with the same rules:
 *
 *   - {@link areaOfPath} — the area a named file or directory belongs to. A file at the
 *     repository root belongs to {@link CONSOLIDATED_ROOT_AREA}, the same area
 *     {@link consolidate} files a root-only run under, so that map is reachable.
 *   - {@link areasOfPathsIn} — the areas of the repo-relative paths a text names.
 *   - {@link areasNamedIn} — the stored areas a request names by WORD ("fix the run
 *     store" names `brain-embedded/src/runStore`), for a request that names no path.
 *   - {@link consolidatedFactsFor} — the maps whose area overlaps the areas a run is
 *     about, most specific first.
 *
 * Pure: no I/O.
 */

/** Key prefix every consolidated code map carries. */
export const CONSOLIDATED_KEY_PREFIX = 'codemap:';
/** The area of files at the repository root (`codemap:(root)`). */
export const CONSOLIDATED_ROOT_AREA = '(root)';
/** Maps a run is handed, at most. */
export const MAX_RECALLED_MAPS = 6;

/** Too generic to say which area a request is about: nearly every area has one. */
const GENERIC_SEGMENTS: ReadonlySet<string> = new Set([
  'src', 'lib', 'libs', 'app', 'apps', 'package', 'packages', 'component', 'components', 'util', 'utils',
  'test', 'tests', 'index', 'main', 'core', 'common', 'shared', 'source', 'internal', 'client', 'clients',
  'server', 'docs', 'scripts', 'public', 'assets',
]);
/** A word shorter than this matches too much to name an area. */
const MIN_AREA_WORD = 4;
/** Areas one request may name by word, at most. */
const MAX_WORD_AREAS = 8;

/**
 * The area a repo-relative path belongs to: a file's directory, a directory itself (a
 * path with no extension), and {@link CONSOLIDATED_ROOT_AREA} for a file at the root.
 * `''` for an empty path.
 */
export function areaOfPath(p: string): string {
  const norm = p.trim().replace(/\\/g, '/').replace(/^(\.\/)+/, '').replace(/\/+$/, '');
  if (!norm) return '';
  if (!/\.[A-Za-z][A-Za-z0-9]{0,7}$/.test(norm)) return norm;
  const i = norm.lastIndexOf('/');
  return i > 0 ? norm.slice(0, i) : CONSOLIDATED_ROOT_AREA;
}

/** The area a `codemap:` key is for, or undefined for any other key. */
export function consolidatedArea(key: string): string | undefined {
  return key.startsWith(CONSOLIDATED_KEY_PREFIX) ? key.slice(CONSOLIDATED_KEY_PREFIX.length) : undefined;
}

/** True when `area` and `dir` are the same directory or one contains the other. */
function overlaps(area: string, dir: string): boolean {
  return area === dir || dir.startsWith(`${area}/`) || area.startsWith(`${dir}/`);
}

/** The `codemap:` facts whose area overlaps a named area, most specific first, capped. */
export function consolidatedFactsFor<F extends { key: string }>(
  facts: readonly F[],
  areas: readonly string[],
  max: number = MAX_RECALLED_MAPS,
): F[] {
  if (areas.length === 0) return [];
  const scored: Array<{ fact: F; depth: number }> = [];
  const seen = new Set<string>();
  for (const fact of facts) {
    const area = consolidatedArea(fact.key);
    if (area === undefined || seen.has(fact.key)) continue;
    if (!areas.some((d) => overlaps(area, d))) continue;
    seen.add(fact.key);
    scored.push({ fact, depth: area.split('/').length });
  }
  return scored.sort((a, b) => b.depth - a.depth).slice(0, max).map((s) => s.fact);
}

/** A repo-relative path token: has a `/` or a file extension, and is not absolute. */
const PATH_TOKEN = /[A-Za-z0-9_.@()\-\\/]+/g;
/** Paths one text may name, at most. */
const MAX_PATH_AREAS = 24;

/**
 * The areas of the repo-relative paths a text names (`api/src/run.ts` → `api/src`,
 * `package.json` → the root area). Absolute paths are somebody else's machine and are
 * skipped, as are URLs.
 */
export function areasOfPathsIn(text: string): string[] {
  const out: string[] = [];
  for (const raw of text.replace(/[a-z][a-z0-9+.-]*:\/\/\S+/gi, ' ').match(PATH_TOKEN) ?? []) {
    // A drive letter (`C:`) is not part of a token, so `C:/x/y.ts` arrives as `/x/y.ts`
    // and is skipped as absolute below.
    const token = raw.replace(/^[(]+|[.)]+$/g, '');
    if (!token || /^[\\/]/.test(token) || token.split(/[\\/]/).includes('..')) continue;
    // A bare file name needs a stem of two or more characters, so `e.g` is not a file.
    const isPath = /[\\/]/.test(token) || /^[^.]{2,}.*\.[A-Za-z][A-Za-z0-9]{0,7}$/.test(token);
    if (!isPath) continue;
    const area = areaOfPath(token);
    if (area && !out.includes(area)) out.push(area);
    if (out.length >= MAX_PATH_AREAS) break;
  }
  return out;
}

/** `runStore`, `run-store`, `run_store` and `run store` all read `runstore`. */
function squash(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]/g, '');
}

/** The request's words and adjacent pairs, squashed — `context pack` also reads `contextpack`. */
function requestTerms(text: string): Set<string> {
  const words = text
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
  const out = new Set<string>();
  for (let i = 0; i < words.length; i++) {
    out.add(words[i]!);
    if (words[i]!.endsWith('s') && words[i]!.length > MIN_AREA_WORD) out.add(words[i]!.slice(0, -1));
    if (i + 1 < words.length) out.add(words[i]! + words[i + 1]!);
  }
  return out;
}

/**
 * The stored areas a request names by word: an area matches when one of its directory
 * names (generic ones like `src` and short ones aside) is a word, or two adjacent words,
 * of the request. Deepest areas first, capped.
 */
export function areasNamedIn(text: string, areas: readonly string[]): string[] {
  const terms = requestTerms(text);
  if (terms.size === 0) return [];
  const hits = [...new Set(areas)].filter((area) =>
    area.split('/').some((seg) => {
      const s = squash(seg);
      return s.length >= MIN_AREA_WORD && !GENERIC_SEGMENTS.has(s) && terms.has(s);
    }));
  return hits.sort((a, b) => b.split('/').length - a.split('/').length).slice(0, MAX_WORD_AREAS);
}
