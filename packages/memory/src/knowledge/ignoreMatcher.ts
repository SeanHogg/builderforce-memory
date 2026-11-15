/**
 * knowledge/ignoreMatcher.ts — THE `.gitignore`-style matcher of a review (plan.ts asks it about every listed file).
 *
 * Semantics (unchanged since it lived in sources.ts): `#` comments, `!` negation, a trailing `/` for directories only, a
 * `/` elsewhere anchors the pattern (to its base), otherwise any depth. A path's CANDIDATES are its ancestor folders, plus
 * the path itself for a rule that is not directory-only; the LAST rule that matches any candidate decides (a negation
 * re-includes). Unlike git, a negation may re-include a file under an excluded folder.
 *
 * Cost: "the last rule that matches any candidate" is the largest index of a matching rule over the candidates, and the
 * candidates of a path are its parent folder's candidates plus one more. So the matcher keeps, per folder, the largest
 * index of a rule matching that folder or an ancestor, and the rules whose literal prefix can still match below it
 * ({@link IGNORE_DIR_CACHE_LIMIT} folders at most). It tests only those rules ABOVE the index — each folder once per walk,
 * each file only against its folder's remaining non-directory rules — and a literal prefix/suffix check skips the regex for
 * a candidate that cannot match (a nested `.gitignore`'s rules all start with `base/`). Same answers as testing every rule
 * against every candidate (regression.ignoreMatcher.test.ts).
 */

import { globToRegExp } from './partitions.js';

/**
 * Always ignored: VCS internals, installed dependencies, builderforce config, and lockfiles (`yarn.lock`, `Cargo.lock` …): generated,
 * huge, and plain text — extracted they would cost judgement calls for nothing; they were never stored (once skipped as binary).
 * A `.builderforce/ignore` rule can still re-include one (`!yarn.lock`).
 */
const DEFAULT_IGNORES = ['.git/', 'node_modules/', '.builderforce/', '.gitignore', '*.lock'];

/** How many folders' answers a matcher remembers; past it the memo starts over (answers stay exact, only work repeats). */
const IGNORE_DIR_CACHE_LIMIT = 100_000;

interface IgnoreRule {
    glob: string;
    negate: boolean;
    dirOnly: boolean;
    /** The literal text every match starts with / ends with ('' when the glob starts / ends with a wildcard). */
    prefix: string;
    suffix: string;
    re: RegExp;
}

/** Characters with a meaning in a glob (partitions.ts globToRegExp): everything else is literal. */
const SPECIAL = new Set(['*', '?', '{', '}', '\\']);

function literalPrefix(glob: string): string {
    let i = 0;
    while (i < glob.length && !SPECIAL.has(glob[i]!)) i++;
    return glob.slice(0, i);
}

function literalSuffix(glob: string): string {
    let i = glob.length;
    while (i > 0 && !SPECIAL.has(glob[i - 1]!)) i--;
    // `**/` may match nothing (`**/x` matches `x`), so the slash after a `**` is not a required literal.
    if (glob[i] === '/' && glob.slice(i - 2, i) === '**') i++;
    return glob.slice(i);
}

/**
 * What the matcher remembers of a folder: `top`, the largest index of a rule matching the folder or an ancestor (-1: none),
 * and `live`, the indices (ascending) of the rules that can still match the folder's descendants — those whose literal
 * prefix agrees with `folder/` (a nested `.gitignore`'s rules drop out everywhere outside its base).
 */
interface DirEntry {
    top: number;
    live: readonly number[];
}

const matches = (r: IgnoreRule, candidate: string): boolean =>
    candidate.startsWith(r.prefix) && candidate.endsWith(r.suffix) && r.re.test(candidate);

/** The rules of the ignore texts, in precedence order (the defaults first). */
function parseRules(texts: ReadonlyArray<string | { text: string; base: string }>): IgnoreRule[] {
    const rules: IgnoreRule[] = [];
    const lines = [
        ...DEFAULT_IGNORES.map((line) => ({ line, base: '' })),
        ...texts.flatMap((t) => {
            const { text, base } = typeof t === 'string' ? { text: t, base: '' } : t;
            return text.split(/\r?\n/).map((line) => ({ line, base }));
        }),
    ];
    for (const { line, base } of lines) {
        let p = line.replace(/\s+$/, '');
        if (!p || p.startsWith('#')) continue;
        const negate = p.startsWith('!');
        if (negate) p = p.slice(1);
        const dirOnly = p.endsWith('/');
        if (dirOnly) p = p.slice(0, -1);
        const anchored = p.includes('/');
        p = p.replace(/^\//, '');
        // A nested `.gitignore` scopes its patterns to its own folder.
        const prefix = base ? `${base}/` : '';
        if (!p) continue;
        const glob = anchored ? `${prefix}${p}` : `${prefix}**/${p}`;
        rules.push({ glob, negate, dirOnly, prefix: literalPrefix(glob), suffix: literalSuffix(glob), re: globToRegExp(glob) });
    }
    return rules;
}

/**
 * A `.gitignore`-style matcher: `#` comments, `!` negation (last match wins), a trailing
 * `/` for directories only, a `/` elsewhere anchors to the root, otherwise any depth.
 */
export function ignoreMatcher(texts: ReadonlyArray<string | { text: string; base: string }>): (path: string) => boolean {
    const rules = parseRules(texts);
    const root: DirEntry = { top: -1, live: rules.map((_, i) => i) };
    const dirs = new Map<string, DirEntry>();

    /** The largest index above `floor` among `live` of a rule (all, or only non-directory ones) matching `candidate`, else `floor`. */
    const topAbove = (candidate: string, live: readonly number[], floor: number, dirOnlyToo: boolean): number => {
        for (let k = live.length - 1; k >= 0; k--) {
            const i = live[k]!;
            if (i <= floor) break;
            const r = rules[i]!;
            if ((dirOnlyToo || !r.dirOnly) && matches(r, candidate)) return i;
        }
        return floor;
    };

    /** The entry of `dir`, from its parent's: the rules that can still match below it are those whose literal prefix agrees with `dir/`. */
    const entryOf = (dir: string, parent: DirEntry): DirEntry => {
        const top = topAbove(dir, parent.live, parent.top, true);
        const below = `${dir}/`;
        const live = parent.live.filter((i) => { const p = rules[i]!.prefix; return p.startsWith(below) || below.startsWith(p); });
        return { top, live: live.length === parent.live.length ? parent.live : live };
    };

    return (path) => {
        const segments = path.split('/');
        let entry = root;
        let dir = '';
        for (let i = 0; i < segments.length - 1; i++) {
            dir = i === 0 ? segments[0]! : `${dir}/${segments[i]!}`;
            let next = dirs.get(dir);
            if (!next) {
                next = entryOf(dir, entry);
                if (dirs.size >= IGNORE_DIR_CACHE_LIMIT) dirs.clear();
                dirs.set(dir, next);
            }
            entry = next;
        }
        const top = topAbove(path, entry.live, entry.top, false);
        return top >= 0 && !rules[top]!.negate;
    };
}
