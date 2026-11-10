/**
 * knowledge/merkle.ts — directory Merkle hashes over a root's file listing (PRD 41 §B2 step 4).
 *
 * A directory's hash is sha256 over the canonical JSON of its entries `{ name, kind, hash }`
 * (`kind` 'file' | 'dir'), sorted by name. An equal directory hash means an equal subtree, so
 * {@link changedPaths} skips it and two trees compare in O(changed).
 *
 * Paths go through the ONE `normalizePath` (refs.ts: POSIX, no leading `./` or `/`, NFC) and sort bytewise (§B8.1). The
 * root directory's key is `""`; a nested one's is its path (`a`, `a/b`). Any input order gives
 * the same tree.
 */

import { sha256Hex } from '../retrieval/knowledge-index/sha256.js';
import { canonicalJson, isRootFilePath, normalizePath } from './refs.js';

export type MerkleKind = 'file' | 'dir';

export interface MerkleEntry {
    name: string;
    kind: MerkleKind;
    /** A file's content hash, or a directory's Merkle hash. */
    hash: string;
}

export interface MerkleTree {
    /** The root directory's hash (`dirs.get('')`). */
    root: string;
    /** Directory path → its Merkle hash; the root is `""`. */
    dirs: ReadonlyMap<string, string>;
    /** Directory path → its entries, sorted by name. */
    entries: ReadonlyMap<string, ReadonlyArray<MerkleEntry>>;
}

/** Bytewise UTF-8 order (code-point order), which UTF-16 `<` is not for astral characters. */
export function compareBytewise(a: string, b: string): number {
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
        const x = a.codePointAt(i)!, y = b.codePointAt(i)!;
        if (x !== y) return x < y ? -1 : 1;
        if (x > 0xffff) i++;
    }
    return a.length - b.length;
}

const dirOf = (path: string): string => {
    const at = path.lastIndexOf('/');
    return at < 0 ? '' : path.slice(0, at);
};
const nameOf = (path: string): string => path.slice(path.lastIndexOf('/') + 1);
const joinPath = (dir: string, name: string): string => (dir === '' ? name : `${dir}/${name}`);

/** Why {@link canonicalListing} left a listed path out. */
export type ListingSkipReason = 'invalid-path' | 'duplicate-path';

/**
 * A host listing made safe for {@link merkleTree} — which then never throws — without depending on input order:
 *   • every path goes through the ONE {@link normalizePath} (POSIX, NFC); the source's own spelling is kept as `listedPath`
 *     when it differs, so a read still opens the file the source listed;
 *   • a path that is empty or has an empty, `.` or `..` segment is skipped as `invalid-path`;
 *   • of paths that collide once normalized (an NFC/NFD pair, a path listed twice) the FIRST by (normalized path, listed path),
 *     bytewise, is kept and the rest are skipped as `duplicate-path`; so is a path below a kept FILE (`a` and `a/b`).
 * Skipped entries carry the listed spelling. `kept` is sorted bytewise by normalized path.
 */
export function canonicalListing<T extends { path: string }>(entries: readonly T[]): {
    kept: Array<T & { listedPath?: string }>;
    skipped: Array<{ path: string; reason: ListingSkipReason }>;
} {
    const skipped: Array<{ path: string; reason: ListingSkipReason }> = [];
    const candidates: Array<{ entry: T; path: string }> = [];
    for (const entry of entries) {
        const path = normalizePath(entry.path);
        if (isRootFilePath(path)) candidates.push({ entry, path });
        else skipped.push({ path: entry.path, reason: 'invalid-path' });
    }
    candidates.sort((a, b) => compareBytewise(a.path, b.path) || compareBytewise(a.entry.path, b.entry.path));
    const files = new Set<string>();
    const kept: Array<T & { listedPath?: string }> = [];
    for (const { entry, path } of candidates) {
        let clash = files.has(path);
        for (let dir = dirOf(path); !clash && dir !== ''; dir = dirOf(dir)) clash = files.has(dir);
        if (clash) { skipped.push({ path: entry.path, reason: 'duplicate-path' }); continue; }
        files.add(path);
        kept.push({ ...entry, path, ...(entry.path !== path ? { listedPath: entry.path } : {}) });
    }
    return { kept, skipped };
}

/**
 * The Merkle tree of a file listing. Throws on an invalid path, a path listed twice with different hashes, or a path
 * that is both a file and a directory — each would make the root depend on input order. {@link canonicalListing}
 * first makes a host listing safe.
 */
export function merkleTree(files: ReadonlyArray<{ path: string; hash: string }>): MerkleTree {
    const children = new Map<string, Map<string, { kind: MerkleKind; hash?: string }>>([['', new Map()]]);
    const childrenOf = (dir: string) => {
        let c = children.get(dir);
        if (!c) children.set(dir, (c = new Map()));
        return c;
    };
    for (const f of files) {
        const path = normalizePath(f.path);
        if (!isRootFilePath(path)) throw new Error(`merkleTree: not a file path: ${JSON.stringify(f.path)}`);
        const own = childrenOf(dirOf(path)).get(nameOf(path));
        if (own && (own.kind !== 'file' || own.hash !== f.hash)) throw new Error(`merkleTree: ${path} listed twice or as a file and a directory`);
        childrenOf(dirOf(path)).set(nameOf(path), { kind: 'file', hash: f.hash });
        for (let dir = dirOf(path); dir !== ''; dir = dirOf(dir)) {
            const slot = childrenOf(dirOf(dir));
            const was = slot.get(nameOf(dir));
            if (was?.kind === 'file') throw new Error(`merkleTree: ${dir} is both a file and a directory`);
            if (was) break; // its ancestors are registered already
            slot.set(nameOf(dir), { kind: 'dir' });
        }
    }

    const dirs = new Map<string, string>();
    const entries = new Map<string, MerkleEntry[]>();
    // Deepest first, so every child directory is hashed before its parent.
    const order = [...children.keys()].sort((a, b) => depth(b) - depth(a) || compareBytewise(a, b));
    for (const dir of order) {
        const list = [...children.get(dir)!].map(([name, c]): MerkleEntry => ({
            name, kind: c.kind, hash: c.kind === 'file' ? c.hash! : dirs.get(joinPath(dir, name))!,
        })).sort((a, b) => compareBytewise(a.name, b.name));
        entries.set(dir, list);
        dirs.set(dir, sha256Hex(canonicalJson(list)));
    }
    return { root: dirs.get('')!, dirs, entries };
}

const depth = (dir: string): number => (dir === '' ? 0 : dir.split('/').length);

/** Every file path under `dir` in `tree`, appended to `out`. */
function filesUnder(tree: MerkleTree, dir: string, out: string[]): void {
    for (const e of tree.entries.get(dir) ?? []) {
        const path = joinPath(dir, e.name);
        if (e.kind === 'file') out.push(path);
        else filesUnder(tree, path, out);
    }
}

/**
 * The file paths changed, added or removed between two trees, sorted bytewise. Walks top-down
 * and never descends into a directory whose hash is equal on both sides.
 */
export function changedPaths(before: MerkleTree, after: MerkleTree): string[] {
    const out: string[] = [];
    const walk = (dir: string): void => {
        if (before.dirs.get(dir) === after.dirs.get(dir)) return;
        const was = new Map((before.entries.get(dir) ?? []).map((e) => [e.name, e] as const));
        const is = new Map((after.entries.get(dir) ?? []).map((e) => [e.name, e] as const));
        for (const name of new Set([...was.keys(), ...is.keys()])) {
            const a = was.get(name), b = is.get(name);
            const path = joinPath(dir, name);
            if (a?.kind === 'dir' && b?.kind === 'dir') { if (a.hash !== b.hash) walk(path); continue; }
            if (a && b && a.kind === 'file' && b.kind === 'file') { if (a.hash !== b.hash) out.push(path); continue; }
            // Added, removed, or a file replaced by a directory (or back): every file on either side.
            for (const [side, e] of [[before, a], [after, b]] as const) {
                if (e?.kind === 'file') out.push(path);
                else if (e) filesUnder(side, path, out);
            }
        }
    };
    walk('');
    return [...new Set(out)].sort(compareBytewise);
}
