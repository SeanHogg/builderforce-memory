/** knowledge/pack/pack.testkit.test-util.ts — a seeded generator of knowledge packs (no new dependency). */

import type { Extracted, KEdge, KNode, NodeRef, StoredFact, StoredRelation } from '../model.js';
import type { Pack } from './index.js';
import { ZERO_DIGEST } from './index.js';

/** mulberry32: a small seeded PRNG. */
export function rng(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export const ROOT = 'ws:gen';
const WORDS = ['alpha', 'béta', '日本語', 'delta', 'emoji😀', 'zeta', 'parse', 'render', 'store', 'sync'];
const KINDS = ['references', 'links', 'contains', 'mentions', 'custom-kind'];

export interface Gen { next(): number; int(n: number): number; pick<T>(xs: readonly T[]): T; chance(p: number): boolean; hex64(): string; word(): string }

export function gen(seed: number): Gen {
    const next = rng(seed);
    const int = (n: number): number => Math.floor(next() * n);
    return {
        next, int, pick: (xs) => xs[int(xs.length)]!, chance: (p) => next() < p, word: () => WORDS[int(WORDS.length)]! + int(50),
        hex64: () => Array.from({ length: 64 }, () => '0123456789abcdef'[int(16)]).join(''),
    };
}

const ref = (path: string, frag = ''): NodeRef => `src:${ROOT}::${path}${frag}` as NodeRef;

/** One file's extraction: a file node, `symbols` symbol nodes, edges to other files, keys. */
export function extractionOf(g: Gen, path: string, hash: string, targets: readonly string[], symbols: number, rich: boolean): Extracted {
    const nodes: KNode[] = [{ ref: ref(path), kind: 'file', label: path, pack: 'code', importance: Math.round(g.next() * 1e4) / 1e4, origin: 'extracted', sourceHash: hash, system: '', app: 'apps/web' }];
    for (let s = 0; s < symbols; s++) {
        const n: KNode = { ref: ref(path, `#sym${s}`), kind: 'symbol', label: `${g.word()}`, pack: 'code', importance: Math.round(g.next() * 1e4) / 1e4, origin: 'extracted', sourceHash: hash, badges: { line: 1 + s * 7 } };
        if (rich && g.chance(0.3)) { n.stale = g.chance(0.5); n.partition = 'ui'; n.badges = { line: s, kind: 'fn', n: g.next() }; }
        nodes.push(n);
    }
    const edges: KEdge[] = targets.map((t) => {
        const e: KEdge = { from: ref(path), to: ref(t), kind: g.pick(KINDS), weight: g.chance(0.5) ? 1 : g.next(), origin: 'extracted' };
        if (rich && g.chance(0.3)) { e.confidence = g.next(); e.verdict = g.pick(['allowed', 'same', 'skips', 'upward'] as const); e.cycle = g.chance(0.5); e.stale = g.chance(0.5); e.byName = g.word(); }
        return e;
    });
    return { sourceRef: ref(path), sourceHash: hash, ...(rich && g.chance(0.5) ? { extractorVersion: g.int(9) } : {}), nodes, edges, keys: [{ ref: ref(path), key: path.toLowerCase() }] };
}

export interface GenOptions { files: number; symbols?: number; edgesPerFile?: number; facts?: number; rich?: boolean; hashes?: boolean }

/** A generated pack: files, a full extraction per file, facts, relations, stale marks, tombstones, scope. */
export function genPack(seed: number, o: GenOptions): Pack {
    const g = gen(seed);
    const rich = o.rich ?? true;
    const paths = Array.from({ length: o.files }, (_, i) => `src/d${i % 37}/${g.word()}-${i}.ts`);
    const hashes = paths.map(() => g.hex64());
    const pack: Pack = { rootId: ROOT, baseGeneration: g.int(5), targetDigest: g.hex64() };
    if (o.files > 0) {
        pack.files = paths.map((path, i) => ({ path, hash: hashes[i]!, bytes: g.int(90_000) }));
        pack.extracted = paths.map((path, i) => ({
            kind: 'full' as const, hash: hashes[i]!,
            entries: [extractionOf(g, path, hashes[i]!, Array.from({ length: o.edgesPerFile ?? 3 }, () => paths[g.int(paths.length)]!), o.symbols ?? 4, rich)],
        }));
        pack.stale = paths.slice(0, Math.ceil(o.files / 4)).map((p, i) => ({ sourceRef: ref(p), hash: hashes[i]! }));
        pack.deletes = { files: [`gone/${g.word()}.ts`], facts: [`claim:${g.hex64().slice(0, 12)}`], relations: [{ from: ref('a.ts'), to: ref('b.ts'), kind: 'references' }] };
    }
    const facts: StoredFact[] = Array.from({ length: o.facts ?? Math.floor(o.files / 4) }, (_, i) => {
        const f: StoredFact = { key: `claim:${i}-${g.word()}`, content: `The ${g.word()} module ${g.word()} the ${g.word()}.`, source: 'review', importance: Math.round(g.next() * 1e3) / 1e3, origin: g.pick(['model', 'human', 'agent'] as const), fileAnchors: [] };
        if (paths.length > 0) f.fileAnchors = Array.from({ length: g.int(3) }, () => ({ path: paths[g.int(paths.length)]!, ...(g.chance(0.5) ? { line: g.int(500) } : {}), ...(g.chance(0.5) ? { contentHash: g.hex64() } : {}) }));
        if (rich && g.chance(0.3)) { f.updatedAt = '2026-11-07T10:00:00.000Z'; f.createdBy = g.chance(0.5) ? null : 'user-1'; }
        return f;
    });
    if (facts.length > 0) pack.facts = facts;
    if (paths.length > 1) {
        pack.relations = Array.from({ length: Math.floor(o.files / 3) }, (_, i): StoredRelation => {
            const r: StoredRelation = { from: ref(paths[g.int(paths.length)]!), to: `fact:claim:${i}` as NodeRef, kind: g.pick(KINDS), weight: g.chance(0.5) ? 1 : g.next(), origin: g.pick(['model', 'human', 'agent'] as const) };
            if (rich && g.chance(0.4)) { r.confidence = g.next(); r.sourceHash = hashes[g.int(hashes.length)]!; r.staleAt = g.chance(0.5) ? null : '2026-11-07T00:00:00.000Z'; }
            return r;
        });
    }
    if (rich && g.chance(0.5)) pack.scope = { rootId: ROOT, title: 'Gen', packs: ['code', 'docs'], lastDigest: g.hex64().slice(0, 32), lastReviewedAt: '2026-11-07T00:00:00.000Z' };
    if (rich && g.chance(0.5)) pack.analysis = { version: 1, nodes: ['a'], edges: [] };
    if (o.hashes) pack.hashes = [g.hex64(), '~tag:odd-id', g.hex64()];
    return pack;
}

export { ZERO_DIGEST };
