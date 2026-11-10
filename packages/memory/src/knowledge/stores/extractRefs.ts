/**
 * knowledge/stores/extractRefs.ts — which extract bodies a {@link LocalSnapshotStore} still needs. The extract cache is keyed by
 * content hash, so every edit of a file adds a body; without this, the superseded bodies were kept in memory, the snapshot, the
 * extract log and the host's rows forever.
 *
 * A source's identity is its `sourceRef` (plus the `~tag` of a root-level output's hash, as `fileLeafKey` cuts it, so two
 * analyzers writing the same ref do not supersede each other). Each source has ONE current body and keeps ONE previous body
 * (a grace: switching a branch back hits the cache). A body is kept while it is the current or the previous body of some
 * source; when it is neither, it is evicted.
 *
 * A put CLAIMS only the entries it adds or changes against the body already stored under that hash. `putExtract` merges a new
 * entry into the cached body of a shared hash, so the body also carries entries of files that have since moved to another hash;
 * re-putting it must never make those files current at it again (that evicted their real current body).
 *
 * A body with no entries (`[]`: a binary file, nothing to extract) or entries without a `sourceRef` names no source and is never
 * evicted by this: it is a few bytes. Pure bookkeeping, no I/O: the store owns the maps, the log and the dirty marks.
 */

import type { Extracted } from '../model.js';
import { canonicalJson } from '../refs.js';

/** The identity of a source within a body: `sourceRef`, or `~tag|sourceRef` for a root-level output's hash. */
export function sourceIdentity(hash: string, sourceRef: string): string {
    if (!hash.startsWith('~')) return sourceRef;
    const colon = hash.indexOf(':');
    return `${colon > 0 ? hash.slice(0, colon) : hash}|${sourceRef}`;
}

/** The source refs of a body's entries (an entry without one names no source). */
export function sourceRefsOf(entries: readonly Extracted[]): string[] {
    const out: string[] = [];
    for (const e of entries) if (typeof e?.sourceRef === 'string' && e.sourceRef) out.push(e.sourceRef);
    return out;
}

/** Adds `v` to the set at `k` of `m`. */
function addTo(m: Map<string, Set<string>>, k: string, v: string): void {
    const s = m.get(k);
    if (s) s.add(v); else m.set(k, new Set([v]));
}

/** Removes `v` from the set at `k` of `m`, dropping an emptied set. */
function removeFrom(m: Map<string, Set<string>>, k: string, v: string): void {
    const s = m.get(k);
    if (!s) return;
    s.delete(v);
    if (s.size === 0) m.delete(k);
}

export class ExtractRefs {
    /** Body hash → the sources it is current for. */
    private readonly live = new Map<string, Set<string>>();
    /** Source → its current bodies (one after a put; several only after `seed`s whose order is unknown). */
    private readonly owners = new Map<string, Set<string>>();
    /** Source → its previous body (the grace). */
    private readonly previous = new Map<string, string>();
    /** Body hash → the sources it is the previous body of. */
    private readonly previousOf = new Map<string, Set<string>>();

    /**
     * `entries` are now stored under `hash`, replacing `prior` (what was stored under it before, if anything). The sources of the
     * entries that are new or changed against `prior` become current at `hash`. Returns the bodies that are now neither current
     * nor previous for any source: the caller evicts them.
     */
    put(hash: string, entries: readonly Extracted[], prior?: readonly Extracted[]): string[] {
        const before = new Map((prior ?? []).filter((e) => typeof e?.sourceRef === 'string').map((e) => [e.sourceRef, canonicalJson(e)] as const));
        const present = new Set(sourceRefsOf(entries).map((r) => sourceIdentity(hash, r)));
        const claims = new Set<string>();
        for (const e of entries) {
            if (typeof e?.sourceRef !== 'string' || !e.sourceRef) continue;
            if (before.get(e.sourceRef) !== canonicalJson(e)) claims.add(sourceIdentity(hash, e.sourceRef));
        }
        const touched = new Set<string>();
        // A source whose entry left this body is no longer current at it.
        for (const id of [...(this.live.get(hash) ?? [])]) {
            if (present.has(id)) continue;
            removeFrom(this.live, hash, id);
            removeFrom(this.owners, id, hash);
        }
        touched.add(hash);
        for (const id of claims) {
            for (const other of [...(this.owners.get(id) ?? [])]) {
                if (other === hash) continue;
                removeFrom(this.live, other, id);
                this.setPrevious(id, other, touched);
                touched.add(other);
            }
            this.owners.set(id, new Set([hash]));
            addTo(this.live, hash, id);
        }
        return [...touched].filter((h) => h !== hash && !this.kept(h));
    }

    /**
     * Records a body whose place in time is unknown (a snapshot, a reload, a lazily read body, a row the host lists) WITHOUT
     * superseding anything: it shares its sources with the bodies already there until the next `put` of one of them.
     */
    seed(hash: string, sourceRefs: readonly string[]): void {
        for (const r of sourceRefs) {
            const id = sourceIdentity(hash, r);
            addTo(this.live, hash, id);
            addTo(this.owners, id, hash);
        }
    }

    /**
     * What `hash` is to its sources, as source refs: the ones it is current for and the ones it is the previous body of. What a
     * compacted extract log writes beside the body, so its replay restores exactly this instead of re-deriving it.
     */
    stateOf(hash: string): { current: string[]; previous: string[] } {
        const refOf = (id: string): string => (hash.startsWith('~') ? id.slice(id.indexOf('|') + 1) : id);
        return { current: [...(this.live.get(hash) ?? [])].map(refOf), previous: [...(this.previousOf.get(hash) ?? [])].map(refOf) };
    }

    /** Puts back what {@link stateOf} recorded, superseding nothing (a compacted log is a consistent state, not a sequence of puts). */
    restore(hash: string, state: { current: readonly string[]; previous: readonly string[] }): void {
        this.seed(hash, state.current);
        for (const r of state.previous) {
            const id = sourceIdentity(hash, r);
            const old = this.previous.get(id);
            if (old !== undefined) removeFrom(this.previousOf, old, id);
            this.previous.set(id, hash);
            addTo(this.previousOf, hash, id);
        }
    }

    /** True when the hash is tracked (current or previous for some source). */
    has(hash: string): boolean { return this.kept(hash); }

    /** Bodies tracked (for tests and diagnostics). */
    get size(): number { return new Set([...this.live.keys(), ...this.previousOf.keys()]).size; }

    /** A copy: the same order knowledge, independent from here on. */
    clone(): ExtractRefs {
        const c = new ExtractRefs();
        for (const [k, s] of this.live) c.live.set(k, new Set(s));
        for (const [k, s] of this.owners) c.owners.set(k, new Set(s));
        for (const [k, v] of this.previous) c.previous.set(k, v);
        for (const [k, s] of this.previousOf) c.previousOf.set(k, new Set(s));
        return c;
    }

    /** Forgets a body that left the store by another path (a reload that no longer lists it). */
    forget(hash: string): void {
        for (const id of [...(this.live.get(hash) ?? [])]) removeFrom(this.owners, id, hash);
        this.live.delete(hash);
        for (const id of [...(this.previousOf.get(hash) ?? [])]) this.previous.delete(id);
        this.previousOf.delete(hash);
    }

    private kept(hash: string): boolean { return this.live.has(hash) || this.previousOf.has(hash); }

    /** `hash` becomes the previous body of `id`; the body it replaces there is touched (it may now be unreferenced). */
    private setPrevious(id: string, hash: string, touched: Set<string>): void {
        const old = this.previous.get(id);
        if (old === hash) return;
        if (old !== undefined) { removeFrom(this.previousOf, old, id); touched.add(old); }
        this.previous.set(id, hash);
        addTo(this.previousOf, hash, id);
    }
}
