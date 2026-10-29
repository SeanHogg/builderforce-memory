/**
 * knowledge/reviewStored.ts — what a review reads back from the store: the stored model
 * knowledge indexed for re-anchoring unchanged sections without a model call, and the
 * rewrite that lets a fact outlive one of its deleted sources.
 */

import type { NodeRef, StoredFact, StoredRelation } from './model.js';
import type { JudgedBatch } from './judge.js';
import { factKeyOf } from './refs.js';

/** Stored knowledge, indexed for re-anchoring unchanged sections without a model call. */
export class StoredIndex {
    readonly facts: Map<string, StoredFact>;
    private readonly from = new Map<string, StoredRelation[]>();
    private readonly to = new Map<string, StoredRelation[]>();

    constructor(facts: StoredFact[], relations: StoredRelation[]) {
        this.facts = new Map(facts.map((f) => [f.key, f] as const));
        for (const r of relations) {
            if (r.origin !== 'model') continue;
            (this.from.get(r.from) ?? this.from.set(r.from, []).get(r.from)!).push(r);
            (this.to.get(r.to) ?? this.to.set(r.to, []).get(r.to)!).push(r);
        }
    }

    /** An unchanged section keeps its claims and mentions; they are re-anchored to the file's new hash. */
    refresh(section: NodeRef, path: string, hash: string, batch: JudgedBatch, entities: Map<string, StoredFact>): void {
        const reanchor = (f: StoredFact): StoredFact => ({
            ...f, fileAnchors: f.fileAnchors.map((a) => (a.path === path ? { ...a, contentHash: hash } : a)),
        });
        const relation = (r: StoredRelation) => batch.relation({ ...r, sourceHash: hash, staleAt: null });
        for (const r of this.to.get(section) ?? []) {
            if (r.kind !== 'about') continue;
            relation(r);
            const key = factKeyOf(r.from);
            const fact = key && (batch.facts.get(key) ?? this.facts.get(key));
            if (fact && fact.origin === 'model') batch.fact(reanchor(fact));
            for (const m of this.from.get(r.from) ?? []) if (m.kind === 'mentions') relation(m);
        }
        for (const r of this.from.get(section) ?? []) {
            if (r.kind !== 'mentions') continue;
            relation(r);
            const ent = entities.get(r.to);
            if (ent && ent.origin === 'model') {
                const next = reanchor(ent);
                entities.set(r.to, next);
                batch.fact(next);
            }
        }
    }
}

/**
 * Model facts anchored to `path` AND to another source, rewritten without that anchor — a
 * deleted file never leaves a fact stale that other sources still back. A fact anchored only
 * to `path` is untouched: it stays and shows stale (§5.2).
 */
export function factsWithoutAnchor(facts: Iterable<StoredFact>, path: string): StoredFact[] {
    return [...facts]
        .filter((f) => f.origin === 'model' && f.fileAnchors.length > 1 && f.fileAnchors.some((a) => a.path === path))
        .map((f) => ({ ...f, fileAnchors: f.fileAnchors.filter((a) => a.path !== path) }));
}
