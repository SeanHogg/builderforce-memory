/**
 * knowledge/judge.ts — the judgement pass of a review: the only step that costs model
 * calls (PRD 38 §4.3).
 *
 *   • extract: one call per changed section of a judged kind → claims (`claim:<hash>`) and
 *     entities (`ent:<pack>:<name>`) as `model` facts anchored with the file's content
 *     hash, plus `about` / `mentions` relations;
 *   • compare: near-duplicate claim pairs → `contradicts` / `agrees` relations.
 *
 * Every response is schema-validated; invalid items are dropped and counted. The budget
 * (calls and a deadline) is shared across the whole review.
 */

import { hashText } from '../ingest/hash.js';
import type {
    JudgedClaim, JudgedEntity, JudgementPort, JudgementResult, NodeRef, StoredFact, StoredRelation,
} from './model.js';
import type { SimilarPair } from '../retrieval/knowledge-index/types.js';
import { entityKey, normalizeKey } from './resolve.js';
import { compareStr, factRef } from './refs.js';

export const DEFAULT_REVIEW_MODEL_CALLS = 200;
export const REVIEW_FACT_SOURCE = 'knowledge-review';
const MAX_TERM = 200;
const MAX_NAME = 120;
const MAX_QUOTE_CHARS = 300;
const MAX_QUOTE_LINES = 3;

export interface JudgeBudget {
    modelCalls: number;
    deadline: number;
    used: number;
    now(): number;
}

export const budgetLeft = (b: JudgeBudget): boolean => b.used < b.modelCalls && b.now() < b.deadline;

/** Facts and relations a review will write, keyed for upsert. */
export class JudgedBatch {
    readonly facts = new Map<string, StoredFact>();
    readonly relations = new Map<string, StoredRelation>();
    judged = 0;
    unjudged = 0;
    invalid = 0;

    fact(f: StoredFact): void { this.facts.set(f.key, f); }

    relation(r: StoredRelation): void { this.relations.set(`${r.from}\u0000${r.to}\u0000${r.kind}`, r); }

    take(): { facts: StoredFact[]; relations: StoredRelation[] } {
        const out = {
            facts: [...this.facts.values()].sort((a, b) => compareStr(a.key, b.key)),
            relations: [...this.relations.values()].sort((a, b) =>
                compareStr(a.from, b.from) || compareStr(a.to, b.to) || compareStr(a.kind, b.kind)),
        };
        this.facts.clear();
        this.relations.clear();
        return out;
    }
}

const str = (v: unknown, max: number, allowEmpty = false): string | undefined => {
    if (typeof v !== 'string') return undefined;
    const t = v.trim();
    return (t || allowEmpty) && t.length <= max ? t : undefined;
};

function validClaim(c: unknown): JudgedClaim | undefined {
    if (!c || typeof c !== 'object') return undefined;
    const o = c as Record<string, unknown>;
    const subject = str(o['subject'], MAX_TERM), predicate = str(o['predicate'], MAX_TERM), object = str(o['object'], MAX_TERM);
    if (!subject || !predicate || !object || typeof o['quote'] !== 'string') return undefined;
    const quote = (o['quote'] as string).split(/\r?\n/).slice(0, MAX_QUOTE_LINES).join('\n').trim().slice(0, MAX_QUOTE_CHARS);
    return { subject, predicate, object, quote };
}

function validEntity(e: unknown): JudgedEntity | undefined {
    if (!e || typeof e !== 'object') return undefined;
    const o = e as Record<string, unknown>;
    const name = str(o['name'], MAX_NAME), type = str(o['type'], MAX_NAME, true);
    return name && normalizeKey(name) && type !== undefined ? { name, type: type || 'thing' } : undefined;
}

/** The valid claims and entities a response holds for `ref`; everything else counted invalid. */
export function validateExtract(result: unknown, ref: NodeRef): { claims: JudgedClaim[]; entities: JudgedEntity[]; invalid: number } {
    const out = { claims: [] as JudgedClaim[], entities: [] as JudgedEntity[], invalid: 0 };
    if (!result || typeof result !== 'object') { out.invalid = 1; return out; }
    const r = result as Partial<JudgementResult>;
    out.invalid += typeof r.invalid === 'number' && r.invalid > 0 ? Math.floor(r.invalid) : 0;
    if (!Array.isArray(r.extract)) { out.invalid += 1; return out; }
    for (const item of r.extract as unknown[]) {
        const it = item as { ref?: unknown; claims?: unknown; entities?: unknown } | null;
        if (!it || it.ref !== ref || !Array.isArray(it.claims) || !Array.isArray(it.entities)) { out.invalid += 1; continue; }
        for (const c of it.claims) { const v = validClaim(c); if (v) out.claims.push(v); else out.invalid += 1; }
        for (const e of it.entities) { const v = validEntity(e); if (v) out.entities.push(v); else out.invalid += 1; }
    }
    return out;
}

export function validateCompare(result: unknown, a: NodeRef, b: NodeRef): { relation?: 'contradicts' | 'agrees' | 'unrelated'; confidence: number; invalid: number } {
    if (!result || typeof result !== 'object') return { confidence: 0, invalid: 1 };
    const r = result as Partial<JudgementResult>;
    let invalid = typeof r.invalid === 'number' && r.invalid > 0 ? Math.floor(r.invalid) : 0;
    for (const item of Array.isArray(r.compare) ? r.compare : []) {
        const ok = item && ((item.a === a && item.b === b) || (item.a === b && item.b === a))
            && ['contradicts', 'agrees', 'unrelated'].includes(item.relation)
            && typeof item.confidence === 'number' && item.confidence >= 0 && item.confidence <= 1;
        if (ok) return { relation: item.relation, confidence: item.confidence, invalid };
        invalid += 1;
    }
    return { confidence: 0, invalid: Math.max(invalid, 1) };
}

/** One model call, or a counted invalid when the port throws (an abort is re-thrown). */
async function call(port: JudgementPort, req: Parameters<JudgementPort['judge']>[0], b: JudgeBudget): Promise<unknown> {
    b.used += 1;
    try {
        return await port.judge(req);
    } catch (err) {
        if (req.signal?.aborted) throw err;
        return undefined;
    }
}

export interface SectionToJudge { ref: NodeRef; text: string }

export interface ExtractArgs {
    port: JudgementPort;
    pack: string;
    path: string;
    fileHash: string;
    sections: readonly SectionToJudge[];
    budget: JudgeBudget;
    batch: JudgedBatch;
    /** Entity facts by key: stored ones plus this review's, merged across files. */
    entities: Map<string, StoredFact>;
    signal?: AbortSignal;
    /** Called for each section actually judged (for the judged-sections manifest). */
    onJudged(ref: NodeRef): void;
}

/** Judges each section within the budget; the rest are counted `unjudged`. */
export async function judgeSections(a: ExtractArgs): Promise<Array<{ ref: NodeRef; text: string }>> {
    const claims: Array<{ ref: NodeRef; text: string }> = [];
    const anchor = { path: a.path, contentHash: a.fileHash };
    for (const section of a.sections) {
        // Honoured per section, not only when the port throws: the caller drops this file whole.
        if (a.signal?.aborted) throw a.signal.reason ?? new Error('aborted');
        if (!budgetLeft(a.budget)) { a.batch.unjudged += 1; continue; }
        const result = await call(a.port, {
            kind: 'extract', items: [{ ref: section.ref, text: section.text, sourceHash: a.fileHash }],
            ...(a.signal ? { signal: a.signal } : {}),
        }, a.budget);
        const v = validateExtract(result, section.ref);
        a.batch.invalid += v.invalid;
        a.batch.judged += 1;
        a.onJudged(section.ref);
        const rel = (from: NodeRef, to: NodeRef, kind: string) =>
            a.batch.relation({ from, to, kind, weight: 1, origin: 'model', sourceHash: a.fileHash });

        const entityRefs = new Map<string, NodeRef>();
        for (const e of v.entities) {
            const key = entityKey(a.pack, e.name);
            const prior = a.entities.get(key);
            const anchors = [...(prior?.fileAnchors ?? []).filter((x) => x.path !== a.path), anchor]
                .sort((x, y) => compareStr(x.path, y.path));
            const fact: StoredFact = {
                key, content: prior?.content ?? `${e.name} (${e.type})`, source: REVIEW_FACT_SOURCE,
                importance: prior?.importance ?? 0.5, origin: prior?.origin ?? 'model', fileAnchors: anchors,
            };
            a.entities.set(key, fact);
            a.batch.fact(fact);
            entityRefs.set(normalizeKey(e.name), key);
            rel(section.ref, key, 'mentions');
        }
        for (const c of v.claims) {
            const key = `claim:${hashText([section.ref, c.subject, c.predicate, c.object].join('\u0000'))}`;
            const text = `${c.subject} ${c.predicate} ${c.object}`;
            a.batch.fact({
                key, content: c.quote ? `${text} — “${c.quote}”` : text, source: REVIEW_FACT_SOURCE,
                importance: 0.5, origin: 'model', fileAnchors: [anchor],
            });
            const ref = factRef(key);
            rel(ref, section.ref, 'about');
            for (const term of [c.subject, c.object]) {
                const ent = entityRefs.get(normalizeKey(term));
                if (ent) rel(ref, ent, 'mentions');
            }
            claims.push({ ref, text });
        }
    }
    return claims;
}

export interface CompareArgs {
    port: JudgementPort;
    pairs: readonly SimilarPair[];
    /** Claim ref → text and the content hash it was judged from. */
    claims: ReadonlyMap<string, { text: string; hash: string }>;
    budget: JudgeBudget;
    batch: JudgedBatch;
    signal?: AbortSignal;
}

/** Near-duplicate claim pairs, most similar first, compared within the budget. */
export async function compareClaims(a: CompareArgs): Promise<void> {
    const pairs = [...a.pairs]
        .filter((p) => p.a !== p.b && a.claims.has(p.a) && a.claims.has(p.b))
        .map((p) => (compareStr(p.a, p.b) <= 0 ? p : { a: p.b, b: p.a, similarity: p.similarity }))
        .sort((x, y) => y.similarity - x.similarity || compareStr(x.a, y.a) || compareStr(x.b, y.b));
    const seen = new Set<string>();
    for (const p of pairs) {
        const id = `${p.a}\u0000${p.b}`;
        if (seen.has(id)) continue;
        seen.add(id);
        if (a.signal?.aborted) return;
        if (!budgetLeft(a.budget)) { a.batch.unjudged += 1; continue; }
        const ca = a.claims.get(p.a)!, cb = a.claims.get(p.b)!;
        const A = p.a as NodeRef, B = p.b as NodeRef;
        const result = await call(a.port, {
            kind: 'compare',
            items: [{ ref: A, text: ca.text, sourceHash: ca.hash }, { ref: B, text: cb.text, sourceHash: cb.hash }],
            ...(a.signal ? { signal: a.signal } : {}),
        }, a.budget);
        const v = validateCompare(result, A, B);
        a.batch.invalid += v.invalid;
        a.batch.judged += 1;
        if (v.relation === 'contradicts' || v.relation === 'agrees') {
            a.batch.relation({
                from: A, to: B, kind: v.relation, weight: 1, origin: 'model', confidence: v.confidence, sourceHash: ca.hash,
            });
        }
    }
}
