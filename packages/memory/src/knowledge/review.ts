/**
 * knowledge/review.ts — `reviewDirectory`: point it at a folder and it learns what the
 * folder says (PRD 38 §4.3).
 *
 *   plan (walk + manifest diff) → per changed file: read · parse (the ONE ParserRegistry,
 *   composed from the packs) · extract (cached by content hash) · judge (changed sections
 *   of judged kinds, within budget) · write · markStale → analyzers → compare near-duplicate
 *   claims → manifest + scope digest.
 *
 * Store judgement, compute perception: only `model` facts and relations are written —
 * `extracted` output goes to the extracted cache, never to memory ({@link assertStoredBatch}).
 * Abort is honoured between files and inside judgement; a file is written whole or not at
 * all. A file whose sections were cut by the budget stays "changed" in the manifest, so
 * *Continue* (the confirmed plan passed back as `args.plan`, no re-walk) judges only the
 * sections not judged yet.
 */

import { hashText } from '../ingest/hash.js';
import { sha256Hex } from '../retrieval/knowledge-index/sha256.js';
import type {
    ExtractContext, Extracted, KnowledgeParser, KnowledgeStore, NodeRef, ReviewArgs, ReviewReport, ScopeRecord,
    StoredFact, StoredRelation,
} from './model.js';
import type { SourceDocument, ParsedSection } from '../ingest/types.js';
import { SECTION_INDEX_BADGE } from '../ingest/docsExtract.js';
import { planDetail, refreshPlan } from './plan.js';
import {
    asPathSource, composePacks, filesManifestKey, isBinaryText, judgedManifestKey, manifestMap, resolveRelativePath,
    saveManifestMap,
} from './sources.js';
import {
    compareClaims, DEFAULT_REVIEW_MODEL_CALLS, JudgedBatch, judgeSections, type JudgeBudget,
} from './judge.js';
import { entityName, knownKeysOf, resolveEntities } from './resolve.js';
import { basenameOf, canonicalJson, compareStr, factKeyOf, srcRef } from './refs.js';

/** Throws if a batch carries `extracted` items — perception is never memory. */
export function assertStoredBatch(batch: { facts: readonly StoredFact[]; relations: readonly StoredRelation[] }): void {
    const bad = [...batch.facts, ...batch.relations].find((x) => (x.origin as string) === 'extracted');
    if (bad) throw new Error('knowledge: extracted output is never written to a KnowledgeStore');
}

async function extractedFor(
    store: KnowledgeStore, parser: KnowledgeParser, pack: string, src: SourceDocument, sections: ParsedSection[],
    ctx: ExtractContext, hash: string,
): Promise<Extracted> {
    const ref = srcRef(ctx.rootId, src.id);
    const cached = await store.getExtracted?.(hash);
    const hit = cached?.find((e) => e.sourceRef === ref);
    if (hit) return hit;
    const ex: Extracted = parser.extract
        ? { ...parser.extract(src, sections, ctx), sourceHash: hash }
        : {
            sourceRef: ref, sourceHash: hash, edges: [], keys: [{ ref, key: src.id }],
            nodes: [{ ref, kind: 'file', label: basenameOf(src.id), pack, importance: 0, origin: 'extracted', sourceHash: hash }],
        };
    await store.putExtracted?.(hash, [...(cached ?? []).filter((e) => e.sourceRef !== ref), ex]);
    return ex;
}

/** Stored knowledge, indexed for re-anchoring unchanged sections without a model call. */
class StoredIndex {
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
 * `[[Name]]` → the one listed file named `Name` (or `Name.md`), case-insensitively, anywhere
 * in the root. Ambiguous names resolve to none, so the link falls back to the linking folder.
 */
export function nameResolver(paths: readonly string[]): (name: string) => string | undefined {
    const byName = new Map<string, string | null>();
    for (const p of [...paths].sort(compareStr)) {
        const base = basenameOf(p).toLowerCase();
        for (const key of new Set([base, base.replace(/\.md$/, '')])) {
            byName.set(key, byName.has(key) && byName.get(key) !== p ? null : p);
        }
    }
    return (name) => byName.get(name.trim().toLowerCase()) ?? undefined;
}

export async function reviewDirectory(args: ReviewArgs): Promise<ReviewReport> {
    const now = () => Date.now();
    const started = now();
    const { source, store, packs, manifest, signal } = args;
    const rootId = source.rootId;
    const scope = { rootId };
    const progress = (p: Parameters<NonNullable<ReviewArgs['onProgress']>>[0]) => args.onProgress?.(p);
    const report: ReviewReport = {
        rootId, digest: '', extracted: 0, judged: 0, unjudged: 0, invalidJudgements: 0,
        factsWritten: 0, relationsWritten: 0, staleMarked: 0, aborted: false, durationMs: 0,
    };

    const detail = args.plan ? await refreshPlan(args.plan, manifest) : await planDetail(source, manifest, packs);
    progress({ phase: 'plan', done: detail.files.length, total: detail.files.length });
    const composed = composePacks(packs);
    const packIds = packs.map((p) => p.id).sort(compareStr);
    const scopeRecord: ScopeRecord = { rootId, title: source.title, packs: packIds };
    const [storedFacts, storedRelations] = await Promise.all([store.facts(scope), store.relations(scope)]);
    const stored = new StoredIndex(storedFacts, storedRelations);
    const entities = new Map(storedFacts.filter((f) => f.key.startsWith('ent:')).map((f) => [f.key, f] as const));
    const ctx: ExtractContext = {
        rootId,
        knownKeys: knownKeysOf(resolveEntities([...entities.keys()].map((k) => ({ ref: k as NodeRef, key: entityName(k) })))),
        resolvePath: resolveRelativePath,
        resolveName: nameResolver(detail.files.map((f) => f.path)),
    };
    const budget: JudgeBudget = {
        modelCalls: args.budget?.modelCalls ?? DEFAULT_REVIEW_MODEL_CALLS,
        deadline: started + (Number.isFinite(args.budget?.ms) ? args.budget.ms : Number.POSITIVE_INFINITY),
        used: 0, now,
    };
    const batch = new JudgedBatch();
    const judgedPrior = await manifestMap(manifest, judgedManifestKey(rootId));
    const judgedNow = new Map(judgedPrior);
    const filesNow = new Map(detail.prior);
    const newClaims = new Map<string, { text: string; hash: string }>();

    const write = async (withScope: ScopeRecord = scopeRecord) => {
        const out = batch.take();
        assertStoredBatch(out);
        if (out.facts.length === 0 && out.relations.length === 0 && withScope === scopeRecord) return;
        await store.write({ scope: withScope, ...out });
        report.factsWritten += out.facts.length;
        report.relationsWritten += out.relations.length;
    };

    const changed = detail.files.filter((f) => f.changed);
    for (const [i, f] of changed.entries()) {
        if (signal?.aborted) { report.aborted = true; break; }
        progress({ phase: 'extract', done: i, total: changed.length, path: f.path });
        const doc = await source.read(f.path);
        // Recorded, so a file found binary on read is not re-read by every later review.
        if (doc.content.kind === 'text' && isBinaryText(doc.content.text)) { filesNow.set(f.path, f.hash); continue; }
        const src = asPathSource(doc, f.path, f.hash);
        const hit = composed.resolve(src);
        if (!hit) continue;
        const sections = hit.parser.parse(src);
        const ex = await extractedFor(store, hit.parser, hit.pack.id, src, sections, ctx, f.hash);
        report.extracted += 1;

        const spec = hit.pack.judgement;
        let complete = true;
        if (spec && args.judgement && (spec.claims || spec.entities)) {
            const kinds = new Set(spec.nodeKinds);
            const toJudge: Array<{ ref: NodeRef; text: string }> = [];
            for (const node of ex.nodes) {
                const index = node.badges?.[SECTION_INDEX_BADGE];
                const text = typeof index === 'number' ? sections[index]?.text : undefined;
                if (!kinds.has(node.kind) || !text) continue;
                if (judgedPrior.get(node.ref) === hashText(text)) stored.refresh(node.ref, f.path, f.hash, batch, entities);
                else toJudge.push({ ref: node.ref, text });
            }
            const unjudgedBefore = batch.unjudged;
            // Staged per file: a section counts as judged only once its facts are written,
            // so an abort mid-file leaves it to be judged again (written whole or not at all).
            const judgedHere = new Map<NodeRef, string>();
            progress({ phase: 'judge', done: i, total: changed.length, path: f.path });
            try {
                const claims = await judgeSections({
                    port: args.judgement, pack: hit.pack.id, path: f.path, fileHash: f.hash, sections: toJudge,
                    budget, batch, entities, ...(signal ? { signal } : {}),
                    onJudged: (ref) => judgedHere.set(ref, hashText(toJudge.find((s) => s.ref === ref)!.text)),
                });
                for (const c of claims) newClaims.set(c.ref, { text: c.text, hash: f.hash });
            } catch (err) {
                if (!signal?.aborted) throw err;
                batch.take();
                report.aborted = true;
                break;
            }
            for (const [ref, h] of judgedHere) judgedNow.set(ref, h);
            complete = batch.unjudged === unjudgedBefore;
        }
        progress({ phase: 'write', done: i, total: changed.length, path: f.path });
        await write();
        await store.markStale(scope, srcRef(rootId, f.path), f.hash);
        report.staleMarked += 1;
        if (complete) filesNow.set(f.path, f.hash);
    }

    if (!report.aborted) {
        for (const path of detail.plan.removed) {
            await store.markStale(scope, srcRef(rootId, path), '');
            report.staleMarked += 1;
            filesNow.delete(path);
        }
        for (const pack of packs) {
            if (!pack.analyzers) continue;
            for (const ex of await pack.analyzers.analyse(rootId, signal)) {
                await store.putExtracted?.(ex.sourceHash, [ex]);
                report.extracted += 1;
            }
        }
        progress({ phase: 'resolve', done: 0, total: newClaims.size });
        const wantsCompare = packs.some((p) => p.judgement?.contradictions);
        // Run even with the budget spent: compareClaims counts every pair it skips as unjudged.
        if (wantsCompare && args.judgement && args.similar && newClaims.size > 0) {
            const pairs = await args.similar([...newClaims.keys()].sort(compareStr));
            const claims = new Map(newClaims);
            for (const p of pairs) {
                for (const id of [p.a, p.b]) {
                    const fact = !claims.has(id) ? stored.facts.get(factKeyOf(id as NodeRef) ?? '') : undefined;
                    if (fact?.key.startsWith('claim:')) claims.set(id, { text: fact.content, hash: fact.fileAnchors[0]?.contentHash ?? '' });
                }
            }
            progress({ phase: 'judge', done: 0, total: pairs.length });
            await compareClaims({ port: args.judgement, pairs, claims, budget, batch, ...(signal ? { signal } : {}) })
                .catch((err: unknown) => { if (!signal?.aborted) throw err; });
            if (signal?.aborted) report.aborted = true;
        }
    }

    report.judged = batch.judged;
    report.unjudged = batch.unjudged;
    report.invalidJudgements = batch.invalid;
    report.digest = sha256Hex(canonicalJson({ rootId, packs: packIds, files: [...filesNow].sort((a, b) => compareStr(a[0], b[0])) }));
    await saveManifestMap(manifest, filesManifestKey(rootId), filesNow);
    await saveManifestMap(manifest, judgedManifestKey(rootId), judgedNow);
    await write(report.aborted ? scopeRecord : { ...scopeRecord, lastDigest: report.digest, lastReviewedAt: new Date(now()).toISOString() });
    report.durationMs = now() - started;
    progress({ phase: 'done', done: changed.length, total: changed.length });
    return report;
}
