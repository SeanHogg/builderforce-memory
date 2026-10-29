/**
 * knowledge/review.ts — `reviewDirectory`: point it at a folder and it learns what the
 * folder says (PRD 38 §4.3).
 *
 *   plan (walk + manifest diff) → per changed file: read · parse (the ONE ParserRegistry,
 *   composed from the packs) · extract (cached by content hash) · judge (changed sections
 *   of judged kinds, within budget) · write · markStale → analyzers → compare near-duplicate
 *   claims → manifest + scope digest. Claims whose compare pairs were skipped wait in the
 *   compare manifest for *Continue*.
 *
 * Store judgement, compute perception: only `model` facts and relations are written —
 * `extracted` output goes to the extracted cache, never to memory ({@link assertStoredBatch}).
 * A file whose read throws is skipped, never retired: its knowledge and manifest hash stay
 * as they were (`readErrors`); only a file the fresh plan no longer lists is removed.
 * Abort is honoured between files and inside judgement; a file is written whole or not at
 * all. A file whose sections were cut by the budget stays "changed" in the manifest, so
 * *Continue* (the confirmed plan passed back as `args.plan`, no re-walk) judges only the
 * sections not judged yet.
 */

import { hashText } from '../ingest/hash.js';
import type { SourceDocument } from '../ingest/types.js';
import { sha256Hex } from '../retrieval/knowledge-index/sha256.js';
import type { NodeRef, ReviewArgs, ReviewReport, ScopeRecord, StoredFact, StoredRelation } from './model.js';
import { SECTION_INDEX_BADGE } from '../ingest/docsExtract.js';
import { planDetail, refreshPlan } from './plan.js';
import {
    composePacks, filesManifestKey, isBinaryText, judgedManifestKey, manifestMap, saveExtractorVersion, saveManifestMap,
} from './sources.js';
import { extractFileDetailed, knownKeysOfEntities } from './extractFile.js';
import { DEFAULT_REVIEW_MODEL_CALLS, JudgedBatch, judgeSections, type CompareOutcome, type JudgeBudget } from './judge.js';
import { compareNew, pendingClaims, saveAwaiting } from './reviewCompare.js';
import { factsWithoutAnchor, StoredIndex } from './reviewStored.js';
import { canonicalJson, compareStr, srcRef } from './refs.js';

/** Throws if a batch carries `extracted` items — perception is never memory. */
export function assertStoredBatch(batch: { facts: readonly StoredFact[]; relations: readonly StoredRelation[] }): void {
    const bad = [...batch.facts, ...batch.relations].find((x) => (x.origin as string) === 'extracted');
    if (bad) throw new Error('knowledge: extracted output is never written to a KnowledgeStore');
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
        factsWritten: 0, relationsWritten: 0, staleMarked: 0, readErrors: 0, aborted: false, durationMs: 0,
    };

    const detail = args.plan
        ? await refreshPlan(args.plan, manifest)
        : await planDetail(source, manifest, packs, args.force ? { force: true } : {});
    progress({ phase: 'plan', done: detail.files.length, total: detail.files.length });
    const composed = composePacks(packs);
    const packIds = packs.map((p) => p.id).sort(compareStr);
    const scopeRecord: ScopeRecord = { rootId, title: source.title, packs: packIds };
    const [storedFacts, storedRelations] = await Promise.all([store.facts(scope), store.relations(scope)]);
    const stored = new StoredIndex(storedFacts, storedRelations);
    const entities = new Map(storedFacts.filter((f) => f.key.startsWith('ent:')).map((f) => [f.key, f] as const));
    const knownKeys = knownKeysOfEntities(entities.keys());
    const budget: JudgeBudget = {
        modelCalls: args.budget?.modelCalls ?? DEFAULT_REVIEW_MODEL_CALLS,
        deadline: started + (Number.isFinite(args.budget?.ms) ? args.budget.ms : Number.POSITIVE_INFINITY),
        used: 0, now,
    };
    const batch = new JudgedBatch();
    const judgedPrior = await manifestMap(manifest, judgedManifestKey(rootId));
    const judgedNow = new Map(judgedPrior);
    const filesNow = new Map(detail.prior);
    // Claims awaiting comparison: those a previous run could not compare, then this run's.
    const newClaims = await pendingClaims(manifest, rootId, stored.facts);

    const write = async (withScope: ScopeRecord = scopeRecord) => {
        const out = batch.take();
        assertStoredBatch(out);
        if (out.facts.length === 0 && out.relations.length === 0 && withScope === scopeRecord) return;
        await store.write({ scope: withScope, ...out });
        report.factsWritten += out.facts.length;
        report.relationsWritten += out.relations.length;
    };

    // A source that is gone: its knowledge goes stale at no hash, its judged sections are forgotten, and
    // facts other sources still back drop its anchor. `rewritten` is the run's one record of such facts,
    // so a later retire() never restores an anchor an earlier one dropped.
    const rewritten = new Map<string, StoredFact>();
    const retire = async (paths: readonly string[]) => {
        const live = new Map([...stored.facts, ...entities, ...rewritten]);
        for (const path of paths) {
            await store.markStale(scope, srcRef(rootId, path), '');
            report.staleMarked += 1;
            filesNow.delete(path);
            const sectionPrefix = `${srcRef(rootId, path)}#`;
            for (const section of [...judgedNow.keys()]) if (section.startsWith(sectionPrefix)) judgedNow.delete(section);
            for (const fact of factsWithoutAnchor(live.values(), path)) {
                batch.fact(fact);
                live.set(fact.key, fact);
                rewritten.set(fact.key, fact);
                if (entities.has(fact.key)) entities.set(fact.key, fact);
            }
        }
        await write();
    };

    const changed = detail.files.filter((f) => f.changed);
    for (const [i, f] of changed.entries()) {
        if (signal?.aborted) { report.aborted = true; break; }
        progress({ phase: 'extract', done: i, total: changed.length, path: f.path });
        // A read that throws (permissions, a lock, a race with a delete) says nothing about the file:
        // skipped with its knowledge and manifest hash intact, tried again next review.
        let doc: SourceDocument;
        try { doc = await source.read(f.path); } catch { report.readErrors += 1; continue; }
        // Recorded, so a file found binary on read is not re-read by every later review.
        if (doc.content.kind === 'text' && isBinaryText(doc.content.text)) {
            filesNow.set(f.path, f.hash);
            await store.markStale(scope, srcRef(rootId, f.path), f.hash);
            report.staleMarked += 1;
            continue;
        }
        const extraction = await extractFileDetailed({
            rootId, path: f.path, hash: f.hash, doc, packs, store, composed, knownKeys,
        });
        if (!extraction) continue;
        const { extracted: ex, sections, pack } = extraction;
        report.extracted += 1;

        const spec = pack.judgement;
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
                    port: args.judgement, pack: pack.id, path: f.path, fileHash: f.hash, sections: toJudge,
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
        await retire(detail.plan.removed);
        for (const pack of packs) {
            if (!pack.analyzers) continue;
            for (const ex of await pack.analyzers.analyse(rootId, signal)) {
                await store.putExtracted?.(ex.sourceHash, [ex]);
                report.extracted += 1;
            }
        }
        progress({ phase: 'resolve', done: 0, total: newClaims.size });
    }

    // The compare phase needs a pack asking for it AND both ports; without them nothing waits for it.
    const canCompare = packs.some((p) => p.judgement?.contradictions) && !!args.judgement && !!args.similar;
    let outcome: CompareOutcome | undefined;
    if (!report.aborted && canCompare) {
        outcome = await compareNew({
            args, pending: newClaims, known: stored.facts, budget, batch,
            onPairs: (total) => progress({ phase: 'judge', done: 0, total }),
        });
        // Aborted only if the abort landed before the last pair was compared.
        if (outcome?.aborted) report.aborted = true;
    }
    await saveAwaiting(manifest, rootId, newClaims, outcome, canCompare);

    report.judged = batch.judged;
    report.unjudged = batch.unjudged;
    report.invalidJudgements = batch.invalid;
    report.digest = sha256Hex(canonicalJson({ rootId, packs: packIds, files: [...filesNow].sort((a, b) => compareStr(a[0], b[0])) }));
    await saveManifestMap(manifest, filesManifestKey(rootId), filesNow);
    await saveManifestMap(manifest, judgedManifestKey(rootId), judgedNow);
    // Recorded only once every changed file was handled: an abort or a skipped read leaves the next review to redo them.
    if (!report.aborted && report.readErrors === 0) await saveExtractorVersion(manifest, rootId);
    await write(report.aborted ? scopeRecord : { ...scopeRecord, lastDigest: report.digest, lastReviewedAt: new Date(now()).toISOString() });
    report.durationMs = now() - started;
    progress({ phase: 'done', done: changed.length, total: changed.length });
    return report;
}
