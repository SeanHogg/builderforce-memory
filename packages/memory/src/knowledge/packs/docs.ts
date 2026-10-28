/**
 * knowledge/packs/docs.ts — the docs pack, built into memory (PRD 38 §4.5).
 *
 * Parsers are the ONE builtin ingest parsers (markdown, HTML, plain text); their `extract`
 * lives in `ingest/docsExtract.ts`. Partitions: one plane per top-level folder, no `allow`
 * (so no verdicts). Findings: broken links and orphan documents. Judgement: claims,
 * entities and contradictions per section.
 */

import { htmlParser, markdownParser, plainTextParser } from '../../ingest/parsers.js';
import { DOCS_PACK_ID } from '../../ingest/docsExtract.js';
import type { Finding, FindingRule, KnowledgeGraph, KnowledgePack, NodeRef } from '../model.js';
import { findingId, MAX_FINDING_NODES, requestKeyFor } from '../findings.js';
import { TOP_LEVEL_FOLDER } from '../partitions.js';
import { basenameOf, compareStr, fileRefOf, pathOfRef } from '../refs.js';

export { DOCS_PACK_ID };

/** Targets with these extensions (or none) are documents; a link to anything else is not judged broken. */
const DOC_TARGET = /(?:\.(?:md|markdown|mdx|html?|txt|rst|adoc)|\/[^./]*|^[^./]*)$/i;
/** Entry documents are expected to have no inbound links. */
const ENTRY_DOC = /^(readme|index|home|_index|summary)\.[a-z]+$/i;

function fileOf(g: KnowledgeGraph): Map<NodeRef, NodeRef> {
    return new Map(g.nodes.map((n) => [n.ref, fileRefOf(n.ref)] as const));
}

/** Links to documents that are not in the directory, grouped per linking file. */
export const brokenLinkRule: FindingRule = {
    id: 'docs.broken-link',
    find(g) {
        const kind = new Map(g.nodes.map((n) => [n.ref, n.kind] as const));
        const file = fileOf(g);
        const bySource = new Map<NodeRef, NodeRef[]>();
        for (const e of g.edges) {
            if (e.kind !== 'links' || kind.get(e.to) !== 'missing') continue;
            if (!DOC_TARGET.test(pathOfRef(e.to) ?? '')) continue;
            const source = file.get(e.from) ?? e.from;
            const list = bySource.get(source) ?? [];
            list.push(e.to);
            bySource.set(source, list);
        }
        return [...bySource].sort((a, b) => compareStr(a[0], b[0])).map(([source, targets]): Finding => {
            const unique = [...new Set(targets)].sort(compareStr);
            const nodes = [source, ...unique];
            return {
                id: findingId('broken-link', nodes), kind: 'broken-link', pack: DOCS_PACK_ID,
                nodes: nodes.slice(0, MAX_FINDING_NODES), severity: 'warn',
                params: { count: unique.length, file: pathOfRef(source) ?? source },
                requestKey: requestKeyFor('broken-link'),
            };
        });
    },
};

/** Documents no other document links to (entry documents aside), as one finding. */
export const orphanDocumentRule: FindingRule = {
    id: 'docs.orphan',
    find(g) {
        const file = fileOf(g);
        const linked = new Set<NodeRef>();
        for (const e of g.edges) {
            if (e.kind !== 'links') continue;
            const from = file.get(e.from) ?? e.from, to = file.get(e.to) ?? e.to;
            if (from !== to) linked.add(to);
        }
        const orphans = g.nodes
            .filter((n) => n.pack === DOCS_PACK_ID && n.kind === 'file' && !linked.has(n.ref))
            .filter((n) => !ENTRY_DOC.test(basenameOf(pathOfRef(n.ref) ?? '')))
            .map((n) => n.ref).sort(compareStr);
        if (orphans.length === 0) return [];
        return [{
            id: findingId('orphan', orphans), kind: 'orphan', pack: DOCS_PACK_ID,
            nodes: orphans.slice(0, MAX_FINDING_NODES), severity: 'info',
            params: { count: orphans.length }, requestKey: requestKeyFor('orphan'),
        }];
    },
};

export const DOCS_PACK: KnowledgePack = {
    id: DOCS_PACK_ID,
    parsers: [markdownParser, htmlParser, plainTextParser],
    partitions: {
        partitions: [
            { id: TOP_LEVEL_FOLDER, order: 0, match: ['*/**'] },
            { id: '(root)', order: 0, match: ['*'] },
        ],
    },
    findings: [brokenLinkRule, orphanDocumentRule],
    judgement: { nodeKinds: ['section'], claims: true, entities: true, contradictions: true },
};
