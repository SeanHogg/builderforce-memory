/**
 * Shared fixtures for the knowledge tests: an in-memory DirectorySource that records its
 * reads, an in-memory snapshot persistence, a fake JudgementPort and node/edge builders.
 */

import { hashText } from '../ingest/hash.js';
import { buildGraph } from './build.js';
import type {
    DirectorySource, Extracted, JudgementPort, JudgementRequest, JudgementResult, KEdge, KNode, KnowledgeStore, NodeRef,
} from './model.js';
import { DOCS_PACK } from './packs/docs.js';
import { LocalSnapshotStore, type SnapshotPersistence } from './stores/localSnapshotStore.js';
import { srcRef } from './refs.js';

export const ROOT = 'r';

export interface SpySource extends DirectorySource {
    files: Record<string, string>;
    reads: string[];
}

export function memSource(files: Record<string, string>, rootId = ROOT): SpySource {
    const src: SpySource = {
        rootId, title: 'Fixture', files, reads: [],
        async *list() {
            for (const path of Object.keys(src.files)) {
                const text = src.files[path]!;
                yield { path, hash: hashText(text), size: text.length };
            }
        },
        async read(path: string) {
            src.reads.push(path);
            return { id: path, tenantId: 't', content: { kind: 'text', text: src.files[path]! } };
        },
    };
    return src;
}

export function memPersistence(): SnapshotPersistence & { value?: string } {
    const p: SnapshotPersistence & { value?: string } = {
        async read() { return p.value; },
        async write(s: string) { p.value = s; },
    };
    return p;
}

export function memStore(): LocalSnapshotStore {
    return new LocalSnapshotStore(memPersistence(), { now: () => '2026-10-28T00:00:00.000Z' });
}

/** Answers every extract with one claim + one entity, every compare with `contradicts`. */
export function fakeJudge(
    shape: (req: JudgementRequest) => JudgementResult = defaultJudgement,
): JudgementPort & { requests: JudgementRequest[] } {
    const port = {
        requests: [] as JudgementRequest[],
        async judge(req: JudgementRequest) {
            port.requests.push(req);
            return shape(req);
        },
    };
    return port;
}

function defaultJudgement(req: JudgementRequest): JudgementResult {
    if (req.kind === 'compare') {
        return { compare: [{ a: req.items[0]!.ref, b: req.items[1]!.ref, relation: 'contradicts', confidence: 0.9 }], invalid: 0 };
    }
    return {
        extract: req.items.map((it) => ({
            ref: it.ref,
            claims: [{ subject: 'Lease', predicate: 'ends', object: it.text.split(/\s+/)[0] ?? 'x', quote: it.text.slice(0, 20) }],
            entities: [{ name: 'Lease', type: 'contract' }],
        })),
        invalid: 0,
    };
}

export function fileNode(path: string, extra: Partial<KNode> = {}): KNode {
    return {
        ref: srcRef(ROOT, path), kind: 'file', label: path, pack: 'docs', importance: 0, origin: 'extracted',
        sourceHash: `h:${path}`, ...extra,
    };
}

export function edge(from: string, to: string, kind = 'links', extra: Partial<KEdge> = {}): KEdge {
    return { from: srcRef(ROOT, from), to: srcRef(ROOT, to), kind, weight: 1, origin: 'extracted', ...extra };
}

export function extractedOf(node: KNode, edges: KEdge[] = []): Extracted {
    return { sourceRef: node.ref, sourceHash: node.sourceHash ?? '', nodes: [node], edges, keys: [] };
}

export const ref = (path: string): NodeRef => srcRef(ROOT, path);

/** The graph of a reviewed source: its files' cached extracts joined with what the store holds. */
export async function graphOf(source: SpySource, store: KnowledgeStore) {
    const extracted: Extracted[] = [];
    for await (const f of source.list()) {
        const hit = (await store.getExtracted!(f.hash))?.find((e) => e.sourceRef === srcRef(ROOT, f.path));
        if (hit) extracted.push(hit);
    }
    return buildGraph({
        rootId: ROOT, extracted, packs: [DOCS_PACK],
        facts: await store.facts({ rootId: ROOT }), relations: await store.relations({ rootId: ROOT }),
    });
}
