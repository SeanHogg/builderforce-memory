/**
 * src/contract/regression.findingsProvider.test.ts — `buildGraph` with a `findings` provider (PRD 41 §B5 host seam). The
 * VS Code fact index serves some finding kinds from its own incremental rule rows; it needs them computed over the WHOLE
 * graph, also when the returned view is truncated. The provider sees the untruncated graph, its findings replace the
 * built-in and pack findings of the kinds it owns, and without a provider the output is byte-identical.
 */

import { buildGraph } from '../knowledge/build.js';
import { brokenLinkRule } from '../knowledge/packs/docs.js';
import { findingId, requestKeyFor } from '../knowledge/findings.js';
import type { BuildGraphInput, Finding, FindingsProvider, KnowledgeGraph, KnowledgePack, StoredFact } from '../knowledge/model.js';
import { edge, extractedOf, fileNode, ref, ROOT } from '../knowledge/testkit.test-util.js';

const PACK: KnowledgePack = { id: 'docs', parsers: [], findings: [brokenLinkRule] };

/** 40 linked files, a cycle back to the start, and one dangling link (a `broken-link` finding). */
function chain(): BuildGraphInput {
    const paths = Array.from({ length: 40 }, (_, i) => `d${i % 4}/f${String(i).padStart(2, '0')}.md`);
    const extracted = paths.map((p, i) => extractedOf(fileNode(p), i + 1 < paths.length ? [edge(p, paths[i + 1]!)] : [edge(p, paths[0]!)]));
    extracted[30]!.edges.push(edge(paths[30]!, 'gone/missing.md'));
    return { rootId: ROOT, packs: [PACK], extracted, facts: [] as StoredFact[], relations: [] };
}

const VIEW = { maxNodes: 8, focus: [ref('d0/f00.md')] };

describe('regression: buildGraph findings provider', () => {
    it('without a provider the output is byte-identical, and its findings are the whole graph\'s', () => {
        const plain = buildGraph(chain(), VIEW);
        expect(plain.stats.truncated).toBe(true);
        expect(JSON.stringify(buildGraph(chain(), { ...VIEW, findings: undefined }))).toBe(JSON.stringify(plain));
        // A provider that owns nothing changes nothing either.
        const none: FindingsProvider = { kinds: [], findings: () => [] };
        expect(JSON.stringify(buildGraph(chain(), { ...VIEW, findings: none }))).toBe(JSON.stringify(plain));
        const whole = buildGraph(chain(), { maxNodes: 2000 });
        expect(whole.stats.truncated).toBe(false);
        expect(plain.findings).toEqual(whole.findings);
    });

    it('a truncated build gets the provider\'s findings for the kinds it owns, computed over the whole graph', () => {
        const plain = buildGraph(chain(), VIEW);
        expect(plain.findings.some((f) => f.kind === 'broken-link')).toBe(true);

        let seen: KnowledgeGraph | undefined;
        let seenRefs = 0;
        const far = ref('d2/f22.md');
        const provider: FindingsProvider = {
            kinds: ['broken-link', 'cycle'],
            findings: ({ graph, knowledge }) => {
                seen = graph;
                seenRefs = knowledge.byRef.size;
                const mine: Finding = { id: '', kind: 'broken-link', pack: 'docs', nodes: [far, far], severity: 'error', params: { from: 'index' } };
                const notMine: Finding = { id: 'x', kind: 'god-module', pack: 'docs', nodes: [far], severity: 'warn', params: {} };
                return [mine, notMine];
            },
        };
        const g = buildGraph(chain(), { ...VIEW, findings: provider });

        // The provider saw the WHOLE graph, not the cut view.
        expect(g.stats.truncated).toBe(true);
        expect(seen!.nodes.length).toBe(buildGraph(chain(), { maxNodes: 2000 }).nodes.length);
        expect(seen!.nodes.length).toBeGreaterThan(g.nodes.length);
        expect(seenRefs).toBe(seen!.nodes.length);

        // Owned kinds come only from the provider (no built-in cycle, no pack broken-link); other kinds pass through.
        expect(g.findings.filter((f) => f.kind === 'cycle')).toEqual([]);
        const broken = g.findings.filter((f) => f.kind === 'broken-link');
        expect(broken).toEqual([{
            // Normalized exactly like a pack finding: the id hashes the nodes as given, then the nodes are deduped and sorted.
            id: findingId('broken-link', [far, far]), kind: 'broken-link', pack: 'docs', nodes: [far], severity: 'error', params: { from: 'index' },
            requestKey: requestKeyFor('broken-link'),
        }]);
        expect(g.findings.some((f) => f.kind === 'god-module' && f.id === 'x')).toBe(false);
        const others = (fs: readonly Finding[]) => fs.filter((f) => f.kind !== 'broken-link' && f.kind !== 'cycle');
        expect(others(g.findings)).toEqual(others(plain.findings));
        // Sorted like every finding list: severity, then id.
        expect(g.findings[0]!.severity).toBe('error');
        // Truncation keeps the provider's finding nodes, as it keeps any finding's.
        expect(g.nodes.some((n) => n.ref === far)).toBe(true);
    });
});
