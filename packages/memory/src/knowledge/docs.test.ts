/**
 * src/knowledge/docs.test.ts
 * The docs pack: the builtin parsers' `extract` (sections, contains, relative and wiki
 * links, mentions, front-matter keys) and its findings (broken link, orphan), plus cards.
 */

import { htmlParser, markdownParser, plainTextParser } from '../ingest/parsers.js';
import type { SourceDocument } from '../ingest/types.js';
import { buildGraph } from './build.js';
import { toCards } from './cards.js';
import type { ExtractContext, Extracted, NodeRef } from './model.js';
import { DOCS_PACK } from './packs/docs.js';
import { asPathSource, resolveRelativePath } from './sources.js';
import { ref, ROOT } from './testkit.test-util.js';

const FILES: Record<string, string> = {
    'README.md': '# Home\nSee [the guide](docs/guide.md) and [[notes]].\n',
    'docs/guide.md': '---\ntitle: The Guide\ntags: [Alpha, beta]\n---\n# Guide\nIntro about the Lease.\n## Setup\nSee [missing](./nope.md), [ext](https://x.com) and ![img](pic.png).\n',
    'notes.md': '# Notes\nBack to [home](README.md), [the page](docs/page.html) and [[docs/plain.txt]].\n',
    'docs/orphan.md': '# Lonely\nNobody links here.\n',
    'docs/page.html': '<html><title>Page</title><body><a href="guide.md">g</a></body></html>',
    'docs/plain.txt': 'Plain text pointing at [[guide]].',
};

const ctx: ExtractContext = {
    rootId: ROOT,
    knownKeys: new Map<string, NodeRef>([['lease', 'ent:docs:lease']]),
    resolvePath: resolveRelativePath,
};

function source(path: string): SourceDocument {
    return asPathSource({ id: path, tenantId: 't', content: { kind: 'text', text: FILES[path]! } }, path, `h:${path}`);
}

function extract(path: string): Extracted {
    const src = source(path);
    const parser = [markdownParser, htmlParser, plainTextParser].find((p) => p.accepts(src))!;
    return parser.extract!(src, parser.parse(src), ctx);
}

const all = Object.keys(FILES).map(extract);
const kinds = (ex: Extracted, kind: string) => ex.edges.filter((e) => e.kind === kind).map((e) => `${e.from} -> ${e.to}`);

describe('docs extract', () => {
    it('emits a file node and section nodes from the heading breadcrumb, joined by contains', () => {
        const guide = extract('docs/guide.md');
        expect(guide.sourceRef).toBe(ref('docs/guide.md'));
        expect(guide.sourceHash).toBe('h:docs/guide.md');
        expect(guide.nodes.map((n) => `${n.kind}:${n.ref}`)).toEqual([
            `file:${ref('docs/guide.md')}`,
            `section:${ref('docs/guide.md')}#guide`,
            `section:${ref('docs/guide.md')}#guide/setup`,
        ]);
        expect(kinds(guide, 'contains')).toEqual([
            `${ref('docs/guide.md')} -> ${ref('docs/guide.md')}#guide`,
            `${ref('docs/guide.md')}#guide -> ${ref('docs/guide.md')}#guide/setup`,
        ]);
        expect(guide.nodes.every((n) => n.origin === 'extracted')).toBe(true);
        expect(guide.nodes[0]!.label).toBe('The Guide');
    });

    it('resolves relative and wiki links; skips external links and images', () => {
        expect(kinds(extract('README.md'), 'links')).toEqual([
            `${ref('README.md')}#home -> ${ref('docs/guide.md')}`,
            `${ref('README.md')}#home -> ${ref('notes.md')}`,
        ]);
        expect(kinds(extract('docs/guide.md'), 'links')).toEqual([`${ref('docs/guide.md')}#guide/setup -> ${ref('docs/nope.md')}`]);
        expect(kinds(extract('docs/page.html'), 'links')).toEqual([`${ref('docs/page.html')} -> ${ref('docs/guide.md')}`]);
        expect(kinds(extract('docs/plain.txt'), 'links')).toEqual([`${ref('docs/plain.txt')}#body -> ${ref('docs/guide.md')}`]);
    });

    it('mentions known keys by exact normalized match', () => {
        expect(kinds(extract('docs/guide.md'), 'mentions')).toEqual([`${ref('docs/guide.md')}#guide -> ent:docs:lease`]);
        expect(kinds(extract('notes.md'), 'mentions')).toEqual([]);
    });

    it('keys carry the path, basename, heading slugs and front-matter values', () => {
        const keys = extract('docs/guide.md').keys.map((k) => k.key);
        expect(keys).toEqual(expect.arrayContaining(['docs/guide.md', 'guide', 'guide.md', 'the guide', 'alpha', 'beta', 'setup']));
    });

    it('is pure: the same source extracts to the same output', () => {
        expect(JSON.stringify(extract('docs/guide.md'))).toBe(JSON.stringify(extract('docs/guide.md')));
    });
});

describe('docs findings and cards', () => {
    const g = buildGraph({ rootId: ROOT, extracted: all, facts: [], relations: [], packs: [DOCS_PACK] });

    it('reports a broken link per linking file', () => {
        const broken = g.findings.filter((f) => f.kind === 'broken-link');
        expect(broken).toHaveLength(1);
        expect(broken[0]!.nodes).toEqual([ref('docs/guide.md'), ref('docs/nope.md')]);
        expect(broken[0]!.requestKey).toBe('knowledgeGraph.fix.brokenLink');
        expect(g.nodes.find((n) => n.ref === ref('docs/nope.md'))?.kind).toBe('missing');
    });

    it('reports orphan documents (entry documents aside)', () => {
        const orphan = g.findings.find((f) => f.kind === 'orphan');
        expect(orphan?.nodes).toEqual([ref('docs/orphan.md')]);
    });

    it('places docs on top-level folder planes without verdicts', () => {
        expect(g.nodes.find((n) => n.ref === ref('docs/guide.md'))?.partition).toBe('docs');
        expect(g.nodes.find((n) => n.ref === ref('README.md'))?.partition).toBe('(root)');
        expect(g.edges.some((e) => e.verdict)).toBe(false);
    });

    it('cards: one per node, a section carries its breadcrumb and excerpt', () => {
        const cards = toCards(g);
        expect(cards.map((c) => c.id)).toEqual(g.nodes.map((n) => n.ref));
        const setup = cards.find((c) => c.id === `${ref('docs/guide.md')}#guide/setup`)!;
        expect(setup.text).toContain('Guide › Setup');
        expect(setup.text).toContain('docs/nope.md');
        expect(setup.keys).toEqual(expect.arrayContaining(['docs/guide.md', 'setup', 'alpha']));
        expect(setup.hash).toMatch(/^[0-9a-f]{64}$/);
    });
});
