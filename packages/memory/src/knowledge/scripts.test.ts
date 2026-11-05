/**
 * src/knowledge/scripts.test.ts
 * Shell scripts are never documents; `[[Name]]` and links inside markdown code are not edges.
 */

import { InMemoryIngestManifest } from '../ingest/IngestionPipeline.js';
import { markdownParser, plainTextParser } from '../ingest/parsers.js';
import type { SourceDocument } from '../ingest/types.js';
import type { ExtractContext, KnowledgePack } from './model.js';
import { DOCS_PACK } from './packs/docs.js';
import { planReview } from './plan.js';
import { asPathSource, resolveRelativePath } from './sources.js';
import { ROOT } from './testkit.test-util.js';

const ctx: ExtractContext = { rootId: ROOT, knownKeys: new Map(), resolvePath: resolveRelativePath };

function srcOf(path: string, text: string): SourceDocument {
    return asPathSource({ id: path, tenantId: 't', content: { kind: 'text', text } }, path, 'h');
}

function linksOf(text: string, parser = markdownParser, path = 'a.md') {
    const src = srcOf(path, text);
    return parser.extract!(src, parser.parse(src), ctx).edges.filter((e) => e.kind === 'links');
}

function listing(files: Record<string, string>) {
    return {
        rootId: ROOT, title: 't',
        async *list() { for (const [path, text] of Object.entries(files)) yield { path, hash: `h:${path}`, size: text.length }; },
        async read(path: string): Promise<SourceDocument> {
            return { id: path, tenantId: 't', content: { kind: 'text', text: files[path]! } };
        },
    };
}

const SHELL = 'if [[ -f x ]]; then :; fi';

describe('scripts are not documents', () => {
    it.each(['run.sh', 'run.bash', 'run.zsh', 'run.ps1', 'run.bat', 'run.cmd', 'RUN.SH'])('%s is not claimed by the docs pack', async (name) => {
        const plan = await planReview(listing({ [name]: SHELL, 'a.md': '# A\n' }), new InMemoryIngestManifest(), [DOCS_PACK]);
        expect(plan.skipped).toContainEqual({ path: name, reason: 'no-pack' });
        expect(plan.files).toBe(1);
    });

    it('a host parser can still claim a .sh file', async () => {
        const hostPack: KnowledgePack = {
            id: 'host', findings: [],
            parsers: [{ id: 'sh', accepts: (s) => s.id.endsWith('.sh'), parse: () => [], extract: () => ({ sourceRef: 'x', sourceHash: 'h', nodes: [], edges: [], keys: [] }) } as never],
        } as never;
        const plan = await planReview(listing({ 'run.sh': SHELL }), new InMemoryIngestManifest(), [DOCS_PACK, hostPack]);
        expect(plan.files).toBe(1);
        expect(plan.byPack['host']).toBe(1);
    });
});

describe('wiki and markdown links ignore code', () => {
    it('prose links still resolve', () => {
        expect(linksOf('See [[notes]].')).toHaveLength(1);
    });

    it('fenced, tilde-fenced, tab- and space-indented, and inline code make no link', () => {
        expect(linksOf('# T\n```bash\nif [[ -f x ]]; then\n```\n')).toHaveLength(0);
        expect(linksOf('# T\n~~~\n[[x]]\n~~~\n')).toHaveLength(0);
        expect(linksOf('# T\n\n    [[x]]\n')).toHaveLength(0);
        expect(linksOf('# T\n\n\t[[x]]\n')).toHaveLength(0);
        expect(linksOf('# T\nuse `[[x]]` here\n')).toHaveLength(0);
    });

    it('a double-backtick span and a span over a soft line break are code', () => {
        expect(linksOf('# T\nuse ``a ` [[x]]`` here\n')).toHaveLength(0);
        expect(linksOf('# T\nuse `a\n[[x]]` here\n')).toHaveLength(0);
    });

    it('handles CRLF input', () => {
        expect(linksOf('# T\r\n```\r\n[[x]]\r\n```\r\nand [[y]]\r\n')).toHaveLength(1);
    });

    it('a closed fence ends: later prose [[y]] is a link', () => {
        const links = linksOf('# T\n```\n[[x]]\n```\nafter [[y]]\n');
        expect(links).toHaveLength(1);
        expect(links[0]!.to).toContain('y');
    });

    it('a four-backtick fence holding ``` stays open; ``` inside ~~~ too', () => {
        expect(linksOf('# T\n````\n```\n[[x]]\n```\n````\n')).toHaveLength(0);
        expect(linksOf('# T\n~~~\n```\n[[x]]\n```\n~~~\n')).toHaveLength(0);
    });

    it('an unclosed fence runs to the end', () => {
        expect(linksOf('# T\n```\n[[x]]\n# not a heading\n[[z]]\n')).toHaveLength(0);
    });

    it('a ~~~ block holding a # line is one section and leaks no link', () => {
        const text = '# T\n~~~bash\n# check\nif [[ -f x ]]; then\n~~~\n';
        expect(markdownParser.parse(srcOf('a.md', text))).toHaveLength(1);
        expect(linksOf(text)).toHaveLength(0);
    });

    it('a fence inside a list item and a blockquote fence are fences', () => {
        expect(linksOf('# T\n- item\n\n    ```\n    [[x]]\n    ```\n')).toHaveLength(0);
        expect(linksOf('# T\n> ```\n> [[x]]\n> ```\n')).toHaveLength(0);
    });

    it('a markdown link inside a fence is not an edge', () => {
        expect(linksOf('# T\n```\n[g](guide.md)\n```\n')).toHaveLength(0);
        expect(linksOf('# T\n[g](guide.md)\n')).toHaveLength(1);
    });

    it('a loose-list continuation keeps its link', () => {
        expect(linksOf('# T\n- item\n\n    continues with [[y]]\n')).toHaveLength(1);
    });

    it('plain text keeps an indented paragraph link', () => {
        expect(linksOf('# T\n\n    see [[y]]\n', plainTextParser, 'a.txt')).toHaveLength(1);
    });
});

describe('markdown section text', () => {
    it('keeps a section first-line indentation; a column-0 section is unchanged', () => {
        const indented = markdownParser.parse(srcOf('a.md', '# T\n\n    code\n'));
        expect(indented[0]!.text).toBe('    code');
        const plain = markdownParser.parse(srcOf('a.md', '# T\n\nhello\nworld\n\n'));
        expect(plain[0]!.text).toBe('hello\nworld');
    });
});
