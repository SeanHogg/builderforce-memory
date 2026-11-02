/**
 * src/ingest/docsExtract.test.ts
 * Section ref bounding: a heading breadcrumb's slug path is capped so the whole ref
 * (`fileRef#slug`) stays under `MAX_SECTION_REF_LEN` — the API validates refs at 512
 * characters and one over-long ref rejects the entire extracted upload. Over the cap the
 * slug is truncated with a stable hash discriminator; a slug that fits is unchanged.
 */

import { markdownParser } from './parsers.js';
import { extractMarkdown, MAX_SECTION_REF_LEN } from './docsExtract.js';
import type { ExtractContext, NodeRef } from '../knowledge/model.js';
import { srcRef } from '../knowledge/refs.js';
import { asPathSource, resolveRelativePath } from '../knowledge/sources.js';
import { ROOT } from '../knowledge/testkit.test-util.js';
import type { SourceDocument } from './types.js';

const ctx: ExtractContext = {
    rootId: ROOT,
    knownKeys: new Map<string, NodeRef>(),
    resolvePath: resolveRelativePath,
};

function md(path: string, text: string): SourceDocument {
    return asPathSource({ id: path, tenantId: 't', content: { kind: 'text', text } }, path, `h:${path}`);
}

function extract(path: string, text: string) {
    const src = md(path, text);
    return extractMarkdown(src, markdownParser.parse(src), ctx);
}

/** Six nested headings whose joined slug path is far over the cap. */
const LONG_CRUMBS = [
    'an-extremely-long-and-descriptive-top-level-heading-name-that-just-keeps-going-and-going',
    'a-rather-deeply-nested-second-level-section-heading-with-plenty-of-verbose-explanatory-wording',
    'an-even-deeper-third-level-heading-with-verbose-wording-describing-internal-behaviour-in-detail',
    'a-fourth-level-heading-describing-configuration-behaviour-and-its-subtle-interactions-in-detail',
    'a-fifth-level-heading-for-an-obscure-edge-case-of-the-system-that-needs-a-lengthy-explanation',
    'a-sixth-level-heading-requiring-its-own-substantial-clarification-to-reach-the-length-needed',
];
const LONG_DOC = LONG_CRUMBS.map((c, i) => `${'#'.repeat(i + 1)} ${c}\nbody ${i}\n`).join('\n');

const sectionRefsOf = (path: string, text: string): NodeRef[] =>
    extract(path, text).nodes.filter((n) => n.kind === 'section').map((n) => n.ref);

describe('docsExtract section ref cap', () => {
    it('keeps a long heading breadcrumb under the cap, deterministically', () => {
        const path = 'docs/guide.md';
        const refs = sectionRefsOf(path, LONG_DOC);
        for (const ref of refs) {
            expect(ref.length).toBeLessThanOrEqual(MAX_SECTION_REF_LEN);
            expect(ref.length).toBeLessThanOrEqual(512);
        }
        // The deepest section's raw slug path alone would blow the cap.
        const rawSlugPath = LONG_CRUMBS.map((c) => c.toLowerCase()).join('/');
        expect(`${srcRef(ROOT, path)}#${rawSlugPath}`.length).toBeGreaterThan(MAX_SECTION_REF_LEN);
        // Deterministic: same input, same refs.
        expect(sectionRefsOf(path, LONG_DOC)).toEqual(refs);
        // The capped ref keeps a readable prefix and ends with the hash discriminator.
        const deepest = refs[refs.length - 1]!;
        expect(deepest.startsWith(`${srcRef(ROOT, path)}#${rawSlugPath.slice(0, 10)}`)).toBe(true);
        expect(deepest).toMatch(/~[0-9a-f]{8}$/);
    });

    it('gives distinct long breadcrumbs sharing a prefix distinct refs', () => {
        const a = sectionRefsOf('a.md', LONG_DOC);
        const variant = [...LONG_CRUMBS.slice(0, -1), 'a-very-different-sixth-level-heading-instead'];
        const b = sectionRefsOf('a.md', variant.map((c, i) => `${'#'.repeat(i + 1)} ${c}\nbody ${i}\n`).join('\n'));
        expect(a).not.toEqual(b);
        expect(new Set(a).size).toBe(a.length);
        expect(new Set(b).size).toBe(b.length);
        for (const ref of [...a, ...b]) expect(ref.length).toBeLessThanOrEqual(MAX_SECTION_REF_LEN);
    });

    it('leaves a normal section ref unchanged', () => {
        const refs = sectionRefsOf('docs/guide.md', '# Guide\nIntro.\n## Setup\nSteps.\n');
        expect(refs).toEqual([
            `${srcRef(ROOT, 'docs/guide.md')}#guide`,
            `${srcRef(ROOT, 'docs/guide.md')}#guide/setup`,
        ]);
        for (const ref of refs) expect(ref.length).toBeLessThanOrEqual(MAX_SECTION_REF_LEN);
    });
});
