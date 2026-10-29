/**
 * ingest/docsExtract.ts — the knowledge-graph `extract` of the builtin text parsers
 * (markdown, HTML, plain text). PRD 38 §4.5, the docs pack.
 *
 * Pure: a function of the source text, its root-relative path (`source.id`), the parsed
 * sections and the extract context. Emits, all with `origin: 'extracted'`:
 *
 *   • a `file` node, and a `section` node per heading breadcrumb, joined by `contains`;
 *   • `links` edges for relative markdown links (inline and reference-style), `[[wiki]]`
 *     links (by name anywhere in the root through `ctx.resolveName`, else beside the linking
 *     file) and HTML hrefs, resolved through `ctx.resolvePath`. A missing target — or one
 *     outside the root — stays an edge, so the graph can report it broken;
 *   • text before the first markdown heading (or a file with none) is a `section` too, so
 *     it is judged and carded like any other;
 *   • `mentions` edges where a section's normalized text contains a known entity key;
 *   • keys: path, basename, heading slugs and front-matter values.
 *
 * A section node carries `badges.section` (its index in the parser's output) so the review
 * can hand exactly that text to judgement, and `badges.excerpt` for the index card.
 */

import type { ExtractContext, Extracted, KEdge, KNode, NodeRef } from '../knowledge/model.js';
import { basenameOf, compareStr, normalizeKey, slugOf, srcRef } from '../knowledge/refs.js';
import { hashText } from './hash.js';
import { CONTENT_HASH_FIELD, type ParsedSection, type SourceDocument } from './types.js';

export const DOCS_PACK_ID = 'docs';
/** Badge: index of the section in the parser's output. */
export const SECTION_INDEX_BADGE = 'section';
const EXCERPT_CHARS = 400;
const BREADCRUMB_SEPARATOR = ' › ';

type Ctx = ExtractContext;

const textOf = (source: SourceDocument): string => (source.content.kind === 'text' ? source.content.text : '');

class ExtractBuilder {
    readonly path: string;
    readonly hash: string;
    readonly fileRef: NodeRef;
    private readonly nodes = new Map<NodeRef, KNode>();
    private readonly edges = new Map<string, KEdge>();
    private readonly keys = new Map<string, { ref: NodeRef; key: string }>();

    constructor(source: SourceDocument, private readonly ctx: Ctx, label?: string) {
        this.path = source.id;
        const stated = source.metadata?.[CONTENT_HASH_FIELD];
        this.hash = typeof stated === 'string' ? stated : hashText(textOf(source));
        this.fileRef = srcRef(ctx.rootId, this.path);
        const base = basenameOf(this.path);
        this.node(this.fileRef, 'file', label || base);
        this.key(this.fileRef, this.path);
        this.key(this.fileRef, base.toLowerCase());
        this.key(this.fileRef, normalizeKey(base.replace(/\.[^.]+$/, '')));
    }

    node(ref: NodeRef, kind: string, label: string, badges?: Record<string, number | string>): KNode {
        const node: KNode = {
            ref, kind, label, pack: DOCS_PACK_ID, importance: 0, origin: 'extracted', sourceHash: this.hash,
            ...(badges ? { badges } : {}),
        };
        this.nodes.set(ref, node);
        return node;
    }

    edge(from: NodeRef, to: NodeRef, kind: string): void {
        if (from === to) return;
        const id = `${from}\u0000${to}\u0000${kind}`;
        const prior = this.edges.get(id);
        if (prior) prior.weight += 1;
        else this.edges.set(id, { from, to, kind, weight: 1, origin: 'extracted' });
    }

    key(ref: NodeRef, key: string): void {
        if (key) this.keys.set(`${ref}\u0000${key}`, { ref, key });
    }

    /** `links` edges to each resolvable target. */
    link(from: NodeRef, target: string): void {
        const clean = target.split('#')[0]!.split('?')[0]!.trim();
        if (!clean || /^[a-z][a-z0-9+.-]*:/i.test(clean)) return;
        let decoded = clean;
        try { decoded = decodeURI(clean); } catch { /* keep the raw target */ }
        const resolved = this.ctx.resolvePath(this.path, decoded);
        // Outside the root: kept under its written path, so it surfaces as a broken link.
        const target = resolved ?? decoded.replace(/^\.\//, '');
        if (target !== this.path) this.edge(from, srcRef(this.ctx.rootId, target), 'links');
    }

    wikiLinks(from: NodeRef, text: string): void {
        for (const m of text.matchAll(/\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]/g)) {
            const name = m[1]!.trim();
            if (!name) continue;
            const byName = name.includes('/') ? undefined : this.ctx.resolveName?.(name);
            if (byName !== undefined) { if (byName !== this.path) this.edge(from, srcRef(this.ctx.rootId, byName), 'links'); continue; }
            this.link(from, /\.[A-Za-z][A-Za-z0-9]{0,7}$/.test(name) ? name : `${name}.md`);
        }
    }

    /** `mentions` edges where the normalized text contains a known key as whole words. */
    mentions(from: NodeRef, text: string): void {
        if (this.ctx.knownKeys.size === 0) return;
        const padded = ` ${normalizeKey(text)} `;
        const known = [...this.ctx.knownKeys.entries()].sort((a, b) => compareStr(a[0], b[0]));
        for (const [key, ref] of known) {
            if (key && padded.includes(` ${key} `)) this.edge(from, ref, 'mentions');
        }
    }

    /** A whole-body `section` (HTML, plain text): judged like a markdown section. */
    bodySection(index: number, text: string): NodeRef {
        const ref: NodeRef = `${this.fileRef}#${index === 0 ? 'body' : `body-${index + 1}`}`;
        this.node(ref, 'section', basenameOf(this.path), { [SECTION_INDEX_BADGE]: index, excerpt: excerpt(text) });
        this.edge(this.fileRef, ref, 'contains');
        return ref;
    }

    build(): Extracted {
        const byRef = <T extends { ref: NodeRef }>(a: T, b: T) => compareStr(a.ref, b.ref);
        return {
            sourceRef: this.fileRef,
            sourceHash: this.hash,
            nodes: [...this.nodes.values()].sort(byRef),
            edges: [...this.edges.values()].sort((a, b) =>
                compareStr(a.from, b.from) || compareStr(a.to, b.to) || compareStr(a.kind, b.kind)),
            keys: [...this.keys.values()].sort((a, b) => byRef(a, b) || compareStr(a.key, b.key)),
        };
    }
}

/** Leading `---` fenced front matter; group 1 is its body. */
const FRONT_MATTER = /^---\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|$)/;

/** `key: value` and `- item` front matter between leading `---` fences. */
export function frontMatterOf(text: string): Record<string, string[]> {
    const out: Record<string, string[]> = {};
    const m = FRONT_MATTER.exec(text);
    if (!m) return out;
    let current: string | undefined;
    for (const line of m[1]!.split(/\r?\n/)) {
        const item = /^\s+-\s+(.*)$/.exec(line);
        if (item && current) { out[current]!.push(unquote(item[1]!)); continue; }
        const kv = /^([A-Za-z0-9_.-]+)\s*:\s*(.*)$/.exec(line);
        if (!kv) continue;
        current = kv[1]!;
        const raw = kv[2]!.trim();
        const list = /^\[(.*)\]$/.exec(raw);
        out[current] = list ? list[1]!.split(',').map(unquote).filter(Boolean) : raw ? [unquote(raw)] : [];
    }
    return out;
}

function unquote(s: string): string {
    return s.trim().replace(/^["']|["']$/g, '').trim();
}

function excerpt(text: string): string {
    return text.replace(/\s+/g, ' ').trim().slice(0, EXCERPT_CHARS);
}

const MD_LINK = /(?<!!)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)/g;
/** `[id]: target` definitions (reference-style links); the use is `[text][id]` or `[id][]`. */
const MD_REF_DEF = /^ {0,3}\[([^\]]+)\]:\s*<?(\S+?)>?(?:\s+["'(].*)?$/gm;
const MD_REF_USE = /(?<!!)\[([^\]]*)\]\[([^\]]*)\]/g;

/** Reference-link definitions of a whole markdown text, by lower-cased label. */
function refDefinitions(text: string): Map<string, string> {
    const out = new Map<string, string>();
    for (const m of text.matchAll(MD_REF_DEF)) {
        const label = m[1]!.trim().toLowerCase();
        if (!out.has(label)) out.set(label, m[2]!);
    }
    return out;
}

export function extractMarkdown(source: SourceDocument, sections: ParsedSection[], ctx: Ctx): Extracted {
    const b = new ExtractBuilder(source, ctx);
    const fm = frontMatterOf(textOf(source));
    const fileBadges: Record<string, string> = {};
    for (const k of Object.keys(fm).sort(compareStr)) {
        fileBadges[`fm.${k}`] = fm[k]!.join(', ');
        for (const v of fm[k]!) b.key(b.fileRef, normalizeKey(v));
    }
    if (Object.keys(fileBadges).length) b.node(b.fileRef, 'file', fm['title']?.[0] || basenameOf(b.path), fileBadges);

    const refByCrumb = new Map<string, NodeRef>();
    const occurrences = new Map<string, number>();
    const sectionNode = (crumbs: string[], index?: number, text?: string): NodeRef => {
        const slugPath = crumbs.map(slugOf).join('/');
        const n = (occurrences.get(slugPath) ?? 0) + 1;
        occurrences.set(slugPath, n);
        const ref: NodeRef = `${b.fileRef}#${n > 1 ? `${slugPath}-${n}` : slugPath}`;
        const badges: Record<string, number | string> = { depth: crumbs.length, breadcrumb: crumbs.join(BREADCRUMB_SEPARATOR) };
        if (index !== undefined) badges[SECTION_INDEX_BADGE] = index;
        if (text) badges['excerpt'] = excerpt(text);
        b.node(ref, 'section', crumbs[crumbs.length - 1]!, badges);
        b.key(ref, slugOf(crumbs[crumbs.length - 1]!));
        const parent = crumbs.length > 1 ? refByCrumb.get(crumbs.slice(0, -1).join('\u0000')) : b.fileRef;
        b.edge(parent ?? b.fileRef, ref, 'contains');
        refByCrumb.set(crumbs.join('\u0000'), ref);
        return ref;
    };

    const refs = refDefinitions(textOf(source));
    sections.forEach((section, index) => {
        const headingPath = section.metadata?.['headingPath'];
        let from: NodeRef;
        if (typeof headingPath === 'string' && headingPath) {
            const crumbs = headingPath.split(BREADCRUMB_SEPARATOR);
            // Headings with no body of their own are not emitted by the parser; recover them.
            for (let k = 1; k < crumbs.length; k++) {
                if (!refByCrumb.has(crumbs.slice(0, k).join('\u0000'))) sectionNode(crumbs.slice(0, k));
            }
            from = sectionNode(crumbs, index, section.text);
        } else {
            // Text before the first heading (or a file with none) is judged like a section;
            // a preamble that is only front matter is the file's, not a section.
            const body = section.text.replace(FRONT_MATTER, '').trim();
            from = body ? b.bodySection(index, body) : b.fileRef;
        }
        for (const m of section.text.matchAll(MD_LINK)) {
            if (!m[1]!.startsWith('#')) b.link(from, m[1]!);
        }
        for (const m of section.text.matchAll(MD_REF_USE)) {
            const target = refs.get((m[2] || m[1] || '').trim().toLowerCase());
            if (target && !target.startsWith('#')) b.link(from, target);
        }
        b.wikiLinks(from, section.text);
        b.mentions(from, section.text);
    });
    return b.build();
}

export function extractHtml(source: SourceDocument, sections: ParsedSection[], ctx: Ctx): Extracted {
    const title = sections[0]?.metadata?.['htmlTitle'];
    const b = new ExtractBuilder(source, ctx, typeof title === 'string' ? title : undefined);
    if (typeof title === 'string' && title) b.key(b.fileRef, normalizeKey(title));
    for (const m of textOf(source).matchAll(/<a\s[^>]*href\s*=\s*["']([^"']+)["']/gi)) {
        if (!m[1]!.startsWith('#')) b.link(b.fileRef, m[1]!);
    }
    sections.forEach((section, index) => b.mentions(b.bodySection(index, section.text), section.text));
    return b.build();
}

export function extractPlainText(source: SourceDocument, sections: ParsedSection[], ctx: Ctx): Extracted {
    const b = new ExtractBuilder(source, ctx);
    sections.forEach((section, index) => {
        const from = b.bodySection(index, section.text);
        b.wikiLinks(from, section.text);
        b.mentions(from, section.text);
    });
    return b.build();
}
