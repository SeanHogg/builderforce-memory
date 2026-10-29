/**
 * ingest/textParsers.ts — the builtin text parsers: markdown, HTML and plain text. Each
 * also carries the knowledge-graph `extract` of the docs pack (`docsExtract.ts`).
 */

import type { MetadataRecord } from '../vectorstore/types.js';
import { extractHtml, extractMarkdown, extractPlainText } from './docsExtract.js';
import { mediaTypeOf, textOf } from './parserSupport.js';
import type { DocumentParser, ParsedSection } from './types.js';

// ── Markdown ──────────────────────────────────────────────────────────────────

export const markdownParser: DocumentParser = {
    id: 'markdown',
    accepts: (source) => {
        const type = mediaTypeOf(source);
        return type.includes('markdown') || type.includes('/md');
    },
    parse: (source) => {
        const lines = textOf(source).split('\n');
        const sections: ParsedSection[] = [];
        // Heading level -> text, so a chunk carries its full breadcrumb rather than
        // only its nearest heading.
        const path: string[] = [];
        let buffer: string[] = [];

        const flush = () => {
            const text = buffer.join('\n').trim();
            buffer = [];
            if (!text) return;
            const metadata: MetadataRecord = {};
            if (path.length) {
                metadata['headingPath'] = path.join(' › ');
                metadata['heading'] = path[path.length - 1] as string;
                metadata['headingDepth'] = path.length;
            }
            sections.push({ text, metadata });
        };

        let inFence = false;
        for (const line of lines) {
            if (/^\s*```/.test(line)) inFence = !inFence;
            const heading = inFence ? null : /^(#{1,6})\s+(.*)$/.exec(line);
            if (heading) {
                flush();
                const depth = (heading[1] as string).length;
                path.length = Math.min(path.length, depth - 1);
                path[depth - 1] = (heading[2] as string).trim();
                path.length = depth;
                continue;
            }
            buffer.push(line);
        }
        flush();
        return sections;
    },
    extract: extractMarkdown,
};

// ── HTML ──────────────────────────────────────────────────────────────────────

export const htmlParser: DocumentParser = {
    id: 'html',
    accepts: (source) => {
        const type = mediaTypeOf(source);
        return type.includes('html') || type.includes('xml');
    },
    parse: (source) => {
        const raw = textOf(source);
        const titleMatch = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(raw);

        const text = raw
            // script/style content is code, not prose — indexing it poisons recall.
            .replace(/<(script|style)[^>]*>[\s\S]*?<\/\1>/gi, ' ')
            .replace(/<!--[\s\S]*?-->/g, ' ')
            .replace(/<\/(p|div|section|article|li|tr|h[1-6])>/gi, '\n')
            .replace(/<br\s*\/?>/gi, '\n')
            .replace(/<[^>]+>/g, ' ')
            .replace(/&nbsp;/g, ' ')
            .replace(/&amp;/g, '&')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&quot;/g, '"')
            .replace(/&#39;/g, "'")
            .replace(/[ \t]+/g, ' ')
            .replace(/\n{3,}/g, '\n\n')
            .trim();

        if (!text) return [];
        const metadata: MetadataRecord = {};
        if (titleMatch?.[1]) metadata['htmlTitle'] = titleMatch[1].trim();
        return [{ text, metadata }];
    },
    extract: extractHtml,
};

// ── plain text ────────────────────────────────────────────────────────────────

export const plainTextParser: DocumentParser = {
    id: 'text',
    accepts: (source) => source.content.kind === 'text',
    parse: (source) => {
        const text = textOf(source).trim();
        return text ? [{ text }] : [];
    },
    extract: extractPlainText,
};
