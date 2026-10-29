/**
 * ingest/parserSupport.ts — what every builtin parser shares: reading a source's text and
 * media type, and telling a binary document from text.
 */

import type { SourceDocument } from './types.js';

export const textOf = (source: SourceDocument): string =>
    source.content.kind === 'text' ? source.content.text : '';

export const mediaTypeOf = (source: SourceDocument): string =>
    source.content.kind === 'text' ? (source.content.mediaType ?? 'text/plain').toLowerCase() : '';

/**
 * Binary documents — pdf, legacy and OOXML/OpenDocument office files. Their bytes are not
 * text. The ingest pipeline's builtin parsers are unchanged (the html/xml and plain-text
 * matchers still accept these types); the knowledge composition (`composePacks`) is what
 * makes the builtin docs parsers refuse them, so only a host extractor that turns them into
 * text claims them there, in any pack order.
 */
export function isBinaryDocumentType(mediaType: string): boolean {
    return /^application\/(?:pdf|msword|vnd\.(?:openxmlformats-officedocument\.|ms-|oasis\.opendocument\.))/
        .test(mediaType.trim().toLowerCase());
}
