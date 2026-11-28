/**
 * Document chunking — recursive character text splitter with overlap.
 *
 * The classic RAG ingestion step the memory layer was missing: large documents
 * are split into smaller, semantically-coherent chunks before embedding so that
 * retrieval returns precise passages rather than whole files. Mirrors the
 * behaviour of LangChain's RecursiveCharacterTextSplitter (split on the largest
 * natural boundary that fits, fall back to finer ones) but is zero-dependency.
 */

import { safeCutEnd, safeTail, splitRecursive } from './splitRecursive.js';

export interface ChunkOptions {
    /** Target maximum chunk size in characters. Default 1000. */
    chunkSize?: number;
    /** Characters of overlap carried from the end of one chunk into the next,
     *  preserving context across boundaries. Default 200. Clamped below chunkSize. */
    chunkOverlap?: number;
    /** Separators tried in order, largest natural boundary first. */
    separators?: string[];
    /** Lossless packing: nothing trimmed, no overlap (`chunkOverlap` is ignored), `start` is the true offset, so
     *  the chunk texts concatenate back to the input (a trailing whitespace-only remainder is dropped). Default false. */
    exact?: boolean;
}

export interface Chunk {
    /** The chunk text. */
    text: string;
    /** 0-based ordinal of this chunk within its source document. */
    index: number;
    /** Character offset of this chunk's start within the original document. */
    start: number;
}

const DEFAULT_SEPARATORS = ['\n\n', '\n', '. ', ' ', ''];

/**
 * Splits `text` into overlapping chunks no larger than `chunkSize` characters,
 * preferring the largest natural separator that keeps a piece under the limit.
 * Returns `[]` for empty/whitespace input. Deterministic and pure.
 */
export function chunkText(text: string, opts: ChunkOptions = {}): Chunk[] {
    const chunkSize = Math.max(1, opts.chunkSize ?? 1000);
    const overlap = Math.min(Math.max(0, opts.chunkOverlap ?? 200), chunkSize - 1);
    const separators = opts.separators ?? DEFAULT_SEPARATORS;

    if (opts.exact) return chunkExact(text, chunkSize, separators);

    const trimmed = text.trim();
    if (trimmed.length === 0) return [];
    if (trimmed.length <= chunkSize) return [{ text: trimmed, index: 0, start: 0 }];

    const pieces = splitRecursive(trimmed, chunkSize, separators);

    // Merge adjacent pieces up to chunkSize, then stitch overlap between chunks.
    const chunks: Chunk[] = [];
    let buf = '';
    const flush = () => {
        const t = buf.trim();
        if (t.length > 0) {
            const start = chunks.length === 0 ? 0 : Math.max(0, trimmed.indexOf(t));
            chunks.push({ text: t, index: chunks.length, start });
        }
        buf = '';
    };

    for (const piece of pieces) {
        if (buf.length + piece.length <= chunkSize) {
            buf += piece;
        } else {
            flush();
            // Carry overlap from the previous chunk's tail.
            const prev = chunks[chunks.length - 1]?.text ?? '';
            buf = safeTail(prev, overlap) + piece;
            // A single piece longer than chunkSize is hard-split.
            while (buf.length > chunkSize) {
                const cut = safeCutEnd(buf, 0, chunkSize);
                const head = buf.slice(0, cut);
                chunks.push({ text: head.trim(), index: chunks.length, start: 0 });
                buf = safeTail(head, overlap) + buf.slice(cut);
            }
        }
    }
    flush();

    return chunks.map((c, i) => ({ ...c, index: i }));
}

/**
 * Exact mode: the SAME recursive split, packed greedily with nothing trimmed and no overlap, so slicing the input by
 * each chunk's `start`/length (or joining the chunk texts) reproduces it. Only a trailing whitespace-only remainder
 * is dropped.
 */
function chunkExact(text: string, chunkSize: number, separators: string[]): Chunk[] {
    if (text.trim().length === 0) return [];
    const chunks: Chunk[] = [];
    let current = '';
    let start = 0;
    const push = () => {
        chunks.push({ text: current, index: chunks.length, start });
        start += current.length;
        current = '';
    };
    for (const piece of splitRecursive(text, chunkSize, separators)) {
        if (current && current.length + piece.length > chunkSize) push();
        current += piece;
    }
    if (current.trim().length > 0) push();
    return chunks;
}
