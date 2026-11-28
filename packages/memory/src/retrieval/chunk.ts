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
    // `start` is an offset into the ORIGINAL text, so the leading whitespace `trim` dropped is added back.
    const lead = text.length - text.trimStart().length;
    if (trimmed.length <= chunkSize) return [{ text: trimmed, index: 0, start: lead }];

    const pieces = splitRecursive(trimmed, chunkSize, separators);

    // Merge adjacent pieces up to chunkSize, then stitch overlap between chunks.
    //
    // `buf` is always a contiguous slice of `trimmed` beginning at `bufStart`; its first `carried` code units are
    // the overlap carry, already emitted with the previous chunk. Every emitted chunk therefore has an exact
    // offset, and the carry is strictly shorter than the chunk it came from, so each chunk ends strictly after
    // the previous one: no chunk is ever re-emitted, and the buffer shrinks on every hard-split iteration.
    const chunks: Chunk[] = [];
    let buf = '';
    let bufStart = 0;
    let carried = 0;
    const emit = (raw: string, at: number) => {
        const t = raw.trim();
        if (t.length === 0) return;
        const skipped = raw.length - raw.trimStart().length;
        chunks.push({ text: t, index: chunks.length, start: lead + at + skipped });
    };
    const carryOf = (raw: string) => safeTail(raw, Math.min(overlap, raw.length - 1));

    for (const piece of pieces) {
        if (buf.length + piece.length <= chunkSize) {
            buf += piece;
            continue;
        }
        if (buf.length > carried) emit(buf, bufStart);
        const keep = carryOf(buf);
        bufStart += buf.length - keep.length;
        buf = keep + piece;
        carried = keep.length;
        // A single piece longer than chunkSize is hard-split.
        while (buf.length > chunkSize) {
            let cut = safeCutEnd(buf, 0, chunkSize);
            if (cut <= carried) {
                // The surrogate-safe cut would emit nothing but the carry: drop the carry, which is whole pairs.
                buf = buf.slice(carried);
                bufStart += carried;
                carried = 0;
                cut = safeCutEnd(buf, 0, chunkSize);
            }
            const head = buf.slice(0, cut);
            emit(head, bufStart);
            const next = carryOf(head);
            bufStart += cut - next.length;
            buf = next + buf.slice(cut);
            carried = next.length;
        }
    }
    if (buf.length > carried) emit(buf, bufStart);

    return chunks;
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
