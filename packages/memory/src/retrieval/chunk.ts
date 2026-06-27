/**
 * Document chunking — recursive character text splitter with overlap.
 *
 * The classic RAG ingestion step the memory layer was missing: large documents
 * are split into smaller, semantically-coherent chunks before embedding so that
 * retrieval returns precise passages rather than whole files. Mirrors the
 * behaviour of LangChain's RecursiveCharacterTextSplitter (split on the largest
 * natural boundary that fits, fall back to finer ones) but is zero-dependency.
 */

export interface ChunkOptions {
    /** Target maximum chunk size in characters. Default 1000. */
    chunkSize?: number;
    /** Characters of overlap carried from the end of one chunk into the next,
     *  preserving context across boundaries. Default 200. Clamped below chunkSize. */
    chunkOverlap?: number;
    /** Separators tried in order, largest natural boundary first. */
    separators?: string[];
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
            buf = (overlap > 0 ? prev.slice(-overlap) : '') + piece;
            // A single piece longer than chunkSize is hard-split.
            while (buf.length > chunkSize) {
                const head = buf.slice(0, chunkSize);
                chunks.push({ text: head.trim(), index: chunks.length, start: 0 });
                buf = (overlap > 0 ? head.slice(-overlap) : '') + buf.slice(chunkSize);
            }
        }
    }
    flush();

    return chunks.map((c, i) => ({ ...c, index: i }));
}

/** Recursively splits text on the first separator that yields sub-chunkSize pieces. */
function splitRecursive(text: string, chunkSize: number, separators: string[]): string[] {
    /* istanbul ignore next -- defensive base case; callers only recurse on parts > chunkSize */
    if (text.length <= chunkSize) return [text];
    const [sep, ...rest] = separators;
    if (sep === undefined) return [text];
    if (sep === '') {
        // Last resort: hard character split.
        const out: string[] = [];
        for (let i = 0; i < text.length; i += chunkSize) out.push(text.slice(i, i + chunkSize));
        return out;
    }
    const parts = text.split(sep);
    const out: string[] = [];
    for (let i = 0; i < parts.length; i++) {
        const part = i < parts.length - 1 ? parts[i]! + sep : parts[i]!;
        if (part.length > chunkSize) out.push(...splitRecursive(part, chunkSize, rest));
        else out.push(part);
    }
    return out;
}
