/**
 * The recursive separator split the chunker is built on (chunk.ts). Separators are KEPT on the piece they end,
 * so joining the pieces gives back the input; a hard cut never separates the halves of a UTF-16 surrogate pair.
 */

/**
 * The end index of a slice of `text` starting at `at` and at most `max` long, moved off the middle of a surrogate
 * pair: a high surrogate at the cut goes with the low one after it (cut before it, or keep the pair whole when it
 * is the whole slice, the one size exception at `max` 1).
 */
export function safeCutEnd(text: string, at: number, max: number): number {
    const end = Math.min(text.length, at + max);
    if (end >= text.length) return end;
    const last = text.charCodeAt(end - 1);
    if (last >= 0xd800 && last <= 0xdbff) return end - 1 > at ? end - 1 : end + 1;
    return end;
}

/** The last `n` code units of `text` as an overlap carry; one fewer when that would start on a low surrogate. */
export function safeTail(text: string, n: number): string {
    if (n <= 0) return '';
    const tail = text.slice(-n);
    const first = tail.charCodeAt(0);
    return first >= 0xdc00 && first <= 0xdfff ? tail.slice(1) : tail;
}

/** Recursively splits text on the first separator that yields sub-`chunkSize` pieces. */
export function splitRecursive(text: string, chunkSize: number, separators: string[]): string[] {
    /* istanbul ignore next -- defensive base case; callers only recurse on parts > chunkSize */
    if (text.length <= chunkSize) return [text];
    const [sep, ...rest] = separators;
    if (sep === undefined) return [text];
    if (sep === '') {
        // Last resort: hard character split, surrogate-pair safe.
        const out: string[] = [];
        for (let at = 0; at < text.length;) {
            const end = safeCutEnd(text, at, chunkSize);
            out.push(text.slice(at, end));
            at = end;
        }
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
