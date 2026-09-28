/**
 * Server-Sent-Events `data:` framing.
 *
 * Every consumer of an SSE LLM stream does the same line-level dance: trim the line,
 * keep only `data:` frames, skip empty payloads, `JSON.parse` the rest and skip
 * anything malformed — and stop at the `[DONE]` sentinel, which ends an
 * OpenAI-style stream (anything after it is not part of the response).
 *
 * The line rule is `slice(5).trim()`, which accepts both `data: {…}` and the
 * spaceless `data:{…}` some providers emit — `slice(6)` silently drops the latter.
 *
 * Pure and defensive: a malformed frame is skipped, never thrown.
 */

/**
 * One SSE line's decoded JSON payload, or `undefined` when the line is not a usable
 * data frame (not a `data:` line, the `[DONE]` sentinel, empty, or malformed JSON).
 */
export function parseSseDataLine(line: string): unknown | undefined {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return undefined;
    const data = trimmed.slice(5).trim();
    if (!data || data === '[DONE]') return undefined;
    try {
        return JSON.parse(data);
    } catch {
        return undefined;
    }
}

/** True for the `data: [DONE]` line that ends an OpenAI-style stream. */
export function isSseDoneLine(line: string): boolean {
    const trimmed = line.trim();
    return trimmed.startsWith('data:') && trimmed.slice(5).trim() === '[DONE]';
}

/** Every parsed `data:` frame of a whole, already-buffered SSE body, up to `[DONE]`. */
export function* parseSseDataFrames(raw: string): Generator<unknown> {
    for (const line of raw.split('\n')) {
        if (isSseDoneLine(line)) return;
        const parsed = parseSseDataLine(line);
        if (parsed !== undefined) yield parsed;
    }
}

/**
 * Every parsed `data:` frame of a live SSE byte stream, as it arrives, up to `[DONE]`.
 * A frame split across chunks is held until its line completes; the reader lock is
 * released however iteration ends.
 */
export async function* readSseDataFrames(body: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() as string; // split() always yields ≥1 element
            for (const line of lines) {
                if (isSseDoneLine(line)) return;
                const parsed = parseSseDataLine(line);
                if (parsed !== undefined) yield parsed;
            }
        }
        buffer += decoder.decode();
        const tail = parseSseDataLine(buffer);
        if (tail !== undefined) yield tail;
    } finally {
        reader.releaseLock();
    }
}
