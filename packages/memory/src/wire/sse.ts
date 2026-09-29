/**
 * Server-Sent-Events `data:` framing.
 *
 * Every consumer of an SSE LLM stream does the same line-level dance: trim the line,
 * keep only `data:` frames, and stop at the `[DONE]` sentinel, which ends an
 * OpenAI-style stream (anything after it is not part of the response). JSON streams
 * then parse each payload and skip empty or malformed ones.
 *
 * The line rule is `slice(5).trim()`, which accepts both `data: {…}` and the
 * spaceless `data:{…}` some providers emit — `slice(6)` silently drops the latter.
 *
 * Two levels, one rule: {@link readSseDataPayloads} yields payload TEXT (for callers
 * whose frames are not all JSON, or that parse per wire shape), and
 * {@link readSseDataFrames} yields parsed JSON on top of it.
 *
 * Pure and defensive: a malformed frame is skipped, never thrown.
 */

/** The payload text of a `data:` line, or `undefined` for any other line. */
export function sseDataPayload(line: string): string | undefined {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return undefined;
    return trimmed.slice(5).trim();
}

/** True for the `data: [DONE]` line that ends an OpenAI-style stream. */
export function isSseDoneLine(line: string): boolean {
    return sseDataPayload(line) === '[DONE]';
}

/**
 * One SSE line's decoded JSON payload, or `undefined` when the line is not a usable
 * data frame (not a `data:` line, the `[DONE]` sentinel, empty, or malformed JSON).
 */
export function parseSseDataLine(line: string): unknown | undefined {
    const data = sseDataPayload(line);
    if (!data || data === '[DONE]') return undefined;
    try {
        return JSON.parse(data);
    } catch {
        return undefined;
    }
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
 * Every `data:` payload of a live SSE byte stream, as text, as it arrives, up to
 * `[DONE]`. A line split across chunks is held until its newline arrives; the reader
 * lock is released however iteration ends. A missing body yields nothing.
 */
export async function* readSseDataPayloads(body: ReadableStream<Uint8Array> | null | undefined): AsyncGenerator<string> {
    if (!body) return;
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
                const data = sseDataPayload(line);
                if (data === undefined) continue;
                if (data === '[DONE]') return;
                yield data;
            }
        }
        buffer += decoder.decode();
        const tail = sseDataPayload(buffer);
        if (tail !== undefined && tail !== '[DONE]') yield tail;
    } finally {
        reader.releaseLock();
    }
}

/** Every parsed JSON `data:` frame of a live SSE byte stream, up to `[DONE]`. */
export async function* readSseDataFrames(body: ReadableStream<Uint8Array> | null | undefined): AsyncGenerator<unknown> {
    for await (const data of readSseDataPayloads(body)) {
        if (!data) continue;
        try {
            yield JSON.parse(data);
        } catch {
            // Skip malformed SSE frames rather than failing the stream.
        }
    }
}
