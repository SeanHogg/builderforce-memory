/**
 * knowledge/pack/deflate.ts — `deflate-raw` through `CompressionStream` / `DecompressionStream`, which
 * Node 18+, Cloudflare Workers and browsers all have (PRD 41 A11). Inflation is CAPPED: a stream that
 * produces more than `maxBytes` is cancelled at once, so a bomb never holds more than the cap.
 */

import { PackTooLargeError } from './types.js';

async function drain(stream: ReadableStream<Uint8Array>, maxBytes: number): Promise<Uint8Array> {
    const reader = stream.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        total += value.length;
        if (total > maxBytes) { await reader.cancel().catch(() => undefined); throw new PackTooLargeError(maxBytes); }
        chunks.push(value);
    }
    const out = new Uint8Array(total);
    let at = 0;
    for (const c of chunks) { out.set(c, at); at += c.length; }
    return out;
}

async function through(data: Uint8Array, transform: { readable: ReadableStream<Uint8Array>; writable: WritableStream<BufferSource> }, maxBytes: number): Promise<Uint8Array> {
    const writer = transform.writable.getWriter();
    // Written without waiting: the reader below must drain it, or a large input would stall on backpressure.
    const wrote = writer.write(data as unknown as BufferSource).then(() => writer.close());
    wrote.catch(() => undefined);
    try {
        return await drain(transform.readable, maxBytes);
    } finally {
        await wrote.catch(() => undefined);
    }
}

/** Writes `chunks` into one transform in order, draining as it goes: the parts of a split pack are inflated as ONE stream, never joined. */
async function throughChunks(chunks: readonly Uint8Array[], transform: { readable: ReadableStream<Uint8Array>; writable: WritableStream<BufferSource> }, maxBytes: number): Promise<Uint8Array> {
    const writer = transform.writable.getWriter();
    const wrote = (async () => {
        for (const c of chunks) await writer.write(c as unknown as BufferSource);
        await writer.close();
    })();
    wrote.catch(() => undefined);
    try {
        return await drain(transform.readable, maxBytes);
    } finally {
        await wrote.catch(() => undefined);
    }
}

export const inflateRawChunks = (chunks: readonly Uint8Array[], maxBytes: number): Promise<Uint8Array> =>
    throughChunks(chunks, new DecompressionStream('deflate-raw') as never, maxBytes);

export const deflateRaw =(data: Uint8Array): Promise<Uint8Array> =>
    through(data, new CompressionStream('deflate-raw') as never, Number.MAX_SAFE_INTEGER);

/** Throws {@link PackTooLargeError} past `maxBytes` of output, and `Error` on a corrupt stream. */
export const inflateRaw = (data: Uint8Array, maxBytes: number): Promise<Uint8Array> =>
    through(data, new DecompressionStream('deflate-raw') as never, maxBytes);
