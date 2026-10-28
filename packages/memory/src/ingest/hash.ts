/**
 * ingest/hash.ts — the content hash every ingest consumer shares (the pipeline's chunk
 * hashes, the knowledge graph's extracted output and card hashes). A leaf module, so
 * `ingest/docsExtract.ts` can use it without an import cycle through the pipeline.
 * (SHA-256 digests live in `retrieval/knowledge-index/sha256.ts`.)
 */

/**
 * FNV-1a, 32-bit, rendered as hex.
 *
 * A content hash here answers "did this chunk change", not "was this chunk
 * tampered with", so a fast non-cryptographic hash is the right tool — and it is
 * synchronous, unlike WebCrypto, which keeps chunking a pure function.
 */
export function hashText(text: string): string {
    let hash = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash.toString(16).padStart(8, '0');
}
