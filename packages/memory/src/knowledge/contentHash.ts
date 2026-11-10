/**
 * knowledge/contentHash.ts — the ONE content identity of a file (PRD 41 §B2 step 0).
 *
 * sha256 (lowercase hex) over the file's bytes, with CRLF and lone CR converted to LF when the
 * bytes are text; binary bytes (a NUL in the first 8 KB, {@link isBinaryBytes}) are hashed raw.
 * So a Windows `core.autocrlf` checkout and a Linux checkout of one commit hash identically, and
 * an LF-only text file hashes exactly as its raw sha256 did before this rule.
 *
 * Byte-level, never decoded: CR (0x0D) and LF (0x0A) cannot occur inside a UTF-8 multibyte
 * sequence, so the rewrite is safe for any UTF-8 text. A file with no CR is hashed in place.
 *
 * The rule is part of the extractor identity: {@link CONTENT_HASH_RULE} is stored in a root's
 * extractor entry (owedFiles.ts), so changing it owes every file one re-extraction.
 */

import { sha256Hex } from '../retrieval/knowledge-index/sha256.js';
import { isBinaryBytes } from './sources.js';

/** Names the normalization {@link contentHash} applies. Change it whenever the hashed bytes of some file would change. */
export const CONTENT_HASH_RULE = 'eol-lf@1';

const CR = 0x0d;
const LF = 0x0a;

/** The bytes with CRLF → LF and lone CR → LF; the input itself when it holds no CR. */
function eolNormalized(bytes: Uint8Array): Uint8Array {
    let i = bytes.indexOf(CR);
    if (i < 0) return bytes;
    const out = new Uint8Array(bytes.length);
    out.set(bytes.subarray(0, i));
    let n = i;
    for (; i < bytes.length; i++) {
        const b = bytes[i]!;
        if (b !== CR) { out[n++] = b; continue; }
        out[n++] = LF;
        if (bytes[i + 1] === LF) i++; // CRLF: the LF is the one just written
    }
    return out.subarray(0, n);
}

/** The content identity of a file: lowercase hex sha256 of its EOL-normalized bytes (raw bytes when binary). */
export function contentHash(bytes: Uint8Array): string {
    return sha256Hex(isBinaryBytes(bytes) ? bytes : eolNormalized(bytes));
}
