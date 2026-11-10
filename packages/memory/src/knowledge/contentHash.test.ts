/**
 * src/knowledge/contentHash.test.ts — the EOL-normalized content identity (PRD 41 §B2 step 0).
 */

import { sha256Hex } from '../retrieval/knowledge-index/sha256.js';
import { BASELINE_CONTENT_HASH_RULE, contentHash, CONTENT_HASH_RULE, contentHashRuleOf } from './contentHash.js';
import { isBinaryBytes, isBinaryText } from './sources.js';

const utf8 = (s: string) => new TextEncoder().encode(s);

describe('contentHash', () => {
    it('names its rule; an absent record reads as the baseline rule', () => {
        expect(CONTENT_HASH_RULE).toBe('eol-lf@1');
        expect(BASELINE_CONTENT_HASH_RULE).toBe('eol-lf@1');
        expect(contentHashRuleOf(undefined)).toBe(BASELINE_CONTENT_HASH_RULE);
        expect(contentHashRuleOf('eol-lf@2')).toBe('eol-lf@2');
    });

    it('CRLF, LF and lone-CR versions of one text hash identically, as the raw sha256 of the LF text', () => {
        const lf = 'line one\nline två\n\nlast';
        const crlf = 'line one\r\nline två\r\n\r\nlast';
        const cr = 'line one\rline två\r\rlast';
        const mixed = 'line one\r\nline två\r\nlast'.replace('två\r\n', 'två\r\r\n'); // CR + CRLF = two line ends
        expect(contentHash(utf8(crlf))).toBe(contentHash(utf8(lf)));
        expect(contentHash(utf8(cr))).toBe(contentHash(utf8(lf)));
        expect(contentHash(utf8(mixed))).toBe(contentHash(utf8(lf)));
        expect(contentHash(utf8(lf))).toBe(sha256Hex(utf8(lf)));
        expect(contentHash(utf8(lf))).toMatch(/^[0-9a-f]{64}$/);
    });

    it('a trailing CR and a trailing CRLF are one LF', () => {
        expect(contentHash(utf8('a\r'))).toBe(sha256Hex('a\n'));
        expect(contentHash(utf8('a\r\n'))).toBe(sha256Hex('a\n'));
        expect(contentHash(utf8('\r\n\r'))).toBe(sha256Hex('\n\n'));
    });

    it('a binary file (a NUL in the first 8 KB) holding CRLF is hashed raw', () => {
        const bytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x0d]);
        expect(contentHash(bytes)).toBe(sha256Hex(bytes));
        expect(contentHash(bytes)).not.toBe(sha256Hex(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0a, 0x0a])));
    });

    it('ONE binary rule: text and bytes agree on where the 8 KB sniff ends (bytes, not UTF-16 units)', () => {
        // 5000 × 'é' is 10000 UTF-8 bytes: the NUL sits past byte 8192 though within the first 8192 code units.
        const late = `${'é'.repeat(5000)}\u0000\r\n`;
        expect(isBinaryText(late)).toBe(false);
        expect(isBinaryBytes(utf8(late))).toBe(false);
        expect(contentHash(utf8(late))).toBe(sha256Hex(utf8(`${'é'.repeat(5000)}\u0000\n`))); // text: its CRLF normalized
        // 4000 × 'é' is 8000 bytes: the NUL is byte 8000, inside the sniff on both sides.
        const early = `${'é'.repeat(4000)}\u0000\r\n`;
        expect(isBinaryText(early)).toBe(true);
        expect(isBinaryBytes(utf8(early))).toBe(true);
        expect(contentHash(utf8(early))).toBe(sha256Hex(utf8(early))); // binary: raw
        // The boundary: a NUL at byte 8191 is in, at byte 8192 is out.
        expect(isBinaryText(`${'a'.repeat(8191)}\u0000`)).toBe(true);
        expect(isBinaryText(`${'a'.repeat(8192)}\u0000`)).toBe(false);
        expect(isBinaryBytes(utf8(`${'a'.repeat(8191)}\u0000`))).toBe(true);
        expect(isBinaryBytes(utf8(`${'a'.repeat(8192)}\u0000`))).toBe(false);
    });

    it('the empty file is the sha256 of nothing', () => {
        expect(contentHash(new Uint8Array(0))).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    });

    it('does not mutate its input', () => {
        const bytes = utf8('a\r\nb\r');
        const copy = bytes.slice();
        contentHash(bytes);
        expect(bytes).toEqual(copy);
    });

    it('hashes a 2 MB CRLF text quickly', () => {
        const line = utf8('const x = 1; // a typical line of source code\r\n');
        const big = new Uint8Array(2 * 1024 * 1024);
        for (let at = 0; at + line.length <= big.length; at += line.length) big.set(line, at);
        big.fill(0x20, big.length - (big.length % line.length)); // pad with spaces, never a NUL
        const started = Date.now();
        contentHash(big);
        expect(Date.now() - started).toBeLessThan(2000);
    });
});
