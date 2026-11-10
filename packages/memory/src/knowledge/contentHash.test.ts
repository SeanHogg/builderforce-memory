/**
 * src/knowledge/contentHash.test.ts — the EOL-normalized content identity (PRD 41 §B2 step 0).
 */

import { sha256Hex } from '../retrieval/knowledge-index/sha256.js';
import { contentHash, CONTENT_HASH_RULE } from './contentHash.js';

const utf8 = (s: string) => new TextEncoder().encode(s);

describe('contentHash', () => {
    it('names its rule', () => {
        expect(CONTENT_HASH_RULE).toBe('eol-lf@1');
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
