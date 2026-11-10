/**
 * src/retrieval/knowledge-index/sha256.test.ts — the portable SHA-256 equals node:crypto for every padding
 * case (input lengths 0..200 cover a message ending before, at and after the 56-byte length boundary), and
 * incremental updates equal one update.
 */

import { createHash } from 'node:crypto';
import { Sha256, sha256Hex } from './sha256.js';

const bytesOf = (n: number) => Uint8Array.from({ length: n }, (_, i) => (i * 31 + 7) & 0xff);
const reference = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');

describe('sha256', () => {
    it('matches node:crypto for every length 0..200', () => {
        for (let n = 0; n <= 200; n++) expect({ n, h: sha256Hex(bytesOf(n)) }).toEqual({ n, h: reference(bytesOf(n)) });
    });

    it('incremental updates equal one update', () => {
        const b = bytesOf(300);
        const h = new Sha256();
        for (let i = 0; i < b.length; i += 7) h.update(b.subarray(i, i + 7));
        expect(h.hex()).toBe(reference(b));
        expect(sha256Hex('héllo')).toBe(createHash('sha256').update('héllo', 'utf8').digest('hex'));
    });
});
