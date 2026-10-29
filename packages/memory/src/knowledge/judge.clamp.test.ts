/**
 * src/knowledge/judge.clamp.test.ts
 * `clampClaim`: the privacy bound on a stored claim, stated once here so a host (the
 * Builderforce API) applies memory's own rules instead of re-declaring the separator and limits.
 * Imported through the subpath's index, so the export surface is under test too.
 */

import { claimContent, claimTextOf } from './judge.js';
import { clampClaim, clampQuote, MAX_CLAIM_TEXT_CHARS, MAX_QUOTE_CHARS, MAX_QUOTE_LINES } from './index.js';

describe('clampClaim', () => {
    it('bounds a claim text at three 200-character terms and their two joining spaces', () => {
        expect(MAX_CLAIM_TEXT_CHARS).toBe(3 * 200 + 2);
        expect(clampClaim('x'.repeat(20_000))).toHaveLength(MAX_CLAIM_TEXT_CHARS);
    });

    it('keeps the lines of a claim text (its terms may span lines) and only bounds its characters', () => {
        const text = Array.from({ length: MAX_QUOTE_LINES + 4 }, (_, i) => `line ${i}`).join('\n');
        expect(clampClaim(text)).toBe(text);
        expect(clampClaim(claimContent(text, 'it says so'))).toBe(claimContent(text, 'it says so'));
    });

    it('lets what the judge produces through unchanged, at its longest', () => {
        const text = [200, 200, 200].map((n) => 'w'.repeat(n)).join(' ');
        expect(text).toHaveLength(MAX_CLAIM_TEXT_CHARS);
        const content = claimContent(text, 'q'.repeat(MAX_QUOTE_CHARS));
        expect(clampClaim(content)).toBe(content);
        expect(clampClaim('a says b')).toBe('a says b');
        expect(clampClaim(claimContent('a says b', 'it says so'))).toBe(claimContent('a says b', 'it says so'));
    });

    it('clamps the quote by the quote bound without cutting the claim text', () => {
        const clamped = clampClaim(`a says b — “${'q'.repeat(20_000)}”`);
        expect(claimTextOf(clamped)).toBe('a says b');
        expect(clamped).toBe(claimContent('a says b', 'q'.repeat(MAX_QUOTE_CHARS)));
        const lines = clampClaim(`a says b — “${Array.from({ length: 9 }, (_, i) => `l${i}`).join('\n')}”`);
        expect(lines).toBe(claimContent('a says b', 'l0\nl1\nl2'));
    });

    it('bounds a long text and its quote independently, and is idempotent', () => {
        const clamped = clampClaim(`${'t'.repeat(5_000)} — “short quote”`);
        expect(claimTextOf(clamped)).toHaveLength(MAX_CLAIM_TEXT_CHARS);
        expect(clamped.endsWith(' — “short quote”')).toBe(true);
        expect(clampClaim(clamped)).toBe(clamped);
    });

    it('is idempotent where the quote cut lands on a space or a line break', () => {
        for (const gap of [' ', '\n']) {
            const once = clampClaim(`a says b — “${'q'.repeat(MAX_QUOTE_CHARS - 1)}${gap}${'r'.repeat(50)}”`);
            expect(once).toBe(claimContent('a says b', 'q'.repeat(MAX_QUOTE_CHARS - 1))); // the cut gap is trimmed, not kept
            expect(clampClaim(once)).toBe(once);
        }
        const quote = 'q'.repeat(MAX_QUOTE_CHARS - 1);
        expect(clampQuote(`${quote} ${'r'.repeat(50)}`)).toBe(quote);
        expect(clampQuote(clampQuote(`${quote}\n${'r'.repeat(50)}`))).toBe(quote);
    });

    it('drops an empty quote to the canonical form', () => {
        expect(clampClaim('a says b — “”')).toBe('a says b');
    });
});
