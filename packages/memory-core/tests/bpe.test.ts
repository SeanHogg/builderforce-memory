/**
 * tests/bpe.test.ts
 * Unit tests for the BPE tokenizer (no GPU required).
 */

import { BPETokenizer } from '../src/tokenizer/bpe';

// ── Minimal synthetic vocabulary for testing ──────────────────────────────────

function buildTinyTokenizer() {
    const tokenizer = new BPETokenizer();

    // Build a minimal vocab that covers ASCII printable characters + a few words
    // The byte encoder maps bytes 33–126 (! to ~) to themselves,
    // so single-char printable ASCII tokens are their own strings.

    const vocab: Record<string, number> = {};
    // Add byte-level tokens (printable ASCII 33–126 → same character)
    for (let b = 33; b <= 126; b++) {
        vocab[String.fromCodePoint(b)] = b - 33;
    }
    // Add a few merged tokens
    vocab['he'] = 100;
    vocab['ll'] = 101;
    vocab['hel'] = 102;
    vocab['hell'] = 103;
    vocab['hello'] = 104;

    // Space is in the non-printable range and gets mapped to Unicode 256+
    // byte 32 (space) → code point 256 + offset
    // Simplify: add it directly
    vocab['\u0100'] = 200;  // byte 32 (space) → code point 256

    const merges = [
        'h e',
        'l l',
        'h e l',  // will be skipped since 'he' + 'l' isn't a valid pair key
        'he l',
        'hel l',
        'hell o',
    ];

    tokenizer.loadFromObjects(vocab, merges);
    return tokenizer;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

test('BPETokenizer loads vocab from objects', () => {
    const t = buildTinyTokenizer();
    expect(t.vocabSize).toBeGreaterThan(0);
});

test('encode returns an array of numbers', () => {
    const t = buildTinyTokenizer();
    const ids = t.encode('hello');
    expect(Array.isArray(ids)).toBe(true);
    for (const id of ids) {
        expect(typeof id).toBe('number');
    }
});

test('decode inverts encode for ASCII tokens', () => {
    const t   = buildTinyTokenizer();
    // Use characters that are in the byte-encoder passthrough range (33–126)
    const text = 'abc';
    const ids  = t.encode(text);
    const back = t.decode(ids);
    // Should recover original string (may differ in whitespace handling)
    expect(back).toBe(text);
});

test('padOrTruncate truncates long sequences', () => {
    const t   = buildTinyTokenizer();
    const ids = [1, 2, 3, 4, 5];
    expect(t.padOrTruncate(ids, 3)).toEqual([1, 2, 3]);
});

test('padOrTruncate pads short sequences (right)', () => {
    const t   = buildTinyTokenizer();
    const ids = [1, 2];
    const out = t.padOrTruncate(ids, 5, 'right');
    expect(out.length).toBe(5);
    expect(out.slice(0, 2)).toEqual([1, 2]);
    // Padded with padId (defaults to 0 if not set)
    expect(out[2]).toBe(0);
});

test('padOrTruncate pads short sequences (left)', () => {
    const t   = buildTinyTokenizer();
    const ids = [1, 2];
    const out = t.padOrTruncate(ids, 4, 'left');
    expect(out.length).toBe(4);
    expect(out.slice(2)).toEqual([1, 2]);
});

test('encode handles empty string', () => {
    const t   = buildTinyTokenizer();
    const ids = t.encode('');
    expect(Array.isArray(ids)).toBe(true);
    expect(ids.length).toBe(0);
});

test('vocabSize reflects loaded vocabulary', () => {
    const t   = buildTinyTokenizer();
    expect(t.vocabSize).toBeGreaterThan(90);  // ASCII 33–126 = 94 entries + extras
});

test('_bpe returns single token for known word', () => {
    const t = new BPETokenizer();
    t.loadFromObjects({ 'hello': 0 }, []);
    expect(t._bpe('hello')).toEqual(['hello']);
});

test('_bpe splits unknown word into characters', () => {
    const t = new BPETokenizer();
    t.loadFromObjects({}, []);
    const result = t._bpe('abc');
    expect(result).toEqual(['a', 'b', 'c']);
});

test('_bpe applies merges correctly', () => {
    const t = new BPETokenizer();
    t.loadFromObjects({ 'ab': 0, 'c': 1 }, ['a b']);
    const result = t._bpe('abc');
    expect(result).toEqual(['ab', 'c']);
});

test('_bpe applies multiple merges in rank order', () => {
    const t = new BPETokenizer();
    // rank 0: 'a b' applied first, then rank 1: 'ab c'
    t.loadFromObjects({ 'ab': 0, 'abc': 1, 'c': 2 }, ['a b', 'ab c']);
    const result = t._bpe('abc');
    expect(result).toEqual(['abc']);
});

test('_bpe returns single-char symbols when no merges apply', () => {
    const t = new BPETokenizer();
    t.loadFromObjects({}, []);
    expect(t._bpe('xyz')).toEqual(['x', 'y', 'z']);
});

// ── encode options ────────────────────────────────────────────────────────────

test('encode with addBos prepends bosId', () => {
    const t = buildTinyTokenizer();
    t.vocab.set('<|im_start|>', 999);
    t.idToToken.set(999, '<|im_start|>');
    t.bosId = 999;

    const ids = t.encode('abc', { addBos: true });
    expect(ids[0]).toBe(999);
});

test('encode with addEos appends eosId', () => {
    const t = buildTinyTokenizer();
    t.vocab.set('<|im_end|>', 998);
    t.idToToken.set(998, '<|im_end|>');
    t.eosId = 998;

    const ids = t.encode('abc', { addEos: true });
    expect(ids[ids.length - 1]).toBe(998);
});

test('encode with addBos false does not prepend bosId', () => {
    const t = buildTinyTokenizer();
    t.vocab.set('<|im_start|>', 999);
    t.idToToken.set(999, '<|im_start|>');
    t.bosId = 999;

    const ids = t.encode('abc', { addBos: false });
    expect(ids[0]).not.toBe(999);
});

// ── padOrTruncate – additional cases ─────────────────────────────────────────

test('padOrTruncate returns same array when already maxLen', () => {
    const t   = buildTinyTokenizer();
    const ids = [1, 2, 3];
    expect(t.padOrTruncate(ids, 3)).toEqual([1, 2, 3]);
});

test('padOrTruncate with custom padId uses that id', () => {
    const t = buildTinyTokenizer();
    t.vocab.set('<|endoftext|>', 777);
    t.idToToken.set(777, '<|endoftext|>');
    t.padId = 777;

    const out = t.padOrTruncate([1], 3, 'right');
    expect(out[1]).toBe(777);
    expect(out[2]).toBe(777);
});

// ── loadFromObjects ────────────────────────────────────────────────────────────

test('loadFromObjects sets bosId when BOS token present in vocab', () => {
    const t = new BPETokenizer();
    t.loadFromObjects({ '<|im_start|>': 5 }, []);
    expect(t.bosId).toBe(5);
});

test('loadFromObjects sets eosId when EOS token present in vocab', () => {
    const t = new BPETokenizer();
    t.loadFromObjects({ '<|im_end|>': 6 }, []);
    expect(t.eosId).toBe(6);
});

test('loadFromObjects sets padId when pad token present in vocab', () => {
    const t = new BPETokenizer();
    t.loadFromObjects({ '<|endoftext|>': 7 }, []);
    expect(t.padId).toBe(7);
});

test('loadFromObjects bosId is null when BOS token absent', () => {
    const t = new BPETokenizer();
    t.loadFromObjects({ 'hello': 0 }, []);
    expect(t.bosId).toBeNull();
});

// ── decode ────────────────────────────────────────────────────────────────────

test('decode empty id list returns empty string', () => {
    const t = buildTinyTokenizer();
    expect(t.decode([])).toBe('');
});

test('decode skips unknown ids without throwing', () => {
    const t = buildTinyTokenizer();
    expect(() => t.decode([99999])).not.toThrow();
});
