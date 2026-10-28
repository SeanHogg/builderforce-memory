/**
 * tokenize.ts — the code-aware tokenizer the knowledge index's lexical tier uses.
 *
 * Cards name things the way code does: `retryPolicy.ts`, `stripe_webhook`,
 * `api/src/billing/invoice.ts`. The package's plain `tokenize` keeps
 * `retrypolicy` whole, so "retry policy" never matches it. This splits paths on
 * separators, then each word on snake_case and camelCase boundaries, and emits
 * the parts AND the whole compound — an exact identifier still scores higher.
 *
 * Lowercasing is `toLowerCase` (locale-free), so the tokens are the same on every
 * host. Pure and zero-dependency.
 */

/** camelCase / PascalCase / acronym boundaries: `parseHTTPResponse` → parse, HTTP, Response. */
function splitCamel(word: string): string[] {
    return word
        .replace(/([\p{Ll}\p{N}])(\p{Lu})/gu, '$1 $2')
        .replace(/(\p{Lu}+)(\p{Lu}\p{Ll})/gu, '$1 $2')
        .split(' ');
}

/**
 * Splits `text` into lowercase terms: every word (a run of letters, digits and `_`)
 * yields its snake/camel parts, plus the whole word when it has more than one part.
 */
export function codeTokenize(text: string): string[] {
    const out: string[] = [];
    for (const word of text.split(/[^\p{L}\p{N}_]+/u)) {
        if (!word) continue;
        const parts = word.split('_').flatMap(splitCamel).filter(Boolean).map((p) => p.toLowerCase());
        if (parts.length === 0) continue;
        out.push(...parts);
        if (parts.length > 1) out.push(word.toLowerCase());
    }
    return out;
}

/** Case-folds an exact-match key or query the same way on insert and on lookup. */
export function foldKey(key: string): string {
    return key.trim().toLowerCase();
}
