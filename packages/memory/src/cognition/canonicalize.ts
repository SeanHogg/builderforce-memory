/**
 * cognition/canonicalize.ts — the stable subject-key canonicalizer.
 *
 * Write-Through Cognition's single-incumbent guarantee rests on logically-equal
 * subjects mapping to the SAME storage key. Without normalization,
 * `"Pkg:SSM-Stack"` and `"pkg:ssm-stack"` are two live incumbents for one
 * subject — exactly the drift the layer exists to prevent. `commit()` runs every
 * subject key through {@link canonicalizeSubjectKey} before recall and write so
 * superseding facts always collide and replace.
 *
 * Canonical form: Unicode NFC → case-fold (locale-independent lower-case) →
 * trim → collapse internal whitespace runs to a single space → optional alias
 * resolution. Conservative by design: it folds case and whitespace/encoding
 * noise but never conflates distinct separators (so `pkg:a-b` ≠ `pkg:a b`).
 */

/** A normalized alias table: every key AND value is already canonical. */
export type AliasTable = ReadonlyMap<string, string>;

/**
 * Normalize a raw subject key to its canonical form (no alias resolution).
 * Exported so callers can canonicalize alias-table entries with the exact same
 * rules used at commit time.
 */
export function normalizeSubjectKey(raw: string): string {
  // NFC unifies composed/decomposed encodings of the same text; toLowerCase is
  // the practical case-fold for our key space (ASCII identifiers + Unicode).
  const folded = raw.normalize('NFC').toLowerCase();
  // Trim, then collapse any run of Unicode whitespace to a single ASCII space.
  return folded.trim().replace(/\s+/g, ' ');
}

/**
 * Build a normalized alias table from a raw `{ alias: canonical }` map. Both
 * sides are normalized so lookups are exact regardless of how the caller cased
 * or spaced them. Self-mapping and empty entries are dropped.
 */
export function buildAliasTable(aliases?: Record<string, string>): AliasTable {
  const table = new Map<string, string>();
  if (!aliases) return table;
  for (const [alias, canonical] of Object.entries(aliases)) {
    const a = normalizeSubjectKey(alias);
    const c = normalizeSubjectKey(canonical);
    if (a && c && a !== c) table.set(a, c);
  }
  return table;
}

/**
 * Canonicalize a subject key: normalize, then resolve through the alias table
 * (one hop — alias values are required to already be canonical). Throws on a key
 * that is empty after normalization, since a blank subject can't anchor a
 * single-incumbent guarantee.
 */
export function canonicalizeSubjectKey(raw: string, aliases?: AliasTable): string {
  const normalized = normalizeSubjectKey(raw);
  if (!normalized) {
    throw new Error('subjectKey is empty after normalization — provide a non-blank subject');
  }
  return aliases?.get(normalized) ?? normalized;
}
