/**
 * knowledge/facts/extractorVersion.ts — the COMPOSITE extractor version of `FileFacts` (PRD 41 §B3).
 *
 * A fact set is identified by more than one number: the extractor code version, each tree-sitter
 * grammar's package version, and the content-hash (EOL) rule. {@link factsExtractorTag} composes them into
 * the `extractor` tag a `FileFacts` carries — per LANGUAGE, so a grammar bump changes the tag of (and so
 * re-extracts) only the files of that language. {@link extractorStampMeta} gives the whole stamp as a
 * `ComparabilityMeta` so `comparable()` (C9) reports a family, code, grammar or EOL change by reason.
 *
 * Separate from `EXTRACTOR_VERSION` (sources.ts), which versions the knowledge packs' `Extracted`
 * nodes and edges; this one versions the fact contract the rules read.
 */

import type { ComparabilityMeta } from '../comparability.js';
import { CONTENT_HASH_RULE, contentHashRuleOf } from '../contentHash.js';
import { compareStr } from '../refs.js';

/** The FileFacts contract version (v3: exports, calls, entry points, capability imports, test facts). Bump on a shape change. */
export const FACTS_EXTRACTOR_VERSION = 3;

/** Tree-sitter in Node hosts; the regex fallback in the Worker. The two never mix in one comparison (§B3, C9). */
export type ExtractorFamily = 'tree-sitter' | 'regex';

/** What a host's extractor is built from. */
export interface ExtractorStamp {
    family: ExtractorFamily;
    /** The host extractor's own revision within {@link FACTS_EXTRACTOR_VERSION} (queries changed, same shape). Default 0. */
    revision?: number;
    /** Language → grammar package version (`typescript` → `0.23.2`). A regex extractor has none. */
    grammars?: Readonly<Record<string, string>>;
    /** The content-hash rule the host hashes with. Default {@link CONTENT_HASH_RULE}. */
    contentHashRule?: string;
}

const codeOf = (stamp: ExtractorStamp) => `${stamp.family}@${FACTS_EXTRACTOR_VERSION}.${stamp.revision ?? 0}`;
const ruleOf = (stamp: ExtractorStamp) => contentHashRuleOf(stamp.contentHashRule ?? CONTENT_HASH_RULE);

/**
 * The composite tag: `<family>@<version>.<revision>/<language>=<grammar>/<eol rule>` for one language (the grammar
 * part omitted when that language has none), or with every grammar (`a=1,b=2`, sorted) when `language` is omitted.
 * The family is the part before the first `@`, as `comparable()` reads it.
 */
export function factsExtractorTag(stamp: ExtractorStamp, language?: string): string {
    const grammars = stamp.grammars ?? {};
    const parts = [codeOf(stamp)];
    if (language !== undefined) {
        if (grammars[language] !== undefined) parts.push(`${language}=${grammars[language]}`);
    } else {
        const all = Object.keys(grammars).sort(compareStr).map((l) => `${l}=${grammars[l]}`);
        if (all.length > 0) parts.push(all.join(','));
    }
    parts.push(ruleOf(stamp));
    return parts.join('/');
}

/** Whether facts were produced by this stamp for their language: false ⇒ the file is owed a re-extraction. */
export function isFactsCurrent(facts: { factsVersion: number; extractor: string; language: string }, stamp: ExtractorStamp): boolean {
    return facts.factsVersion === FACTS_EXTRACTOR_VERSION && facts.extractor === factsExtractorTag(stamp, facts.language);
}

/**
 * The stamp as comparability meta: `extractor` = family@version.revision, the EOL rule, and each grammar as tool
 * `grammar:<language>` — so `comparable()` names exactly which part moved.
 */
export function extractorStampMeta(stamp: ExtractorStamp): ComparabilityMeta {
    const grammars = stamp.grammars ?? {};
    return {
        extractor: codeOf(stamp),
        extractorVersion: FACTS_EXTRACTOR_VERSION,
        contentHashRule: ruleOf(stamp),
        tools: Object.fromEntries(Object.keys(grammars).sort(compareStr).map((l) => [`grammar:${l}`, grammars[l]!])),
    };
}
