/**
 * knowledge/findingIdentity.ts — finding identity v2 (PRD 41 §C3): the exact `id` (set membership,
 * unchanged), a STABLE `anchor` that survives line movement and member churn, and the sorted
 * `members` lineage compares by similarity.
 *
 * The anchor follows SARIF `partialFingerprints`:
 *
 *     <kind>@<ruleVersion>:<anchorKind>:<primary anchor>[~<normalized-snippet hash>]
 *
 * It NEVER holds a line number or the member set. The rule that emits a finding names its primary
 * anchor (`Finding.anchor`, `Finding.anchorKind`); {@link identityOf} composes the rest. A finding
 * whose rule names none gets the {@link defaultAnchorOf default}: its lexicographically first node
 * ref (one member, never the set; refs hold no line numbers), else its pack.
 */

import { hashText } from '../ingest/hash.js';
import type { Finding, NodeRef } from './model.js';
import { compareStr, fileRefOf } from './refs.js';

/** What a finding's primary anchor names. `group` is a named group (a clone-group badge, a pack). */
export type AnchorKind = NonNullable<Finding['anchorKind']>;

export interface FindingIdentity {
    /** Exact identity (kind + all refs), unchanged from v1 — set membership. */
    id: string;
    /** `<kind>@<ruleVersion>:<anchorKind>:<primary>[~<snippetHash>]` — no line numbers, no member set. */
    anchor: string;
    anchorKind: AnchorKind;
    /** Sorted, deduped refs (files/symbols) — lineage similarity. */
    members: string[];
}

/** The rule version an identity is composed with when the caller names none. */
export const DEFAULT_RULE_VERSION = 1;

/** Separates the primary anchor from its normalized-snippet hash. */
const SNIPPET_SEPARATOR = '~';

/**
 * A snippet with formatting noise removed: CRLF → LF, each line trimmed, inner whitespace runs
 * collapsed to one space, blank lines dropped. Re-indenting or re-wrapping blank lines keeps the hash.
 */
export function normalizeSnippet(snippet: string): string {
    return snippet.replace(/\r\n?/g, '\n').split('\n')
        .map((l) => l.trim().replace(/\s+/g, ' '))
        .filter((l) => l.length > 0)
        .join('\n');
}

/** A primary anchor with the normalized-snippet hash appended (unchanged when there is no snippet). */
export function anchorWithSnippet(primary: string, snippet?: string): string {
    return snippet === undefined ? primary : `${primary}${SNIPPET_SEPARATOR}${hashText(normalizeSnippet(snippet))}`;
}

/** The anchor kind a node ref implies: a section/symbol ref (`…#name`) → symbol, a source ref → file, else group. */
export function anchorKindOfRef(ref: string): AnchorKind {
    if (!ref.startsWith('src:')) return ref.startsWith('ent:') ? 'symbol' : 'group';
    return fileRefOf(ref as NodeRef) === ref ? 'file' : 'symbol';
}

/** The ONE anchor string: `<kind>@<ruleVersion>:<anchorKind>:<primary>`. */
export function composeAnchor(kind: string, ruleVersion: number, anchorKind: AnchorKind, primary: string): string {
    return `${kind}@${ruleVersion}:${anchorKind}:${primary}`;
}

/** The finding kind an identity anchor was composed for (the part before `@`). */
export function kindOfAnchor(anchor: string): string {
    const at = anchor.indexOf('@');
    return at < 0 ? anchor : anchor.slice(0, at);
}

/**
 * The default primary anchor of a finding whose rule set none: the lexicographically first node
 * ref (kind from {@link anchorKindOfRef}), else `pack:<pack>` (group). Never a line, never the set.
 */
export function defaultAnchorOf(f: Pick<Finding, 'nodes' | 'pack'>): { anchor: string; anchorKind: AnchorKind } {
    const first = [...f.nodes].sort(compareStr)[0];
    return first === undefined ? { anchor: `pack:${f.pack}`, anchorKind: 'group' } : { anchor: first, anchorKind: anchorKindOfRef(first) };
}

/** The v2 identity of a finding. `ruleVersion` is the emitting rule's version (default {@link DEFAULT_RULE_VERSION}). */
export function identityOf(f: Finding, ruleVersion: number = DEFAULT_RULE_VERSION): FindingIdentity {
    const fallback = f.anchor === undefined ? defaultAnchorOf(f) : undefined;
    const primary = f.anchor ?? fallback!.anchor;
    const anchorKind = f.anchorKind ?? fallback?.anchorKind ?? anchorKindOfRef(primary);
    return {
        id: f.id,
        anchor: composeAnchor(f.kind, ruleVersion, anchorKind, primary),
        anchorKind,
        members: [...new Set<string>(f.nodes as readonly NodeRef[])].sort(compareStr),
    };
}
