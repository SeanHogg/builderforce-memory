/**
 * knowledge/patternFindings.ts — the file nodes a pattern-badge finding is raised on: the code pack
 * writes detected pattern ids into each file's `patterns` badge (comma-separated); a finding such as
 * `god-module` or `io-in-loop` reads them back from here, so the badge is parsed in one place.
 */

import type { KNode, KnowledgeGraph } from './model.js';
import { compareStr } from './refs.js';

const PATTERNS_BADGE = 'patterns';

/** File nodes whose `patterns` badge names `kind`, sorted by ref. */
export function filesWithPattern(g: KnowledgeGraph, kind: string): KNode[] {
    return g.nodes
        .filter((n) => {
            if (n.kind !== 'file') return false;
            const patterns = n.badges?.[PATTERNS_BADGE];
            return typeof patterns === 'string' && patterns.split(',').includes(kind);
        })
        .sort((a, b) => compareStr(a.ref, b.ref));
}
