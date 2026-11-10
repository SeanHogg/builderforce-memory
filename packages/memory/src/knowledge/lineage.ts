/**
 * knowledge/lineage.ts — the ONE deterministic lineage matcher (PRD 41 §C3), between the findings of
 * two sides of a run:
 *
 *   1. Exact `id` match → `persisting`.
 *   2. Same `anchor` → `changed`, sub-kind `grew` / `shrank` (member count) or `moved` (equal counts,
 *      different members). Several findings on one anchor pair in ascending id order.
 *   3. Same kind AND same anchor kind, TWO OR MORE members on both sides, member-set Jaccard ≥ 0.5 →
 *      `changed`. Pairs are taken greedily by descending Jaccard, then ascending before id, then
 *      ascending after id. A single-member finding never matches here, so two unrelated findings in
 *      one file are never "changed".
 *   4. The rest → `introduced` (after only) or `resolved` (before only). Comparability (C9) is the
 *      caller's: a `not_comparable` kind must not be fed here.
 *
 * Input order never matters: each side is deduped by id (first kept) and sorted first. Jaccard is
 * compared as exact integer ratios, so no float enters an ordering.
 */

import type { FindingIdentity } from './findingIdentity.js';
import { kindOfAnchor } from './findingIdentity.js';
import { compareStr } from './refs.js';

export type Lineage = 'introduced' | 'resolved' | 'changed' | 'persisting';
export type LineageChange = 'grew' | 'shrank' | 'moved';

export interface LineageEntry {
    lineage: Lineage;
    /** Only on anchor (rule 2) matches whose members differ. */
    change?: LineageChange;
    /** The before finding's id (absent for `introduced`). */
    before?: string;
    /** The after finding's id (absent for `resolved`). */
    after?: string;
}

/** Rule 3's member-set Jaccard floor (compared as `inter >= floor × union`, exact in binary). */
export const LINEAGE_MIN_JACCARD = 0.5;
/** Rule 3 needs at least this many members on BOTH sides. */
export const LINEAGE_MIN_MEMBERS = 2;

function uniqueSorted(list: readonly FindingIdentity[]): FindingIdentity[] {
    const byId = new Map<string, FindingIdentity>();
    for (const f of list) if (!byId.has(f.id)) byId.set(f.id, f);
    return [...byId.values()].sort((a, b) => compareStr(a.id, b.id));
}

function changeOf(before: FindingIdentity, after: FindingIdentity): LineageChange | undefined {
    const b = new Set(before.members), a = new Set(after.members);
    if (a.size > b.size) return 'grew';
    if (a.size < b.size) return 'shrank';
    for (const m of a) if (!b.has(m)) return 'moved';
    return undefined;
}

/** |A ∩ B| and |A ∪ B| of two member lists. */
function overlap(a: readonly string[], b: readonly string[]): { inter: number; union: number } {
    const sa = new Set(a), sb = new Set(b);
    let inter = 0;
    for (const m of sa) if (sb.has(m)) inter++;
    return { inter, union: sa.size + sb.size - inter };
}

interface Candidate { before: FindingIdentity; after: FindingIdentity; inter: number; union: number }

const groupKey = (f: FindingIdentity) => `${kindOfAnchor(f.anchor)}\u0000${f.anchorKind}`;

export function matchLineage(before: readonly FindingIdentity[], after: readonly FindingIdentity[]): LineageEntry[] {
    const b = uniqueSorted(before), a = uniqueSorted(after);
    const out: LineageEntry[] = [];
    const usedB = new Set<string>(), usedA = new Set<string>();
    const pair = (x: FindingIdentity, y: FindingIdentity, entry: LineageEntry) => { usedB.add(x.id); usedA.add(y.id); out.push(entry); };

    // Rule 1: exact id.
    const afterIds = new Set(a.map((f) => f.id));
    for (const f of b) if (afterIds.has(f.id)) pair(f, f, { lineage: 'persisting', before: f.id, after: f.id });

    // Rule 2: same anchor, paired in ascending id order within the anchor.
    const byAnchor = (list: FindingIdentity[], used: Set<string>) => {
        const m = new Map<string, FindingIdentity[]>();
        for (const f of list) if (!used.has(f.id)) (m.get(f.anchor) ?? m.set(f.anchor, []).get(f.anchor)!).push(f);
        return m;
    };
    const afterByAnchor = byAnchor(a, usedA);
    for (const [anchor, olds] of [...byAnchor(b, usedB)].sort((x, y) => compareStr(x[0], y[0]))) {
        const news = afterByAnchor.get(anchor) ?? [];
        for (let i = 0; i < Math.min(olds.length, news.length); i++) {
            const change = changeOf(olds[i]!, news[i]!);
            pair(olds[i]!, news[i]!, { lineage: 'changed', ...(change ? { change } : {}), before: olds[i]!.id, after: news[i]!.id });
        }
    }

    // Rule 3: same kind + anchor kind, ≥2 members both sides, Jaccard ≥ 0.5; greedy.
    const eligible = (f: FindingIdentity) => new Set(f.members).size >= LINEAGE_MIN_MEMBERS;
    const afterByGroup = new Map<string, FindingIdentity[]>();
    for (const f of a) if (!usedA.has(f.id) && eligible(f)) (afterByGroup.get(groupKey(f)) ?? afterByGroup.set(groupKey(f), []).get(groupKey(f))!).push(f);
    const candidates: Candidate[] = [];
    for (const x of b) {
        if (usedB.has(x.id) || !eligible(x)) continue;
        for (const y of afterByGroup.get(groupKey(x)) ?? []) {
            const { inter, union } = overlap(x.members, y.members);
            if (union > 0 && inter >= LINEAGE_MIN_JACCARD * union) candidates.push({ before: x, after: y, inter, union });
        }
    }
    candidates.sort((p, q) => q.inter * p.union - p.inter * q.union || compareStr(p.before.id, q.before.id) || compareStr(p.after.id, q.after.id));
    for (const c of candidates) {
        if (usedB.has(c.before.id) || usedA.has(c.after.id)) continue;
        pair(c.before, c.after, { lineage: 'changed', before: c.before.id, after: c.after.id });
    }

    // Rule 4: the remainder.
    for (const f of b) if (!usedB.has(f.id)) out.push({ lineage: 'resolved', before: f.id });
    for (const f of a) if (!usedA.has(f.id)) out.push({ lineage: 'introduced', after: f.id });

    return out.sort((p, q) => compareStr(p.after ?? p.before!, q.after ?? q.before!) || compareStr(p.before ?? '', q.before ?? ''));
}
