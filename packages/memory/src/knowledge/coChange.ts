/**
 * knowledge/coChange.ts — co-change association over git history (PRD 38 P4: impact prediction).
 *
 * For files A and B over the N kept commits:
 *   support    = together(A, B)                (commits touching both; kept when ≥ COCHANGE_MIN_SUPPORT)
 *   confidence = P(B|A) = together / ofSelf(A), with its credible interval (`rateInterval`, ../stats)
 *   lift       = P(A∧B) / (P(A)P(B)) = together·N / (ofSelf(A)·ofSelf(B))   (kept when ≥ COCHANGE_MIN_LIFT)
 *
 * A commit touching more than COCHANGE_MAX_COMMIT_FILES files (its whole-repo `fileCount` when given, else its distinct
 * refs) is a bulk change and is dropped before anything is counted (it is not in N either); so is a commit touching no
 * ref. Of the rest, the newest COCHANGE_MAX_COMMITS are kept. Each file keeps at most COCHANGE_MAX_PARTNERS partners,
 * ranked confidence desc, lift desc, ref asc.
 *
 * History order: by `Date.parse(at)` (stable) when every commit has a parseable `at`; otherwise the input order, which
 * must then be oldest first.
 *
 * Determinism: refs are deduped per commit and numbered in sorted order; pair counts are integers keyed by index;
 * thresholds compare integers. Same commits in any order ⇒ identical output (when dated, or under the commit cap).
 *
 * The badge `coChange` is a string written ONLY by {@link encodeCoChange} and read ONLY by {@link decodeCoChange}.
 * The caller (it holds the git history) stamps it; `buildGraph` then stamps `coChangeHidden`
 * ({@link applyCoChangeHidden}): the partners that no reference explains.
 */

import { rateInterval, type InsufficientData } from '../stats/index.js';
import { isReferenceEdge } from './communities.js';
import type { KEdge, KNode, NodeRef } from './model.js';
import { compareStr, fileRefOf, round6 } from './refs.js';

/** Commits a pair must share to be kept. */
export const COCHANGE_MIN_SUPPORT = 3;
/** Lift a pair must reach to be kept. */
export const COCHANGE_MIN_LIFT = 2;
/** A commit touching more files than this is a bulk change and is excluded. */
export const COCHANGE_MAX_COMMIT_FILES = 50;
/** Partners kept per file. */
export const COCHANGE_MAX_PARTNERS = 8;
/** At most this many (newest) commits are counted. */
export const COCHANGE_MAX_COMMITS = 5000;
/** The badge keys: the encoded partner list, and how many of them no reference explains. */
export const COCHANGE_BADGE = 'coChange';
export const COCHANGE_HIDDEN_BADGE = 'coChangeHidden';
/** First line of a badge value; a reader returns nothing for any other version. */
const BADGE_VERSION = 'v1';

export interface CoChangeCommit {
    /** ISO time of the commit; orders the history. */
    at?: string;
    /** File node refs the commit touched (the caller maps paths to refs). */
    refs: readonly NodeRef[];
    /** Files the commit touched in the whole repo (refs only cover the root); the bulk cut uses the larger. */
    fileCount?: number;
}

export interface CoChangePartner {
    ref: NodeRef;
    /** Kept commits touching both files. */
    together: number;
    /** Kept commits touching the file itself. */
    ofSelf: number;
    /** P(partner | self) = together / ofSelf, round6. */
    confidence: number;
    /** P(self ∧ partner) / (P(self) P(partner)), round6. */
    lift: number;
    /** Credible interval of the confidence (`rateInterval(together, ofSelf)`), round6. */
    low: number;
    high: number;
}

/** What the `coChange` badge carries per partner. */
export interface CoChangeBadgeEntry {
    ref: NodeRef;
    together: number;
    ofSelf: number;
    low: number;
    high: number;
}

export interface CoChangeOptions {
    minSupport?: number;
    minLift?: number;
    maxCommitFiles?: number;
    maxPartners?: number;
    maxCommits?: number;
}

/** `skipped`: held-out (commit, file) cases whose file had no partners to predict with. */
export type CoChangeRecall = { recall: number; cases: number; skipped: number } | InsufficientData;

type KeptCommit = { refs: NodeRef[] };

/** The history oldest first: by parsed `at` (stable) when every commit is dated, else as given. */
function chronological(commits: ReadonlyArray<CoChangeCommit>): CoChangeCommit[] {
    const times = commits.map((c) => (c.at === undefined ? Number.NaN : Date.parse(c.at)));
    if (!times.every(Number.isFinite)) return [...commits];
    return commits.map((c, i) => ({ c, i, t: times[i]! })).sort((a, b) => a.t - b.t || a.i - b.i).map((x) => x.c);
}

/** Oldest first: each commit's distinct refs (sorted), bulk and empty commits dropped, the newest `maxCommits` kept. */
function historyOf(commits: ReadonlyArray<CoChangeCommit>, opts: CoChangeOptions): KeptCommit[] {
    const maxFiles = opts.maxCommitFiles ?? COCHANGE_MAX_COMMIT_FILES;
    const out: KeptCommit[] = [];
    for (const c of chronological(commits)) {
        const refs = [...new Set(c.refs)].sort(compareStr);
        if (refs.length === 0 || Math.max(c.fileCount ?? 0, refs.length) > maxFiles) continue;
        out.push({ refs });
    }
    const max = opts.maxCommits ?? COCHANGE_MAX_COMMITS;
    return out.length > max ? out.slice(out.length - max) : out;
}

const imul = Math.imul;

/**
 * Counts (i, j) index pairs in an open-addressing table: one Int32Array of [i, j, count] slots (i = -1 is empty), so a probe
 * touches one cache line and no pair allocates or boxes a key — where a `Map` spent most of the time on a 50-file commit's
 * 1,225 pairs. Linear probing, grown at half load.
 */
class PairCounter {
    private cap = 1 << 12;
    private slots = PairCounter.empty(this.cap);
    private size = 0;

    add(i: number, j: number): void {
        if ((this.size + 1) * 2 > this.cap) this.grow();
        const mask = this.cap - 1;
        const slots = this.slots;
        for (let s = PairCounter.hash(i, j) & mask; ; s = (s + 1) & mask) {
            const at = s * 3;
            const si = slots[at]!;
            if (si === -1) {
                slots[at] = i; slots[at + 1] = j; slots[at + 2] = 1; this.size += 1;
                return;
            }
            if (si === i && slots[at + 1] === j) { slots[at + 2]! += 1; return; }
        }
    }

    forEach(fn: (i: number, j: number, count: number) => void): void {
        const slots = this.slots;
        for (let at = 0; at < slots.length; at += 3) if (slots[at] !== -1) fn(slots[at]!, slots[at + 1]!, slots[at + 2]!);
    }

    private grow(): void {
        const old = this.slots;
        this.cap *= 2;
        this.slots = PairCounter.empty(this.cap);
        const mask = this.cap - 1;
        for (let from = 0; from < old.length; from += 3) {
            if (old[from] === -1) continue;
            let s = PairCounter.hash(old[from]!, old[from + 1]!) & mask;
            while (this.slots[s * 3] !== -1) s = (s + 1) & mask;
            this.slots.set(old.subarray(from, from + 3), s * 3);
        }
    }

    private static empty(cap: number): Int32Array {
        const slots = new Int32Array(cap * 3);
        for (let at = 0; at < slots.length; at += 3) slots[at] = -1;
        return slots;
    }

    /** murmur3's 32-bit finaliser over the two indexes. */
    private static hash(i: number, j: number): number {
        let h = imul(i, 0x9e3779b1) ^ j;
        h ^= h >>> 16; h = imul(h, 0x85ebca6b);
        h ^= h >>> 13; h = imul(h, 0xc2b2ae35);
        return (h ^ (h >>> 16)) >>> 0;
    }
}

const byRank = (a: Pick<CoChangePartner, 'confidence' | 'lift' | 'ref'>, b: Pick<CoChangePartner, 'confidence' | 'lift' | 'ref'>): number =>
    b.confidence - a.confidence || b.lift - a.lift || compareStr(a.ref, b.ref);

/** Partners over an already-kept history. */
function partnersOf(kept: readonly KeptCommit[], opts: CoChangeOptions): Map<NodeRef, CoChangePartner[]> {
    const minSupport = opts.minSupport ?? COCHANGE_MIN_SUPPORT;
    const minLift = opts.minLift ?? COCHANGE_MIN_LIFT;
    const maxPartners = opts.maxPartners ?? COCHANGE_MAX_PARTNERS;
    const n = kept.length;

    // Pass 1: ofSelf. A file seen in fewer than minSupport commits can be in no kept pair, so pass 2 skips it.
    const order = [...new Set(kept.flatMap((c) => c.refs))].sort(compareStr);
    const index = new Map(order.map((r, i) => [r, i] as const));
    const m = order.length;
    const ofSelf = new Array<number>(m).fill(0);
    for (const c of kept) for (const r of c.refs) ofSelf[index.get(r)!]! += 1;

    // Pass 2: pair counts over index pairs (i < j).
    const pairs = new PairCounter();
    for (const c of kept) {
        const ids = c.refs.map((r) => index.get(r)!).filter((i) => ofSelf[i]! >= minSupport); // sorted refs ⇒ ascending ids
        for (let x = 0; x < ids.length; x++) {
            for (let y = x + 1; y < ids.length; y++) pairs.add(ids[x]!, ids[y]!);
        }
    }

    // Candidates carry only counts and the ranking keys; the interval (an iterative inverse-beta solve) is computed after the
    // cap, for at most maxPartners per file, and memoised by (together, ofSelf) for this call.
    type Candidate = Omit<CoChangePartner, 'low' | 'high'>;
    const lists = new Map<number, Candidate[]>();
    const add = (self: number, other: number, together: number, lift: number) => {
        const list = lists.get(self) ?? [];
        list.push({ ref: order[other]!, together, ofSelf: ofSelf[self]!, confidence: round6(together / ofSelf[self]!), lift });
        lists.set(self, list);
    };
    // Slot order is arbitrary; every list is fully sorted (ref breaks the last tie), so the output is not.
    pairs.forEach((i, j, together) => {
        if (together < minSupport) return;
        // lift ≥ minLift ⇔ together·N ≥ minLift·ofSelf(i)·ofSelf(j): no float at the threshold.
        if (together * n < minLift * ofSelf[i]! * ofSelf[j]!) return;
        const lift = round6((together * n) / (ofSelf[i]! * ofSelf[j]!));
        add(i, j, together, lift);
        add(j, i, together, lift);
    });

    const intervals = new Map<number, { low: number; high: number }>();
    const intervalOf = (k: number, total: number) => {
        const key = k * (n + 1) + total; // k, total ≤ n
        let hit = intervals.get(key);
        if (!hit) {
            const { low, high } = rateInterval(k, total);
            hit = { low: round6(low), high: round6(high) };
            intervals.set(key, hit);
        }
        return hit;
    };
    const out = new Map<NodeRef, CoChangePartner[]>();
    for (const i of [...lists.keys()].sort((a, b) => a - b)) {
        const kept = lists.get(i)!.sort(byRank).slice(0, maxPartners);
        out.set(order[i]!, kept.map((c) => ({ ...c, ...intervalOf(c.together, c.ofSelf) })));
    }
    return out;
}

/** Every file's co-change partners (only files with at least one kept partner appear), keys in ref order. */
export function coChangePartners(commits: ReadonlyArray<CoChangeCommit>, opts: CoChangeOptions = {}): Map<NodeRef, CoChangePartner[]> {
    return partnersOf(historyOf(commits, opts), opts);
}

/**
 * The `coChange` badge value: the line `v1`, then `${together}\t${ofSelf}\t${low}\t${high}\t${ref}` per partner, '\n'-joined
 * (the ref is the line remainder). A ref holding a newline cannot be carried and is skipped.
 */
export function encodeCoChange(partners: readonly CoChangePartner[]): string {
    const lines = partners.filter((p) => !p.ref.includes('\n')).map((p) => `${p.together}\t${p.ofSelf}\t${p.low}\t${p.high}\t${p.ref}`);
    return [BADGE_VERSION, ...lines].join('\n');
}

const NUM = String.raw`(\d+(?:\.\d+)?(?:e-\d+)?)`;
// `s`: the ref is the line remainder and may hold '\r', U+2028 or U+2029 (the value is already split on '\n').
const LINE = new RegExp(String.raw`^(\d+)\t(\d+)\t${NUM}\t${NUM}\t(.+)$`, 's');

/** The partners of a `coChange` badge; [] for anything that is not a well-formed v1 value (one bad line voids it). */
export function decodeCoChange(value: unknown): CoChangeBadgeEntry[] {
    if (typeof value !== 'string') return [];
    const [version, ...lines] = value.split('\n');
    if (version !== BADGE_VERSION) return [];
    const out: CoChangeBadgeEntry[] = [];
    for (const line of lines) {
        const hit = LINE.exec(line);
        if (!hit) return [];
        const together = Number(hit[1]), ofSelf = Number(hit[2]), low = Number(hit[3]), high = Number(hit[4]);
        if (!Number.isSafeInteger(together) || !Number.isSafeInteger(ofSelf) || together < 1 || together > ofSelf) return [];
        if (!(low >= 0 && low <= high && high <= 1)) return [];
        out.push({ ref: hit[5]! as NodeRef, together, ofSelf, low, high });
    }
    return out;
}

/**
 * The PRD's accept measure: fit on the older (1 − heldOutShare) kept commits, then for each held-out commit and each file A
 * in it that has partners, recall = |top-k partners ∩ (commit − A)| / |commit − A|; the mean over those cases. A held-out
 * file with no partners is counted in `skipped`. Insufficient data when nothing can be held out or no case is predicted.
 */
export function coChangeRecallAtK(
    commits: ReadonlyArray<CoChangeCommit>, k: number = COCHANGE_MAX_PARTNERS, heldOutShare = 0.2, opts: CoChangeOptions = {},
): CoChangeRecall {
    const insufficient: InsufficientData = { status: 'insufficient-data' };
    const kept = historyOf(commits, opts);
    const held = Math.floor(kept.length * heldOutShare);
    if (held < 1 || held >= kept.length || k < 1) return insufficient;
    const model = partnersOf(kept.slice(0, kept.length - held), { ...opts, maxPartners: k });
    let sum = 0, cases = 0, skipped = 0;
    for (const c of kept.slice(kept.length - held)) {
        if (c.refs.length < 2) continue;
        for (const a of c.refs) {
            const predicted = model.get(a);
            if (!predicted || predicted.length === 0) { skipped += 1; continue; }
            const others = new Set(c.refs.filter((r) => r !== a));
            sum += predicted.filter((p) => others.has(p.ref)).length / others.size;
            cases += 1;
        }
    }
    return cases === 0 ? insufficient : { recall: round6(sum / cases), cases, skipped };
}

/**
 * The one call `buildGraph` makes: on every file node carrying a `coChange` badge, `coChangeHidden` = how many of its
 * partners (that are nodes of the graph) share no reference with it — a live COMMUNITY_EDGE_KINDS edge in either direction,
 * both ends rolled up to their file (a symbol or section ref counts for its file; an out-of-root placeholder explains too).
 * Absent when 0, and removed from a file that no longer carries `coChange`.
 */
export function applyCoChangeHidden(nodes: Iterable<KNode>, edges: Iterable<KEdge>): void {
    const all = new Map<NodeRef, KNode>();
    for (const n of nodes) all.set(n.ref, n);
    const explained = new Map<NodeRef, Set<NodeRef>>();
    const link = (a: NodeRef, b: NodeRef) => {
        const set = explained.get(a) ?? new Set<NodeRef>();
        set.add(b);
        explained.set(a, set);
    };
    for (const e of edges) {
        if (!isReferenceEdge(e)) continue;
        const a = fileRefOf(e.from), b = fileRefOf(e.to);
        if (a === b) continue;
        link(a, b);
        link(b, a);
    }
    for (const n of all.values()) {
        if (n.kind !== 'file' || !n.badges) continue;
        const value = n.badges[COCHANGE_BADGE];
        const near = explained.get(n.ref);
        const hidden = value === undefined
            ? 0
            : decodeCoChange(value).filter((p) => p.ref !== n.ref && all.has(p.ref) && !near?.has(p.ref)).length;
        const badges = { ...n.badges };
        if (hidden > 0) badges[COCHANGE_HIDDEN_BADGE] = hidden;
        else delete badges[COCHANGE_HIDDEN_BADGE];
        if (Object.keys(badges).length > 0) n.badges = badges;
        else delete n.badges;
    }
}
