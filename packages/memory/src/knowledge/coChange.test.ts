/**
 * src/knowledge/coChange.test.ts
 * PRD 38 P4: co-change association — the support / lift thresholds at their boundaries, the bulk-commit cut (distinct refs
 * and whole-repo `fileCount`), the commit cap, the partner cap and ranking, determinism, the versioned badge codec, recall@k
 * on a history with known coupling (and without leakage), and the hidden-coupling badge through `buildGraph` and the outline.
 * Hand computations are in the comments.
 */

import {
    applyCoChangeHidden, coChangePartners, coChangeRecallAtK, COCHANGE_MAX_COMMIT_FILES, COCHANGE_MAX_COMMITS, COCHANGE_MAX_PARTNERS,
    COCHANGE_MIN_LIFT, COCHANGE_MIN_SUPPORT, decodeCoChange, encodeCoChange, type CoChangeCommit, type CoChangePartner,
} from './coChange.js';
import { buildGraph } from './build.js';
import { buildOutline } from './outline.js';
import type { BuildGraphInput, KEdge } from './model.js';
import { rateInterval } from '../stats/index.js';
import { round6 } from './refs.js';
import { edge, extractedOf, fileNode, ref } from './testkit.test-util.js';

const C = (...paths: string[]): CoChangeCommit => ({ refs: paths.map(ref) });
const times = (n: number, c: CoChangeCommit): CoChangeCommit[] => Array.from({ length: n }, () => c);
/** n commits that each touch one file of their own (they only raise N). */
const fillers = (n: number, tag = 'z'): CoChangeCommit[] => Array.from({ length: n }, (_, i) => C(`${tag}${String(i).padStart(4, '0')}.ts`));
const same = (a: unknown, b: unknown) => expect(JSON.stringify(a)).toBe(JSON.stringify(b));
/** The counted fields of a partner (the interval is checked on its own). */
const counts = (ps: readonly CoChangePartner[] | undefined) => ps?.map(({ ref: r, together, ofSelf, confidence, lift }) => ({ ref: r, together, ofSelf, confidence, lift }));

describe('coChangePartners — thresholds', () => {
    it('exposes the PRD constants', () => {
        expect([COCHANGE_MIN_SUPPORT, COCHANGE_MIN_LIFT, COCHANGE_MAX_COMMIT_FILES, COCHANGE_MAX_PARTNERS, COCHANGE_MAX_COMMITS]).toEqual([3, 2, 50, 8, 5000]);
    });

    it('keeps a pair at exactly lift 2 and drops it just below', () => {
        // a, b together 3 times, never apart: lift = 3·N / (3·3) = N/3. N = 6 → 2 (kept); N = 5 → 1.666… (dropped).
        const at2 = coChangePartners([...times(3, C('a.ts', 'b.ts')), ...fillers(3)]);
        expect(counts(at2.get(ref('a.ts')))).toEqual([{ ref: ref('b.ts'), together: 3, ofSelf: 3, confidence: 1, lift: 2 }]);
        expect(counts(at2.get(ref('b.ts')))).toEqual([{ ref: ref('a.ts'), together: 3, ofSelf: 3, confidence: 1, lift: 2 }]);
        expect(coChangePartners([...times(3, C('a.ts', 'b.ts')), ...fillers(2)]).size).toBe(0);
    });

    it('keeps a pair at support 3 and drops it at 2, however high the lift', () => {
        expect(coChangePartners([...times(3, C('a.ts', 'b.ts')), ...fillers(50)]).size).toBe(2);
        expect(coChangePartners([...times(2, C('a.ts', 'b.ts')), ...fillers(50)]).size).toBe(0);
    });

    it('confidence is P(B|A), asymmetric, and carries the rate interval of together / ofSelf', () => {
        // a in 6 commits (3 with b), b in 3. N = 6 + 20 = 26. lift = 3·26/(6·3) = 4.333333.
        const r = coChangePartners([...times(3, C('a.ts', 'b.ts')), ...times(3, C('a.ts')), ...fillers(20)]);
        const ab = r.get(ref('a.ts'))![0]!;
        expect(counts([ab])).toEqual([{ ref: ref('b.ts'), together: 3, ofSelf: 6, confidence: 0.5, lift: 4.333333 }]);
        const i = rateInterval(3, 6);
        expect([ab.low, ab.high]).toEqual([round6(i.low), round6(i.high)]);
        expect(ab.low).toBeLessThan(0.5);
        expect(ab.high).toBeGreaterThan(0.5);
        // b → a is 3 of 3: the interval's upper end is pinned to 1.
        expect(r.get(ref('b.ts'))![0]).toMatchObject({ ofSelf: 3, confidence: 1, lift: 4.333333, high: 1 });
    });
});

describe('coChangePartners — bulk commits and the commit cap', () => {
    const bulk = (size: number) => C('a.ts', 'b.ts', ...Array.from({ length: size - 2 }, (_, i) => `bulk${i}.ts`));
    const base = [...times(2, C('a.ts', 'b.ts')), ...fillers(10)];

    it('a commit of exactly 50 files counts; one of 51 is excluded', () => {
        // Two ordinary a+b commits and one bulk commit with a and b: support 3 with the bulk commit, 2 without it.
        expect(coChangePartners([...base, bulk(50)]).get(ref('a.ts'))!.map((p) => [p.ref, p.together])).toEqual([[ref('b.ts'), 3]]);
        expect(coChangePartners([...base, bulk(51)]).size).toBe(0);
    });

    it('counts distinct refs: duplicates in a commit do not push it over the cap', () => {
        const dupes: CoChangeCommit = { refs: [...bulk(50).refs, ref('a.ts'), ref('b.ts')] };
        expect(coChangePartners([...base, dupes]).size).toBe(2);
    });

    it('uses the whole-repo fileCount: 300 files touched with only 40 in the root is still bulk', () => {
        expect(coChangePartners([...base, { ...bulk(40), fileCount: 300 }]).size).toBe(0);
        // A fileCount below the distinct refs never lets a bulk commit through, and one at the cap keeps it.
        expect(coChangePartners([...base, { ...bulk(51), fileCount: 10 }]).size).toBe(0);
        expect(coChangePartners([...base, { ...bulk(40), fileCount: 50 }]).size).toBe(2);
    });

    it('keeps the newest maxCommits by `at`', () => {
        // Old a+b coupling (3 commits) then 10 newer fillers. With maxCommits 10 only the fillers are counted.
        const dated = [...times(3, C('a.ts', 'b.ts')), ...fillers(10)].map((c, i) => ({ ...c, at: new Date(Date.UTC(2026, 0, 1 + i)).toISOString() }));
        expect(coChangePartners(dated).size).toBe(2);
        expect(coChangePartners([...dated].reverse(), { maxCommits: 10 }).size).toBe(0);
        expect(coChangePartners([...dated].reverse(), { maxCommits: 13 }).size).toBe(2);
    });
});

describe('coChangePartners — partner cap and ranking', () => {
    // Hub h changes with p0..p8 three times each and with p9 four times: ofSelf(h) = 31; 31 fillers make N = 62, so every lift is
    // t·62/(31·t) = 2. Confidence ranks p9 (4/31) first; p0..p8 tie (3/31) on confidence and lift, so the ref breaks the tie.
    const history = [
        ...Array.from({ length: 9 }, (_, i) => times(3, C('h.ts', `p${i}.ts`))).flat(),
        ...times(4, C('h.ts', 'p9.ts')),
        ...fillers(31),
    ];
    const r = coChangePartners(history);

    it('keeps at most 8 partners, ranked confidence desc, lift desc, ref asc', () => {
        const hub = r.get(ref('h.ts'))!;
        expect(hub).toHaveLength(8);
        expect(hub.map((p) => p.ref)).toEqual(['p9', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6'].map((p) => ref(`${p}.ts`)));
        expect(counts([hub[0]!])).toEqual([{ ref: ref('p9.ts'), together: 4, ofSelf: 31, confidence: 0.129032, lift: 2 }]);
        expect(hub[1]!.confidence).toBe(0.096774);
    });

    it('the cap is per file: a spoke still lists the hub', () => {
        expect(counts(r.get(ref('p8.ts')))).toEqual([{ ref: ref('h.ts'), together: 3, ofSelf: 3, confidence: 1, lift: 2 }]);
    });

    it('lift ranks before ref when confidence ties', () => {
        // x with y and w three times each; w also changes alone 3 times, so lift(x,w) < lift(x,y) while P(·|x) ties at 0.5.
        const t = coChangePartners([...times(3, C('x.ts', 'w.ts')), ...times(3, C('x.ts', 'y.ts')), ...times(3, C('w.ts')), ...fillers(30)]);
        // N = 39. lift(x,y) = 3·39/(6·3) = 6.5; lift(x,w) = 3·39/(6·6) = 3.25.
        expect(t.get(ref('x.ts'))!.map((p) => [p.ref, p.confidence, p.lift])).toEqual([[ref('y.ts'), 0.5, 6.5], [ref('w.ts'), 0.5, 3.25]]);
    });

    it('options override the defaults', () => {
        expect(coChangePartners(history, { maxPartners: 2 }).get(ref('h.ts'))).toHaveLength(2);
        expect(coChangePartners(history, { minLift: 2.5 }).size).toBe(0);
    });

    it('shuffled commits and refs give identical output', () => {
        const shuffled = [...history].reverse().map((c) => ({ refs: [...c.refs].reverse() }));
        same([...coChangePartners(shuffled)], [...r]);
        expect([...r.keys()]).toEqual([...r.keys()].sort());
    });
});

describe('co-change badge codec (v1)', () => {
    const partners = coChangePartners([...times(3, C('a.ts', 'b.ts')), ...times(3, C('a.ts', 'c d.ts')), ...fillers(20)]).get(ref('a.ts'))!;
    const { low, high } = partners[0]!;

    it('round-trips refs, counts, the interval and order behind a version line', () => {
        const value = encodeCoChange(partners);
        expect(value).toBe(`v1\n3\t6\t${low}\t${high}\t${ref('b.ts')}\n3\t6\t${low}\t${high}\t${ref('c d.ts')}`);
        expect(decodeCoChange(value)).toEqual(partners.map((p) => ({ ref: p.ref, together: p.together, ofSelf: p.ofSelf, low: p.low, high: p.high })));
    });

    it('an empty list is the bare version line, which decodes to nothing', () => {
        expect(encodeCoChange([])).toBe('v1');
        expect(decodeCoChange('v1')).toEqual([]);
    });

    it.each([
        ['a number', 3], ['undefined', undefined], ['an object', { a: 1 }], ['the empty string', ''],
        ['no version line', `3\t6\t0.2\t0.8\t${ref('b.ts')}`], ['an unknown version', `v2\n3\t6\t0.2\t0.8\t${ref('b.ts')}`],
        ['a non-numeric count', `v1\nx\t6\t0.2\t0.8\t${ref('b.ts')}`], ['a missing ref', 'v1\n3\t6\t0.2\t0.8\t'],
        ['the old three-field line', `v1\n3\t6\t${ref('b.ts')}`],
        ['together above ofSelf', `v1\n7\t6\t0.2\t0.8\t${ref('b.ts')}`], ['together 0', `v1\n0\t6\t0\t0.3\t${ref('b.ts')}`],
        ['ofSelf 0', `v1\n0\t0\t0\t1\t${ref('b.ts')}`], ['low above high', `v1\n3\t6\t0.9\t0.8\t${ref('b.ts')}`],
        ['high above 1', `v1\n3\t6\t0.2\t1.5\t${ref('b.ts')}`], ['one bad line among good ones', `v1\n3\t6\t0.2\t0.8\t${ref('b.ts')}\nnope`],
    ])('decodes %s to []', (_name, value) => {
        expect(decodeCoChange(value)).toEqual([]);
    });
});

describe('coChangeRecallAtK', () => {
    // 21 commits by `at`: i % 3 = 0 → {a,b}, 1 → {c,d}, 2 → a filler; commit 18 also touches e. 20% held out = the last 4
    // (17..20). Fit on 0..16: a+b 6 times, c+d 6, 5 fillers, N = 17 (lift 6·17/36 = 2.83). Held-out cases: commit 18 {a,b,e}
    // a → [b] covers 1 of {b,e} = 0.5, b → [a] 0.5, e has no partners (skipped); commit 19 {c,d}: 1 and 1. Mean 0.75 over 4.
    const at = (i: number) => `2026-01-${String(i + 1).padStart(2, '0')}T00:00:00Z`;
    const history: CoChangeCommit[] = Array.from({ length: 21 }, (_, i) => {
        const paths = i % 3 === 0 ? ['a.ts', 'b.ts', ...(i === 18 ? ['e.ts'] : [])] : i % 3 === 1 ? ['c.ts', 'd.ts'] : [`z${i}.ts`];
        return { at: at(i), refs: paths.map(ref) };
    });

    it('is the mean recall over held-out (commit, file) cases, counting the unpredicted ones as skipped', () => {
        expect(coChangeRecallAtK(history)).toEqual({ recall: 0.75, cases: 4, skipped: 1 });
    });

    it('orders by the parsed time, not the input order nor the string', () => {
        expect(coChangeRecallAtK([...history].reverse())).toEqual({ recall: 0.75, cases: 4, skipped: 1 });
        // Hourly instants, the odd ones written at +05:00: as strings they would sort out of order ("T06:00+05:00" after "T02:00Z"),
        // as times they keep the history's order.
        const hourly = history.map((c, i) => {
            const t = Date.UTC(2026, 0, 1) + i * 3_600_000;
            return { ...c, at: i % 2 === 0 ? new Date(t).toISOString() : new Date(t + 5 * 3_600_000).toISOString().replace('Z', '+05:00') };
        });
        expect(hourly[1]!.at > hourly[2]!.at).toBe(true);
        expect(coChangeRecallAtK([...hourly].reverse())).toEqual({ recall: 0.75, cases: 4, skipped: 1 });
    });

    it('undated input is taken oldest first, as given', () => {
        expect(coChangeRecallAtK(history.map(({ refs }) => ({ refs })))).toEqual({ recall: 0.75, cases: 4, skipped: 1 });
    });

    it('is insufficient-data with nothing to hold out or nothing predicted', () => {
        expect(coChangeRecallAtK(history.slice(0, 4))).toEqual({ status: 'insufficient-data' });
        expect(coChangeRecallAtK(fillers(30))).toEqual({ status: 'insufficient-data' });
    });

    it('does not leak: coupling seen only in held-out commits is never predicted', () => {
        // 16 training fillers, then 4 held-out a+b commits. The fit never sees a+b, so no case can be scored.
        expect(coChangeRecallAtK([...fillers(16), ...times(4, C('a.ts', 'b.ts'))])).toEqual({ status: 'insufficient-data' });
    });
});

describe('applyCoChangeHidden / buildGraph', () => {
    const badge = (...paths: string[]) => encodeCoChange(paths.map((p) => ({ ref: ref(p), together: 3, ofSelf: 3, confidence: 1, lift: 2, low: 0.5, high: 1 })));

    it('counts graph-node partners no live reference explains, either direction', () => {
        const nodes = [
            fileNode('a.ts', { badges: { coChange: badge('b.ts', 'c.ts', 'd.ts', 'gone.ts') } }),
            fileNode('b.ts', { badges: { coChange: badge('a.ts') } }),
            fileNode('c.ts', { badges: { coChange: badge('a.ts'), coChangeHidden: 9 } }),
            fileNode('d.ts', { badges: { complexity: 2 } }),
        ];
        // b → a explains a~b both ways; the a → c edge is stale; d has no edge; gone.ts is no node, so it is not counted.
        applyCoChangeHidden(nodes, [edge('b.ts', 'a.ts', 'references'), edge('a.ts', 'c.ts', 'references', { stale: true })]);
        expect(nodes[0]!.badges!['coChangeHidden']).toBe(2);
        expect(nodes[1]!.badges).toEqual({ coChange: badge('a.ts') });
        expect(nodes[2]!.badges!['coChangeHidden']).toBe(1);
        expect(nodes[3]!.badges).toEqual({ complexity: 2 });
    });

    it('is absent (never 0) when every partner is referenced, and on a malformed badge', () => {
        const nodes = [fileNode('a.ts', { badges: { coChange: badge('b.ts') } }), fileNode('b.ts', { badges: { coChange: 'junk' } })];
        applyCoChangeHidden(nodes, [edge('a.ts', 'b.ts', 'links')]);
        expect(nodes.map((n) => n.badges!['coChangeHidden'])).toEqual([undefined, undefined]);
    });

    it('clears a stale coChangeHidden from a file that no longer carries coChange', () => {
        const nodes = [fileNode('a.ts', { badges: { coChangeHidden: 3, complexity: 1 } }), fileNode('b.ts', { badges: { coChangeHidden: 2 } })];
        applyCoChangeHidden(nodes, []);
        expect(nodes[0]!.badges).toEqual({ complexity: 1 });
        expect(nodes[1]!.badges).toBeUndefined();
    });

    it('a symbol-level reference rolls up to its file and explains the coupling', () => {
        const nodes = [fileNode('a.ts', { badges: { coChange: badge('b.ts') } }), fileNode('b.ts')];
        const symbolEdge: KEdge = { from: ref('a.ts#x'), to: ref('b.ts'), kind: 'references', weight: 1, origin: 'extracted' };
        applyCoChangeHidden(nodes, [symbolEdge]);
        expect(nodes[0]!.badges!['coChangeHidden']).toBeUndefined();
    });

    it('an edge to a node that is not a file (an out-of-root placeholder) explains the coupling too', () => {
        const pkg = ref('node_modules/x/index.js');
        const nodes = [
            fileNode('a.ts', { badges: { coChange: encodeCoChange([{ ref: pkg, together: 3, ofSelf: 3, confidence: 1, lift: 2, low: 0.5, high: 1 }]) } }),
            { ...fileNode('node_modules/x/index.js'), kind: 'missing' },
        ];
        applyCoChangeHidden(nodes, [edge('a.ts', 'node_modules/x/index.js', 'references')]);
        expect(nodes[0]!.badges!['coChangeHidden']).toBeUndefined();
        applyCoChangeHidden(nodes, []);
        expect(nodes[0]!.badges!['coChangeHidden']).toBe(1);
    });

    function input(): BuildGraphInput {
        return {
            rootId: 'r', packs: [], facts: [], relations: [],
            extracted: [
                extractedOf(fileNode('a.ts', { badges: { coChange: badge('b.ts', 'c.ts') } }), [edge('a.ts', 'b.ts', 'references')]),
                extractedOf(fileNode('b.ts')),
                extractedOf(fileNode('c.ts', { badges: { coChange: badge('a.ts') } })),
            ],
        };
    }

    it('buildGraph stamps coChangeHidden and the outline carries both badges', () => {
        const g = buildGraph(input());
        const node = (p: string) => g.nodes.find((n) => n.ref === ref(p))!;
        expect(node('a.ts').badges).toMatchObject({ coChange: badge('b.ts', 'c.ts'), coChangeHidden: 1 });
        expect(node('c.ts').badges).toMatchObject({ coChangeHidden: 1 });
        const o = buildOutline(g);
        const file = (p: string) => o.files.find((f) => f.path.endsWith(p))!;
        expect(file('a.ts').badges).toMatchObject({ coChange: badge('b.ts', 'c.ts'), coChangeHidden: 1 });
        expect(decodeCoChange(file('a.ts').badges!['coChange']).map((e) => e.ref)).toEqual([ref('b.ts'), ref('c.ts')]);
    });
});
