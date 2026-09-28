/**
 * tests/weightedFusion.test.ts
 * Score fusion (moved here from agent-tools), plus the RRF term, text MMR and
 * recency decay that the hosts rank with.
 */

import {
  DEFAULT_HYBRID_TEXT_WEIGHT,
  DEFAULT_HYBRID_VECTOR_WEIGHT,
  bm25RankToScore,
  fuseHybridArms,
  hybridScore,
  normalizeHybridWeights,
} from "../src/retrieval/weightedFusion.js";
import { rrfTerm, reciprocalRankFusion } from "../src/retrieval/fusion.js";
import { mmrSelect, textMaximalMarginalRelevance } from "../src/retrieval/mmr.js";
import { applyTemporalDecayToScore, calculateTemporalDecayMultiplier, toDecayLambda } from "../src/retrieval/decay.js";

/**
 * The shared fusion both retrieval surfaces rank with. These pin the arithmetic
 * itself, because the cloud and on-prem arms are different engines and the ONLY
 * thing making their orderings comparable is that this formula is the same.
 */
describe("normalizeHybridWeights", () => {
  it("defaults to the documented 0.7 / 0.3 split", () => {
    expect(normalizeHybridWeights()).toEqual({
      vectorWeight: DEFAULT_HYBRID_VECTOR_WEIGHT,
      textWeight: DEFAULT_HYBRID_TEXT_WEIGHT,
    });
  });

  it("renormalises a pair that does not sum to one", () => {
    expect(normalizeHybridWeights(3, 1)).toEqual({ vectorWeight: 0.75, textWeight: 0.25 });
  });

  it("falls back to the defaults for a degenerate pair", () => {
    expect(normalizeHybridWeights(0, 0)).toEqual({ vectorWeight: 0.7, textWeight: 0.3 });
    expect(normalizeHybridWeights(Number.NaN, undefined)).toEqual({ vectorWeight: 0.7, textWeight: 0.3 });
  });
});

describe("hybridScore", () => {
  it("weights the two arms and treats a missing arm as zero, not as a drop", () => {
    const w = normalizeHybridWeights();
    expect(hybridScore({ vectorScore: 1, textScore: 1 }, w)).toBeCloseTo(1);
    expect(hybridScore({ vectorScore: 1 }, w)).toBeCloseTo(0.7);
    expect(hybridScore({ textScore: 1 }, w)).toBeCloseTo(0.3);
  });
});

describe("bm25RankToScore", () => {
  it("maps a better (lower) rank to a higher score inside (0,1]", () => {
    expect(bm25RankToScore(0)).toBe(1);
    expect(bm25RankToScore(1)).toBe(0.5);
    expect(bm25RankToScore(1)).toBeGreaterThan(bm25RankToScore(9));
    expect(bm25RankToScore(Number.NaN)).toBeCloseTo(1 / 1000);
  });
});

describe("fuseHybridArms", () => {
  it("unions both arms, scores each once, and orders best-first", () => {
    const out = fuseHybridArms({
      vector: [{ id: 'a', text: 'semantic only' }, { id: 'b', text: 'both' }],
      text: [{ id: 'b', text: 'both' }, { id: 'c', text: 'lexical only' }],
      vectorScoreOf: (r) => (r.id === 'b' ? 0.8 : 0.9),
      textScoreOf: (r) => (r.id === 'b' ? 0.9 : 1),
      merge: (row, score) => ({ id: row.id, score }),
    });
    expect(out.map((r) => r.id)).toEqual(['b', 'a', 'c']);
    // b: .7*.8 + .3*.9 = .83 · a: .7*.9 = .63 · c: .3*1 = .3
    expect(out[0]?.score).toBeCloseTo(0.83);
    expect(out[2]?.score).toBeCloseTo(0.3);
  });

  it("keeps a candidate found by only one arm", () => {
    const out = fuseHybridArms({
      vector: [],
      text: [{ id: 'only' }],
      vectorScoreOf: () => 0,
      textScoreOf: () => 0.5,
      merge: (row, score) => ({ id: row.id, score }),
    });
    expect(out).toHaveLength(1);
    expect(out[0]?.score).toBeCloseTo(0.15);
  });
});

describe("rrfTerm", () => {
  it("is the per-list RRF contribution reciprocalRankFusion sums", () => {
    expect(rrfTerm(0)).toBeCloseTo(1 / 61, 12);
    expect(rrfTerm(4, 10)).toBeCloseTo(1 / 15, 12);
    const [top] = reciprocalRankFusion([{ ids: ["a", "b"] }]);
    expect(top).toEqual({ id: "a", score: rrfTerm(0) });
  });
});

describe("textMaximalMarginalRelevance", () => {
  const items = [
    { id: "a", score: 1.0, content: "deploy the api worker to cloudflare" },
    { id: "b", score: 0.95, content: "deploy the api worker to cloudflare today" },
    { id: "c", score: 0.5, content: "rotate the stripe webhook secret" },
  ];

  it("keeps the most relevant first and pushes a near-duplicate below a distinct item", () => {
    expect(textMaximalMarginalRelevance(items, 0.3).map((i) => i.id)).toEqual(["a", "c", "b"]);
    // At 0.5 relevance still outweighs the overlap, so the near-duplicate keeps its place.
    expect(textMaximalMarginalRelevance(items, 0.5).map((i) => i.id)).toEqual(["a", "b", "c"]);
  });

  it("is a plain score sort at lambda 1 and returns every item", () => {
    expect(textMaximalMarginalRelevance(items, 1).map((i) => i.id)).toEqual(["a", "b", "c"]);
    expect(textMaximalMarginalRelevance([], 0.5)).toEqual([]);
  });

  it("breaks an exact MMR tie on the raw score, then on input order", () => {
    // Scores 3 and 1 normalise to 1 and 0; with lambda 0.5 and identical text the
    // second pick is forced, so check the first: equal raw scores keep input order.
    const tied = [
      { id: "first", score: 1, content: "x" },
      { id: "second", score: 1, content: "x" },
    ];
    expect(textMaximalMarginalRelevance(tied, 0.5)[0]!.id).toBe("first");
    const tieScore = mmrSelect(["p", "q"], { relevance: () => 1, similarity: () => 0, tieScore: (s) => (s === "q" ? 2 : 1) });
    expect(tieScore).toEqual(["q", "p"]);
  });
});

describe("mmrSelect", () => {
  it("stops at topK and never picks the same candidate twice", () => {
    const picked = mmrSelect([1, 2, 3, 4], { relevance: (n) => n, similarity: () => 0, topK: 2 });
    expect(picked).toEqual([4, 3]);
  });
});

describe("recency decay", () => {
  it("halves a score at one half-life and leaves it alone when disabled", () => {
    expect(toDecayLambda(30)).toBeCloseTo(Math.LN2 / 30, 12);
    expect(calculateTemporalDecayMultiplier({ ageInDays: 30, halfLifeDays: 30 })).toBeCloseTo(0.5, 12);
    expect(applyTemporalDecayToScore({ score: 2, ageInDays: 60, halfLifeDays: 30 })).toBeCloseTo(0.5, 12);
    expect(calculateTemporalDecayMultiplier({ ageInDays: 10, halfLifeDays: 0 })).toBe(1);
    expect(calculateTemporalDecayMultiplier({ ageInDays: -5, halfLifeDays: 30 })).toBe(1);
  });
});
