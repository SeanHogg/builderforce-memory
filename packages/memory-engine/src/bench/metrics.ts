/**
 * bench/metrics.ts — pure scoring primitives over per-position logits.
 *
 * These are the building blocks the harness composes. They are exact, allocation
 * -light, and numerically stable (cross-entropy reuses the engine's stable
 * log-sum-exp). Nothing here trains or mutates the model — benchmarking is a
 * read-only forward pass over held-out data.
 */

import { crossEntropyLoss } from "../training/autograd.js";

/** Natural log of 2 — used to convert nats → bits. */
export const LN2 = Math.LN2;

/**
 * Indices of the `k` largest logits, descending. O(V·k) — fine for the small
 * top-k used in accuracy scoring; avoids a full sort of the vocab per position.
 */
export function topKIndices(logits: Float32Array, k: number): number[] {
  const n = logits.length;
  const kk = Math.max(1, Math.min(k, n));
  const idx: number[] = [];
  const used = new Uint8Array(n);
  for (let s = 0; s < kk; s++) {
    let best = -1;
    let bestVal = -Infinity;
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      const v = logits[i]!;
      if (v > bestVal) {
        bestVal = v;
        best = i;
      }
    }
    if (best < 0) break;
    used[best] = 1;
    idx.push(best);
  }
  return idx;
}

/** Index of the single largest logit (argmax). */
export function argmax(logits: Float32Array): number {
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i]! > bestVal) {
      bestVal = logits[i]!;
      best = i;
    }
  }
  return best;
}

/** Perplexity from a mean cross-entropy (nats). */
export function perplexity(meanCrossEntropy: number): number {
  return Math.exp(meanCrossEntropy);
}

/** Bits-per-token from a mean cross-entropy (nats). */
export function bitsPerToken(meanCrossEntropy: number): number {
  return meanCrossEntropy / LN2;
}

/** Accumulator over predicted positions — cross-entropy + top-1/top-k hits. */
export interface ScoreAccumulator {
  tokens: number;
  ceSum: number;
  top1Hits: number;
  topKHits: number;
}

export function newAccumulator(): ScoreAccumulator {
  return { tokens: 0, ceSum: 0, top1Hits: 0, topKHits: 0 };
}

/**
 * Score one (logits, targets) pair into the accumulator. `logits[t]` predicts
 * `targets[t]`; positions with an out-of-range target are skipped. The caller
 * aligns logits/targets (next-token: logits[0..T-2] predict tokens[1..T-1]).
 */
export function scoreInto(
  acc: ScoreAccumulator,
  logits: Float32Array[],
  targets: number[],
  k: number,
): void {
  const n = Math.min(logits.length, targets.length);
  for (let t = 0; t < n; t++) {
    const target = targets[t]!;
    const lg = logits[t]!;
    if (target < 0 || target >= lg.length) continue;
    acc.ceSum += crossEntropyLoss(lg, target);
    if (argmax(lg) === target) acc.top1Hits++;
    if (topKIndices(lg, k).includes(target)) acc.topKHits++;
    acc.tokens++;
  }
}
