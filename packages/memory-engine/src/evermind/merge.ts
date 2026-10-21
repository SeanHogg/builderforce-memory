/**
 * evermind/merge.ts — the conflict resolver for concurrent Evermind learning.
 *
 * Many learners adapt their LOCAL replica of the same model version and each
 * produces a row-sparse weight delta (`diffCheckpoints(base, adapted)`). A single
 * writer folds N such deltas into one canonical update here, then republishes.
 *
 * Merge policy = **FedAvg over contributors**: each delta stores the contributor's
 * ABSOLUTE new values for the elements it touched, so the merged value of an element
 * is the (optionally sample-weighted) MEAN of the contributors that touched it.
 * Elements only one learner touched take that learner's value; non-touchers do NOT
 * vote the base value back in (a lone real update is kept, not diluted toward base).
 *
 * Pure, GPU-free and deterministic — the math lives here, isolated from whatever
 * storage/orchestration the host wraps around it.
 */
import {
  applyCheckpointDiff,
  deserializeRowDelta,
  serializeRowDelta,
  type RowDelta,
} from "../utils/delta.js";
import { verifyCrcTrailer } from "../utils/crc32.js";

export interface MergeResult {
  /** The new canonical checkpoint (base + merged delta), ready to repackage. */
  checkpoint: ArrayBuffer;
  /** How many distinct elements the merged delta touches. */
  mergedRows: number;
  /** How many contributor deltas were folded in. */
  contributors: number;
  /** L2 norm of the ACTUAL weight movement base→merged, (Σ(merged−base)²)^½. 0 when nothing moved. */
  deltaNorm: number;
}

/** Per-element weighted accumulator while merging. */
interface Acc {
  sum: number;
  weight: number;
}

/**
 * Fold contributor weight-deltas (each a serialized RowDelta vs the SAME base) into
 * one merged checkpoint by FedAvg-over-contributors.
 *
 * @param baseCheckpoint the canonical checkpoint the deltas were diffed against
 * @param diffs          serialized RowDelta buffers (from `diffCheckpoints`)
 * @param weights        optional per-diff sample weights (default 1 each = plain mean);
 *                       a zero/negative weight excludes that contributor
 */
export function mergeCheckpointDiffs(
  baseCheckpoint: ArrayBuffer,
  diffs: ArrayBuffer[],
  weights?: number[],
): MergeResult {
  if (diffs.length === 0) {
    return { checkpoint: baseCheckpoint, mergedRows: 0, contributors: 0, deltaNorm: 0 };
  }
  if (weights && weights.length !== diffs.length) {
    throw new Error(`mergeCheckpointDiffs: weights length ${weights.length} != diffs length ${diffs.length}`);
  }

  const parsed: RowDelta[] = diffs.map((d) => deserializeRowDelta(d));
  // `diffCheckpoints` emits element-granular (rowSize 1) deltas, which is what makes
  // element-wise FedAvg exact: a partially-overlapping merge can never zero a column.
  for (const rd of parsed) {
    if (rd.rowSize !== 1) {
      throw new Error(`mergeCheckpointDiffs: expected element-granular deltas (rowSize 1), got ${rd.rowSize}`);
    }
  }

  const acc = new Map<number, Acc>();
  parsed.forEach((rd, k) => {
    const w = weights ? weights[k]! : 1;
    if (!(w > 0)) return;
    rd.rows.forEach((idx, i) => {
      const value = rd.data[i]!;
      const prev = acc.get(idx);
      if (prev) { prev.sum += w * value; prev.weight += w; }
      else acc.set(idx, { sum: w * value, weight: w });
    });
  });

  if (acc.size === 0) {
    return { checkpoint: baseCheckpoint, mergedRows: 0, contributors: parsed.length, deltaNorm: 0 };
  }

  const rows = [...acc.keys()].sort((a, b) => a - b);
  const data = new Float32Array(rows.length);
  rows.forEach((idx, i) => {
    const a = acc.get(idx)!;
    data[i] = a.sum / a.weight;
  });

  // The merged delta stores ABSOLUTE values, so movement is (merged − base) per element.
  const baseBody = new Float32Array(verifyCrcTrailer(baseCheckpoint).body);
  let sumSq = 0;
  rows.forEach((idx, i) => {
    const moved = data[i]! - (baseBody[idx] ?? 0);
    sumSq += moved * moved;
  });

  const merged: RowDelta = { rowSize: 1, rows, data };
  const checkpoint = applyCheckpointDiff(baseCheckpoint, serializeRowDelta(merged));
  return { checkpoint, mergedRows: rows.length, contributors: parsed.length, deltaNorm: Math.sqrt(sumSq) };
}

/** A merge that passed (or failed) its held-out regression gate. */
export interface GatedMergeResult extends MergeResult {
  /** True when the merged checkpoint was kept. When false, `checkpoint` is the BASE. */
  accepted: boolean;
  /** Held-out loss of the base / the merged candidate (null when unscorable). */
  baseLoss: number | null;
  newLoss: number | null;
  /** Why the merge was refused. */
  refusedReason?: "nothing-merged" | "regressed" | "unscored";
}

export interface MergeGateOptions {
  /** Loss increase still accepted (absolute). Default 0 — any regression refuses. */
  tolerance?: number;
  /** Refuse when either side cannot be scored. Default false (an unscored merge is kept). */
  requireEval?: boolean;
}

/**
 * {@link mergeCheckpointDiffs}, then the pre/post held-out check that LM merges
 * record (`meanEvalLoss` of the previous vs the merged version) — enforced: a merge
 * whose held-out loss rises by more than `tolerance` is refused and the base is
 * returned unchanged. `score` maps a checkpoint to its held-out loss (lower is
 * better); for an LM that is `meanEvalLoss` over the reloaded model, for a trained
 * ML head its task's eval loss. Still the ONE merge — this only gates it.
 */
export function mergeCheckpointDiffsGated(
  baseCheckpoint: ArrayBuffer,
  diffs: ArrayBuffer[],
  weights: number[] | undefined,
  score: (checkpoint: ArrayBuffer) => number | null,
  opts: MergeGateOptions = {},
): GatedMergeResult {
  const merged = mergeCheckpointDiffs(baseCheckpoint, diffs, weights);
  const refuse = (refusedReason: GatedMergeResult["refusedReason"], baseLoss: number | null, newLoss: number | null): GatedMergeResult => ({
    ...merged,
    checkpoint: baseCheckpoint,
    deltaNorm: 0,
    accepted: false,
    baseLoss,
    newLoss,
    refusedReason,
  });
  if (merged.mergedRows === 0) return refuse("nothing-merged", null, null);
  const baseLoss = score(baseCheckpoint);
  const newLoss = score(merged.checkpoint);
  if (baseLoss == null || newLoss == null) {
    return opts.requireEval ? refuse("unscored", baseLoss, newLoss) : { ...merged, accepted: true, baseLoss, newLoss };
  }
  if (newLoss > baseLoss + (opts.tolerance ?? 0)) return refuse("regressed", baseLoss, newLoss);
  return { ...merged, accepted: true, baseLoss, newLoss };
}
