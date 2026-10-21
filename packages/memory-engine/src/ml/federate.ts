/**
 * ml/federate.ts — trained ML heads federate like the model's own weights.
 *
 * A head's trained state is ONE flat f32 vector (`MlPredictor.exportParams`),
 * stored as a CRC-trailed checkpoint so it is the same shape the delta pipeline
 * already handles: a learner diffs its trained copy against the version it
 * started from (`diffCheckpoints`), ships it on the existing delta wire with
 * `target: "ml:<task>"`, and the writer folds N of them with the existing FedAvg
 * (`mergeCheckpointDiffsGated` → `mergeCheckpointDiffs`), refusing a merge whose
 * held-out loss regresses. No second wire, no second merge.
 *
 * The head's manifest entry and checkpoint format live with the artifact
 * (`evermind/ml_head.ts`); this module only imports evermind/, never the reverse.
 */
import {
  buildDeltaLearnPayload,
  decodeDeltaPayload,
  deltaTargetMismatch,
  deltaUnusableReason,
  mlHeadTarget,
  type DeltaLearnPayload,
} from "../evermind/delta_wire.js";
import { mergeCheckpointDiffsGated, type GatedMergeResult, type MergeGateOptions } from "../evermind/merge.js";
import { headCheckpoint, headParams, type EvermindMlHead } from "../evermind/ml_head.js";
import type { EvermindModelPackage } from "../evermind/package.js";
import { base64ToBytes, bytesToBase64 } from "../utils/base64.js";
import { diffCheckpoints } from "../utils/delta.js";

export { headCheckpoint, headParams, mlHeadUnusableReason } from "../evermind/ml_head.js";
export type { EvermindMlHead } from "../evermind/ml_head.js";

/** A manifest entry for a head's current parameters. */
export function mlHeadEntry(
  task: string,
  engine: string,
  version: number,
  params: ArrayLike<number>,
  shape: Record<string, number> = {},
  evalLoss?: number,
): EvermindMlHead {
  return {
    task,
    engine,
    version,
    shape,
    params: bytesToBase64(headCheckpoint(params)),
    ...(evalLoss != null && Number.isFinite(evalLoss) ? { evalLoss } : {}),
  };
}

/** The entry's flat parameters. */
export function mlHeadParams(entry: EvermindMlHead): Float32Array {
  return headParams(base64ToBytes(entry.params));
}

/** The head a package carries for `task`, if any. */
export function packageMlHead(pkg: EvermindModelPackage, task: string): EvermindMlHead | undefined {
  return pkg.manifest.mlHeads?.find((h) => h.task === task);
}

/** `pkg` with `entry` as its head for `entry.task` (replacing the old one). */
export function withPackageMlHead(pkg: EvermindModelPackage, entry: EvermindMlHead): EvermindModelPackage {
  return pkg.withMlHeads([...(pkg.manifest.mlHeads ?? []).filter((h) => h.task !== entry.task), entry]);
}

/**
 * The delta-wire payload for a locally trained head: its new parameters diffed
 * against the version it started from. The caller owns the size guard, as for LM deltas.
 */
export function buildMlHeadDelta(base: EvermindMlHead, trained: ArrayLike<number>, weight?: number, label?: string): DeltaLearnPayload {
  const diff = diffCheckpoints(base64ToBytes(base.params), headCheckpoint(trained));
  return buildDeltaLearnPayload(diff, base.version, weight, label ?? `ml head ${base.task}`, mlHeadTarget(base.task));
}

export interface MlHeadMergeOutcome {
  /** The head after the round: version + 1 when accepted, the base unchanged when refused. */
  head: EvermindMlHead;
  merge: GatedMergeResult;
  /** Contributions left out of the batch, by index, with why. */
  rejected: { index: number; reason: string }[];
}

/**
 * One federated round for a head: validate each contribution ON ITS OWN (wrong
 * target, stale base or a malformed diff costs only itself), FedAvg the rest, and
 * keep the result only if `score` (held-out loss of a parameter vector, lower is
 * better) does not regress.
 *
 * Fails CLOSED: `requireEval` defaults to true here, so a round `score` cannot judge
 * (null — e.g. `mlEvalLoss` for an anomaly/cluster head or an empty holdout — or a
 * non-finite loss) is refused. A caller that federates an unscorable head on purpose
 * passes `{ requireEval: false }` explicitly.
 */
export function mergeMlHeadDeltas(
  base: EvermindMlHead,
  payloads: readonly DeltaLearnPayload[],
  score: (params: Float32Array) => number | null,
  opts: MergeGateOptions = {},
): MlHeadMergeOutcome {
  const baseCp = base64ToBytes(base.params);
  const target = mlHeadTarget(base.task);
  const diffs: ArrayBuffer[] = [];
  const weights: number[] = [];
  const rejected: { index: number; reason: string }[] = [];
  payloads.forEach((p, index) => {
    const wrongTarget = deltaTargetMismatch(p.target, target);
    if (wrongTarget) return void rejected.push({ index, reason: wrongTarget });
    if (p.baseVersion !== base.version) return void rejected.push({ index, reason: `stale base version ${p.baseVersion} (head is ${base.version})` });
    let diff: ArrayBuffer;
    try {
      diff = decodeDeltaPayload(p);
    } catch (error) {
      return void rejected.push({ index, reason: `diff is not base64: ${error instanceof Error ? error.message : String(error)}` });
    }
    const why = deltaUnusableReason(diff, baseCp);
    if (why) return void rejected.push({ index, reason: why });
    diffs.push(diff);
    weights.push(p.weight ?? 1);
  });

  const merge = mergeCheckpointDiffsGated(baseCp, diffs, weights, (cp) => score(headParams(cp)), { ...opts, requireEval: opts.requireEval ?? true });
  if (!merge.accepted) return { head: base, merge, rejected };
  const head: EvermindMlHead = {
    ...base,
    version: base.version + 1,
    params: bytesToBase64(merge.checkpoint),
    ...(merge.newLoss != null ? { evalLoss: merge.newLoss } : {}),
  };
  return { head, merge, rejected };
}
