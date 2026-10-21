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
 */
import { buildDeltaLearnPayload, decodeDeltaPayload, deltaUnusableReason, mlHeadTarget, type DeltaLearnPayload } from "../evermind/delta_wire.js";
import { mergeCheckpointDiffsGated, type GatedMergeResult, type MergeGateOptions } from "../evermind/merge.js";
import type { EvermindModelPackage } from "../evermind/package.js";
import { base64ToBytes, bytesToBase64 } from "../utils/base64.js";
import { appendCrcTrailer, verifyCrcTrailer } from "../utils/crc32.js";
import { diffCheckpoints } from "../utils/delta.js";

/**
 * One trained ML-catalog head carried in a model's manifest (`mlHeads`): small, so
 * it rides in the manifest, and versioned, so it federates — a delta against it is
 * `target: "ml:<task>"`, `baseVersion: version` on the delta wire.
 */
export interface EvermindMlHead {
  /** Catalog task id, e.g. `run.local-success`. */
  task: string;
  /** Engine key, e.g. `local:classical:logistic`. */
  engine: string;
  /** Integer head version — what a federated delta for it is diffed against. */
  version: number;
  /** What a loader needs to rebuild the predictor (e.g. `{ dim, classes }`). */
  shape: Record<string, number>;
  /** base64 of the CRC-trailed f32 checkpoint of the head's flat parameters. */
  params: string;
  /** Held-out loss of this version, when measured. */
  evalLoss?: number;
}

/** A head's flat parameters as a CRC-trailed f32 checkpoint. */
export function headCheckpoint(params: ArrayLike<number>): ArrayBuffer {
  const body = new ArrayBuffer(params.length * 4);
  new Float32Array(body).set(Array.from(params));
  return appendCrcTrailer(body);
}

/** The flat parameters of a head checkpoint. Throws on a failed CRC. */
export function headParams(checkpoint: ArrayBuffer): Float32Array {
  const crc = verifyCrcTrailer(checkpoint);
  if (!crc.ok) throw new Error("headParams: checkpoint failed its CRC check");
  if (crc.body.byteLength % 4 !== 0) throw new Error("headParams: checkpoint is not f32");
  return new Float32Array(crc.body);
}

/** Why a manifest head entry is unusable, or null (the package's validate() calls this). */
export function mlHeadUnusableReason(h: EvermindMlHead): string | null {
  if (!h || typeof h.task !== "string" || typeof h.engine !== "string") return "task and engine are required";
  if (!Number.isInteger(h.version) || h.version < 0) return "version must be a non-negative integer";
  try {
    const crc = verifyCrcTrailer(base64ToBytes(h.params));
    if (!crc.hasTrailer || !crc.ok) return "params checkpoint failed its CRC check";
    if (crc.body.byteLength % 4 !== 0) return "params checkpoint is not f32";
  } catch {
    return "params is not base64";
  }
  return null;
}

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
    if (p.target !== target) return void rejected.push({ index, reason: `target ${p.target ?? "(model)"} is not ${target}` });
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

  const merge = mergeCheckpointDiffsGated(baseCp, diffs, weights, (cp) => score(headParams(cp)), opts);
  if (!merge.accepted) return { head: base, merge, rejected };
  const head: EvermindMlHead = {
    ...base,
    version: base.version + 1,
    params: bytesToBase64(merge.checkpoint),
    ...(merge.newLoss != null ? { evalLoss: merge.newLoss } : {}),
  };
  return { head, merge, rejected };
}
