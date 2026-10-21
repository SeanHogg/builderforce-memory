/**
 * evermind/delta_wire.ts — the wire contract of a pre-diffed weight-delta contribution.
 *
 * A learner with CPU to spare adapts a private copy of a model version, diffs it
 * against that version (`diffCheckpoints`) and ships only the sparse diff; the single
 * writer then merges it (`mergeCheckpointDiffs`) instead of running the fit.
 * Everything the PRODUCER and the WRITER must agree on lives here, once:
 *   - the payload shape (`diff` base64 + integer `baseVersion`, optional weight/label,
 *     optional `target` naming WHICH checkpoint the diff is against — the model's own
 *     weights, or one trained ML head carried in the manifest's `mlHeads`);
 *   - which checkpoint a door merges into: a parse names its door's target (the model
 *     by default), so an ML-head diff can never reach the model's own merge — it would
 *     pass every structural check and overwrite the first n floats of the LM weights;
 *   - the size cap, so a producer can fall back before it is refused;
 *   - the structural check a delta must pass before it may join a merge batch.
 *
 * That last one is load-bearing: `mergeCheckpointDiffs` throws on a malformed diff,
 * and one throw drops the whole batch. A single bad push must cost only itself.
 */
import { deserializeRowDelta } from "../utils/delta.js";
import { verifyCrcTrailer } from "../utils/crc32.js";
import { base64ToBytes, bytesToBase64 } from "../utils/base64.js";
import { ML_TASK_ID_RE } from "./ml_head.js";

/** Max accepted base64 delta, in characters (~8 MiB). */
export const MAX_DELTA_B64_CHARS = 8 * 1024 * 1024;

/** Chars of provenance label kept — a diff carries no text, so the label is its only description. */
export const DELTA_LABEL_MAX_CHARS = 800;

/** A delta's target: absent = the model's own checkpoint; `ml:<task id>` = that trained ML head. */
export const DELTA_TARGET_RE = new RegExp(`^ml:${ML_TASK_ID_RE.source.slice(1)}`);

/** The delta target of the trained head for catalog task `taskId`. */
export function mlHeadTarget(taskId: string): string {
  return `ml:${taskId}`;
}

/**
 * Why a delta's `target` does not belong at a door that merges into `expected`
 * (undefined = the model's own checkpoint), or null when it does. The ONE target
 * check: the parse and the ML-head merge both use it.
 */
export function deltaTargetMismatch(target: string | undefined, expected: string | undefined): string | null {
  if (target === expected) return null;
  return `target ${target ?? "(model)"} does not belong here; this door merges into ${expected ?? "the model checkpoint"}`;
}

/** Standard base64 alphabet with optional padding. */
const BASE64_RE = /^[A-Za-z0-9+/]+={0,2}$/;

/** A validated delta contribution — the exact body a producer sends. */
export interface DeltaLearnPayload {
  /** base64 serialized RowDelta (`diffCheckpoints(base, adapted)`). */
  diff: string;
  /** The model version the delta was diffed against. */
  baseVersion: number;
  /** FedAvg sample weight (> 0). Absent means the writer's default (1). */
  weight?: number;
  /** Provenance for the merged row (e.g. the run's task). */
  label?: string;
  /**
   * Which checkpoint the diff is against. Absent = the model's own weights (every
   * delta written before heads federated); `ml:<task id>` = that trained ML head.
   */
  target?: string;
}

/** Why a body was refused: `invalid` (malformed) or `too-large` (over {@link MAX_DELTA_B64_CHARS}). */
export type DeltaParseResult =
  | { ok: true; payload: DeltaLearnPayload }
  | { ok: false; reason: "invalid" | "too-large"; error: string };

/** Which checkpoint the door calling {@link parseDeltaLearnPayload} merges into. */
export interface DeltaParseOptions {
  /** Absent = the model's own checkpoint (the LM door); `ml:<task id>` = that head's door. */
  target?: string;
}

/**
 * Parse + validate a delta body. Pure — a front door calls it to refuse a bad body
 * early, and the single writer calls it again because it trusts no caller.
 *
 * A body whose `target` is not the door's (`opts.target`; absent = the model) is
 * refused as `invalid`: the LM door never admits an ML-head delta, and a head's door
 * never admits a model delta.
 */
export function parseDeltaLearnPayload(raw: unknown, opts: DeltaParseOptions = {}): DeltaParseResult {
  const body = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const diff = typeof body.diff === "string" ? body.diff : "";
  const baseVersion = typeof body.baseVersion === "number" ? body.baseVersion : Number.NaN;
  if (!diff || !Number.isInteger(baseVersion)) {
    return {
      ok: false,
      reason: "invalid",
      error: "diff (base64) and baseVersion (integer head version the delta was taken against) are required",
    };
  }
  if (diff.length > MAX_DELTA_B64_CHARS) {
    return { ok: false, reason: "too-large", error: `delta too large (max ${MAX_DELTA_B64_CHARS} base64 characters)` };
  }
  if (diff.length % 4 !== 0 || !BASE64_RE.test(diff)) {
    return { ok: false, reason: "invalid", error: "diff must be standard base64" };
  }
  const weight = typeof body.weight === "number" && Number.isFinite(body.weight) && body.weight > 0 ? body.weight : undefined;
  const label = typeof body.label === "string" && body.label.trim() ? body.label.trim().slice(0, DELTA_LABEL_MAX_CHARS) : undefined;
  if (body.target !== undefined && (typeof body.target !== "string" || !DELTA_TARGET_RE.test(body.target))) {
    return { ok: false, reason: "invalid", error: "target, when present, must be ml:<task id>" };
  }
  const target = body.target as string | undefined;
  const mismatch = deltaTargetMismatch(target, opts.target);
  if (mismatch) return { ok: false, reason: "invalid", error: mismatch };
  return {
    ok: true,
    payload: {
      diff,
      baseVersion,
      ...(weight !== undefined ? { weight } : {}),
      ...(label ? { label } : {}),
      ...(target ? { target } : {}),
    },
  };
}

/**
 * Build the payload for one computed diff. The caller owns the size guard (compare
 * `payload.diff.length` with {@link MAX_DELTA_B64_CHARS}) so it can choose a fallback.
 */
export function buildDeltaLearnPayload(
  diff: ArrayBuffer,
  baseVersion: number,
  weight?: number,
  label?: string,
  target?: string,
): DeltaLearnPayload {
  const trimmed = (label ?? "").trim();
  if (target !== undefined && !DELTA_TARGET_RE.test(target)) throw new Error(`buildDeltaLearnPayload: bad target '${target}'`);
  return {
    diff: bytesToBase64(diff),
    baseVersion,
    ...(typeof weight === "number" && Number.isFinite(weight) && weight > 0 ? { weight } : {}),
    ...(trimmed ? { label: trimmed.slice(0, DELTA_LABEL_MAX_CHARS) } : {}),
    ...(target ? { target } : {}),
  };
}

/** Decode a payload's base64 `diff` to the serialized RowDelta bytes. Throws on malformed base64. */
export function decodeDeltaPayload(payload: Pick<DeltaLearnPayload, "diff">): ArrayBuffer {
  return base64ToBytes(payload.diff);
}

/**
 * Structural check of ONE delta against the base it claims to diff: it must
 * deserialize, be element-granular, index only elements the base has, and carry
 * finite values. Returns why it is unusable, or null when it can join a merge.
 */
export function deltaUnusableReason(delta: ArrayBuffer, baseCheckpoint: ArrayBuffer): string | null {
  try {
    const rd = deserializeRowDelta(delta);
    if (rd.rowSize !== 1) return `expected an element-granular delta (rowSize 1), got ${rd.rowSize}`;
    const elements = verifyCrcTrailer(baseCheckpoint).body.byteLength / 4;
    for (let i = 0; i < rd.rows.length; i++) {
      const idx = rd.rows[i]!;
      if (!Number.isInteger(idx) || idx < 0 || idx >= elements) {
        return `row index ${idx} is outside the base checkpoint (${elements} elements)`;
      }
      if (!Number.isFinite(rd.data[i]!)) return `non-finite value at row ${idx}`;
    }
    return null;
  } catch (error) {
    return `not a serialized RowDelta: ${error instanceof Error ? error.message : String(error)}`;
  }
}
