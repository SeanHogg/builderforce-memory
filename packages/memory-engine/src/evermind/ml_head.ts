/**
 * evermind/ml_head.ts — what the `.evermind` artifact needs to know about a trained
 * ML-catalog head: its manifest entry shape, its checkpoint format and its structural
 * check, plus the task-id grammar the delta wire's `ml:<task>` target uses.
 *
 * A leaf: it imports only utils/. `evermind/package.ts` and `evermind/delta_wire.ts`
 * import it, and so does `ml/` (which re-exports it) — so the dependency runs one way,
 * ml → evermind, never back.
 */
import { base64ToBytes } from "../utils/base64.js";
import { appendCrcTrailer, verifyCrcTrailer } from "../utils/crc32.js";

/** Task ids: lowercase dotted segments, e.g. `run.local-success`. */
export const ML_TASK_ID_RE = /^[a-z][a-z0-9-]*(\.[a-z][a-z0-9-]*)+$/;

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
