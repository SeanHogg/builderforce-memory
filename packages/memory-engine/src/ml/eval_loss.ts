/**
 * ml/eval_loss.ts — the ONE held-out loss for catalog models.
 *
 * What the trainer reports as `holdoutLoss`, what a federated merge is gated on, and
 * what `LogisticModel.loss` returns. A leaf (types only), so a classical algorithm can
 * use it without importing the trainer that imports the algorithms.
 */
import type { MlPredictor, MlSample, MlTaskKind } from "./types.js";

/**
 * Mean loss of `predictor` on `samples` for a task of `kind` — classify: weighted
 * cross-entropy of the true class; regress: weighted ½·squared error. Null for
 * kinds without a supervised loss, or an empty (or zero-weight) set.
 */
export function mlEvalLoss(kind: MlTaskKind, predictor: MlPredictor, samples: readonly MlSample[]): number | null {
  if (samples.length === 0 || (kind !== "classify" && kind !== "regress")) return null;
  let sum = 0;
  let wSum = 0;
  for (const s of samples) {
    const w = s.weight ?? 1;
    const p = predictor.predict(s.x);
    const loss = kind === "classify" ? -Math.log(Math.max(p.scores[s.y] ?? 0, 1e-12)) : 0.5 * (p.scores[0]! - s.y) ** 2;
    sum += w * loss;
    wSum += w;
  }
  return wSum > 0 ? sum / wSum : null;
}
