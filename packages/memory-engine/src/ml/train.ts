/**
 * ml/train.ts — the ONE trainer seam for catalog tasks.
 *
 * Classical engines fit in closed form or by small iteration (`classical/`);
 * head engines train a {@link DenseHead} through the shared {@link HeadTrainer}
 * (GPU AdamW when a device is supplied, CPU AdamW otherwise). A task can name its
 * own composite fit (`TrainSpec.fit`), or say that something else trains it
 * (`TrainSpec.trainedBy`, e.g. `LimbicTrainer` for `limbic.affect`).
 *
 * Held-out loss is measured the same way everywhere ({@link mlEvalLoss}) — it is
 * also what gates a federated merge.
 */
import { fitClassical, type ClassicalOptions } from "./classical/index.js";
import { DenseHead } from "./dense_head.js";
import { extractFeatureMatrix, type TextEmbedder } from "./features.js";
import { HeadTrainer, type HeadTrainOptions } from "./head_trainer.js";
import type { MlModelStore } from "./run.js";
import { engineKey, type MlEngineRef, type MlPredictor, type MlSample, type MlTask, type MlTaskKind } from "./types.js";

/** One labelled example in input space. */
export interface MlTrainExample<I> {
  input: I;
  /** classify: class index; regress: target value; cluster/anomaly: ignored (use 0). */
  label: number;
  weight?: number;
}

export interface MlTrainOptions {
  /** Engine to fit (default: the task's first trainable local engine). */
  engine?: MlEngineRef;
  embedder?: TextEmbedder;
  device?: GPUDevice | null;
  classical?: ClassicalOptions;
  head?: { hiddenDim?: number; seed?: number; train?: HeadTrainOptions };
  /** Overrides the task's `holdoutFraction`. 0 disables the held-out measurement. */
  holdoutFraction?: number;
  /** After measuring on the held-out tail, refit on everything. Default true. */
  refitOnAll?: boolean;
  /** Store the fitted predictor here under (task id, engine key). */
  models?: MlModelStore;
}

export interface MlTrainResult {
  predictor: MlPredictor;
  engine: MlEngineRef;
  /** Examples the final fit saw. */
  samples: number;
  /** Size of the held-out tail, and the model's loss on it (null when none was held out). */
  holdout: number;
  holdoutLoss: number | null;
}

/**
 * Mean loss of `predictor` on `samples` for a task of `kind` — classify: weighted
 * cross-entropy of the true class; regress: weighted ½·squared error. Null for
 * kinds without a supervised loss, or an empty set.
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

/** The task's first local engine the seam can train. */
function defaultEngine(task: MlTask<any, any>): MlEngineRef {
  const e = task.engines.find((r) => r.tier === "local" && (r.engine === "classical" || r.engine === "head"));
  if (!e) throw new Error(`trainMlTask(${task.id}): no trainable local engine`);
  return e;
}

async function fitEngine(task: MlTask<any, any>, ref: MlEngineRef, samples: MlSample[], opts: MlTrainOptions): Promise<MlPredictor> {
  if (task.train?.fit) return task.train.fit(samples, ref);
  if (ref.engine === "classical") return fitClassical(ref.algorithm, samples, opts.classical);
  if (ref.engine === "head") {
    const objective = task.kind === "classify" ? "softmax" : task.kind === "regress" ? "mse" : null;
    if (!objective) throw new Error(`trainMlTask(${task.id}): a head engine trains classify or regress tasks only`);
    const head = new DenseHead({
      inputDim: samples[0]!.x.length,
      hiddenDim: ref.head === "linear" ? 0 : (opts.head?.hiddenDim ?? 16),
      outputDim: objective === "softmax" ? task.output.labels!.length : 1,
      objective,
      ...(opts.head?.seed != null ? { seed: opts.head.seed } : {}),
    });
    await new HeadTrainer(head, opts.device ?? null, `trainMlTask(${task.id})`).train(samples, opts.head?.train);
    return head;
  }
  throw new Error(`trainMlTask(${task.id}): engine '${engineKey(ref)}' is not trained here`);
}

/**
 * Fit task `task` on labelled examples. Returns null when there are fewer than
 * the task's `minSamples` (it stays untrained). Throws for a task with no train
 * spec, or one another trainer owns.
 */
export async function trainMlTask<I>(
  task: MlTask<I, any>,
  examples: readonly MlTrainExample<I>[],
  opts: MlTrainOptions = {},
): Promise<MlTrainResult | null> {
  const spec = task.train;
  if (!spec) throw new Error(`trainMlTask(${task.id}): task is rule-based or pre-trained only`);
  if (spec.trainedBy) throw new Error(`trainMlTask(${task.id}): trained by ${spec.trainedBy}`);
  if (examples.length < spec.minSamples) return null;

  const ref = opts.engine ?? defaultEngine(task);
  const X = await extractFeatureMatrix(task.features, examples.map((e) => e.input), opts.embedder);
  const samples: MlSample[] = examples.map((e, i) => ({ x: X[i]!, y: e.label, ...(e.weight != null ? { weight: e.weight } : {}) }));

  // Hold out the chronological TAIL: examples arrive oldest first, and the question
  // is always "does this generalise to what comes next".
  const supervised = task.kind === "classify" || task.kind === "regress";
  const frac = supervised ? (opts.holdoutFraction ?? spec.holdoutFraction ?? 0.2) : 0;
  const nHold = Math.floor(samples.length * frac);
  let holdoutLoss: number | null = null;
  if (nHold >= 1 && samples.length - nHold >= 1) {
    const train = samples.slice(0, samples.length - nHold);
    const hold = samples.slice(samples.length - nHold);
    const predictor = await fitEngine(task, ref, train, opts);
    holdoutLoss = mlEvalLoss(task.kind, predictor, hold);
    if (opts.refitOnAll === false) {
      opts.models?.set(task.id, engineKey(ref), predictor);
      return { predictor, engine: ref, samples: train.length, holdout: nHold, holdoutLoss };
    }
  }
  const predictor = await fitEngine(task, ref, samples, opts);
  opts.models?.set(task.id, engineKey(ref), predictor);
  return { predictor, engine: ref, samples: samples.length, holdout: nHold >= 1 ? nHold : 0, holdoutLoss };
}
