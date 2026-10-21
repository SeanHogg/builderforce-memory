/**
 * tests/ml_federation.test.ts — trained ML heads federate through the EXISTING
 * delta wire and FedAvg merge, gated by held-out loss (docs/ml-task-catalog.md §6):
 * two heads → diff → wire → merge → eval gate, plus the manifest's `mlHeads`.
 */
import {
  buildDeltaLearnPayload,
  parseDeltaLearnPayload,
  mlHeadTarget,
} from "../src/evermind/delta_wire.js";
import { mergeCheckpointDiffs, mergeCheckpointDiffsGated } from "../src/evermind/merge.js";
import { EvermindModelPackage } from "../src/evermind/package.js";
import { EvermindLM } from "../src/lm/evermind_lm.js";
import { diffCheckpoints } from "../src/utils/delta.js";
import {
  headCheckpoint,
  headParams,
  mlHeadEntry,
  mlHeadParams,
  packageMlHead,
  withPackageMlHead,
  buildMlHeadDelta,
  mergeMlHeadDeltas,
} from "../src/ml/federate.js";
import { fitLogistic, LogisticModel } from "../src/ml/classical/logistic.js";
import { mlEvalLoss } from "../src/ml/train.js";
import type { MlSample } from "../src/ml/types.js";

const s = (x: number[], y: number): MlSample => ({ x: Float32Array.from(x), y });

/** A 1-D separable task: label = x > 0. */
const heldOut = [-2, -1, -0.5, 0.5, 1, 2].map((x) => s([x], x > 0 ? 1 : 0));
const localA = [-3, -1.5, 1.5, 3].map((x) => s([x], x > 0 ? 1 : 0));
const localB = [-2.5, -0.8, 0.8, 2.5].map((x) => s([x], x > 0 ? 1 : 0));

function scoreParams(params: Float32Array): number | null {
  const m = new LogisticModel(1, 2);
  m.importParams(params);
  return mlEvalLoss("classify", m, heldOut);
}

describe("delta wire: target", () => {
  test("a head delta carries ml:<task>; the model's own deltas stay target-less", () => {
    const diff = diffCheckpoints(headCheckpoint([0, 0]), headCheckpoint([1, 0]));
    const head = buildDeltaLearnPayload(diff, 3, 2, "x", mlHeadTarget("run.local-success"));
    expect(head.target).toBe("ml:run.local-success");
    const parsed = parseDeltaLearnPayload(head);
    expect(parsed.ok && parsed.payload.target).toBe("ml:run.local-success");
    const plain = parseDeltaLearnPayload(buildDeltaLearnPayload(diff, 3));
    expect(plain.ok && "target" in plain.payload).toBe(false);
    expect(parseDeltaLearnPayload({ ...head, target: "lm" })).toMatchObject({ ok: false, reason: "invalid" });
    expect(parseDeltaLearnPayload({ ...head, target: 7 })).toMatchObject({ ok: false, reason: "invalid" });
    expect(() => buildDeltaLearnPayload(diff, 3, 1, "x", "nope")).toThrow(/bad target/);
  });
});

describe("mergeCheckpointDiffsGated", () => {
  const base = headCheckpoint([0, 0, 0]);
  const loss = (cp: ArrayBuffer) => {
    const p = headParams(cp);
    return (p[0]! - 1) ** 2 + p[1]! ** 2; // optimum at [1, 0, *]
  };

  test("keeps an improving merge and equals the plain FedAvg result", () => {
    const d1 = diffCheckpoints(base, headCheckpoint([1, 0, 0]));
    const d2 = diffCheckpoints(base, headCheckpoint([0.8, 0, 2]));
    const g = mergeCheckpointDiffsGated(base, [d1, d2], undefined, loss);
    expect(g.accepted).toBe(true);
    expect(g.newLoss!).toBeLessThan(g.baseLoss!);
    expect(Array.from(headParams(g.checkpoint))).toEqual(Array.from(headParams(mergeCheckpointDiffs(base, [d1, d2]).checkpoint)));
  });

  test("refuses a regressing merge and returns the base unchanged", () => {
    const bad = diffCheckpoints(base, headCheckpoint([0, 5, 0]));
    const g = mergeCheckpointDiffsGated(base, [bad], undefined, loss);
    expect(g).toMatchObject({ accepted: false, refusedReason: "regressed", deltaNorm: 0 });
    expect(g.checkpoint).toBe(base);
    // Within tolerance it is kept.
    expect(mergeCheckpointDiffsGated(base, [bad], undefined, loss, { tolerance: 100 }).accepted).toBe(true);
  });

  test("nothing merged / unscored", () => {
    expect(mergeCheckpointDiffsGated(base, [], undefined, loss).refusedReason).toBe("nothing-merged");
    const d = diffCheckpoints(base, headCheckpoint([1, 0, 0]));
    expect(mergeCheckpointDiffsGated(base, [d], undefined, () => null).accepted).toBe(true);
    expect(mergeCheckpointDiffsGated(base, [d], undefined, () => null, { requireEval: true }).refusedReason).toBe("unscored");
  });
});

describe("federation round-trip: two heads → diff → wire → FedAvg → eval gate", () => {
  test("two locally-trained logistic heads merge into a head no worse on held-out data", () => {
    // Version 0 is an untrained head everyone starts from.
    const start = new LogisticModel(1, 2);
    const v0 = mlHeadEntry("run.local-success", "local:classical:logistic", 0, start.exportParams(), { dim: 1, classes: 2 });

    // Two learners continue from v0 on their own data.
    const a = new LogisticModel(1, 2);
    a.importParams(mlHeadParams(v0));
    fitLogistic(localA, { epochs: 150 }, a);
    const b = new LogisticModel(1, 2);
    b.importParams(mlHeadParams(v0));
    fitLogistic(localB, { epochs: 150 }, b);

    // Each ships a pre-diffed delta on the existing wire (through JSON, as on the network).
    const wire = [buildMlHeadDelta(v0, a.exportParams(), 4), buildMlHeadDelta(v0, b.exportParams(), 4)].map((p) => {
      const parsed = parseDeltaLearnPayload(JSON.parse(JSON.stringify(p)));
      if (!parsed.ok) throw new Error(parsed.error);
      return parsed.payload;
    });

    const out = mergeMlHeadDeltas(v0, wire, scoreParams);
    expect(out.rejected).toEqual([]);
    expect(out.merge.accepted).toBe(true);
    expect(out.merge.contributors).toBe(2);
    expect(out.head.version).toBe(1);
    expect(out.merge.newLoss!).toBeLessThan(out.merge.baseLoss!);
    expect(out.head.evalLoss).toBe(out.merge.newLoss);

    // FedAvg of equal-weight contributors = the element-wise mean.
    const merged = mlHeadParams(out.head);
    const pa = a.exportParams();
    const pb = b.exportParams();
    for (let i = 0; i < merged.length; i++) expect(merged[i]).toBeCloseTo((pa[i]! + pb[i]!) / 2, 5);

    // The merged head classifies the held-out set.
    const m = new LogisticModel(1, 2);
    m.importParams(merged);
    expect(m.predict(Float32Array.of(1)).scores[1]).toBeGreaterThan(0.5);
    expect(m.predict(Float32Array.of(-1)).scores[0]).toBeGreaterThan(0.5);
  });

  test("a regressing round is refused; bad contributions cost only themselves", () => {
    const good = new LogisticModel(1, 2);
    fitLogistic(heldOut, { epochs: 300 }, good);
    const v3 = mlHeadEntry("run.local-success", "local:classical:logistic", 3, good.exportParams());

    // A learner trained on INVERTED labels.
    const inverted = new LogisticModel(1, 2);
    inverted.importParams(mlHeadParams(v3));
    fitLogistic(heldOut.map((x) => ({ ...x, y: 1 - x.y })), { epochs: 300 }, inverted);

    const stale = { ...buildMlHeadDelta(v3, inverted.exportParams()), baseVersion: 2 };
    const wrongTarget = { ...buildMlHeadDelta(v3, inverted.exportParams()), target: "ml:experience.quality" };
    const lmDelta = { ...buildMlHeadDelta(v3, inverted.exportParams()) };
    delete (lmDelta as { target?: string }).target;
    const out = mergeMlHeadDeltas(v3, [buildMlHeadDelta(v3, inverted.exportParams()), stale, wrongTarget, lmDelta], scoreParams);
    expect(out.rejected.map((r) => r.index)).toEqual([1, 2, 3]);
    expect(out.merge).toMatchObject({ accepted: false, refusedReason: "regressed" });
    expect(out.head).toBe(v3);
  });
});

describe("Evermind package carries mlHeads", () => {
  function pkg(): EvermindModelPackage {
    const lm = new EvermindLM({ vocabSize: 24, dModel: 8, numLayers: 1, hiddenDim: 16, seed: 3 });
    return EvermindModelPackage.fromLM(lm, { name: "t", version: "1", card: { description: "test" } });
  }

  test("heads round-trip through the blob, survive withCheckpoint, and validate", () => {
    const head = mlHeadEntry("run.anomaly", "local:classical:robust-z", 0, [1, 2, 3], { dim: 3 }, 0.25);
    const p = withPackageMlHead(pkg(), head);
    const back = EvermindModelPackage.fromBlob(p.toBlob());
    expect(back.validate()).toEqual({ ok: true, errors: [] });
    expect(packageMlHead(back, "run.anomaly")).toEqual(head);
    expect(Array.from(mlHeadParams(packageMlHead(back, "run.anomaly")!))).toEqual([1, 2, 3]);
    expect(packageMlHead(back.withCheckpoint(back.checkpoint, "2"), "run.anomaly")).toEqual(head);

    // Replacing a head keeps one entry per task; an empty list removes the field.
    const replaced = withPackageMlHead(back, { ...head, version: 1 });
    expect(replaced.manifest.mlHeads).toHaveLength(1);
    expect(replaced.manifest.mlHeads![0]!.version).toBe(1);
    expect("mlHeads" in replaced.withMlHeads([]).manifest).toBe(false);
  });

  test("a corrupt head fails validation", () => {
    const head = mlHeadEntry("run.anomaly", "local:classical:robust-z", 0, [1, 2, 3]);
    const corrupt = pkg().withMlHeads([{ ...head, params: head.params.slice(0, -8) + "AAAAAAAA" }]);
    expect(corrupt.validate().ok).toBe(false);
    const dup = pkg().withMlHeads([head, head]);
    expect(dup.validate().errors.join(" ")).toMatch(/more than once/);
  });
});
