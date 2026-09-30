/**
 * tests/limbic_affect.test.ts
 * The heuristic limbic regions and the directive compiler: labelled state
 * plumbing, personality setpoints, amygdala / hypothalamus / thalamus / basal
 * ganglia, the state → behaviour compiler and task appraisal.
 */

import {
  LIMBIC_DIM_NAMES,
  appraiseAmygdala,
  appraiseTask,
  applyLimbicDelta,
  basalGangliaExploreBias,
  basalGangliaSelect,
  buildLimbicBlock,
  clampLimbicDim,
  compileLimbicState,
  homeostasis,
  limbicSetpoints,
  meanLimbicSetpoints,
  neutralLimbicState,
  personalitySetpoint,
  recordToState,
  stateToRecord,
  thalamusGate,
  LIMBIC_BLOCK_HEADER,
  type LimbicState,
} from "../src/limbic/affect.js";

describe("labelled state plumbing", () => {
  it("neutral state is bounded and stable through the dense round-trip", () => {
    const s = neutralLimbicState();
    const round = stateToRecord(recordToState(s));
    for (const n of LIMBIC_DIM_NAMES) expect(round[n]).toBeCloseTo(s[n], 6);
  });

  it("clampLimbicDim respects signed valence and [0,1] drives", () => {
    expect(clampLimbicDim("valence", -3)).toBe(-1);
    expect(clampLimbicDim("valence", 3)).toBe(1);
    expect(clampLimbicDim("arousal", -1)).toBe(0);
    expect(clampLimbicDim("driveEffort", 9)).toBe(1);
  });

  it("applyLimbicDelta clamps and ignores NaN entries", () => {
    const s = applyLimbicDelta(neutralLimbicState(), { valence: 5, arousal: -5, driveCuriosity: Number.NaN });
    expect(s.valence).toBe(1);
    expect(s.arousal).toBe(0);
    expect(s.driveCuriosity).toBe(neutralLimbicState().driveCuriosity);
  });
});

describe("limbicSetpoints (personality = setpoints)", () => {
  it("absent traits → neutral resting setpoints", () => {
    expect(limbicSetpoints(undefined)).toEqual(neutralLimbicState());
  });

  it("high openness rests more curious and exploratory", () => {
    const sp = limbicSetpoints({ openness: 95, riskTolerance: 80 });
    const base = neutralLimbicState();
    expect(sp.driveCuriosity).toBeGreaterThan(base.driveCuriosity);
    expect(sp.exploration).toBeGreaterThan(base.exploration);
  });

  it("conscientious + emotional + prevention rests more cautious", () => {
    const sp = limbicSetpoints({ conscientiousness: 90, emotionality: 85, regulatoryFocus: 10, riskTolerance: 10 });
    expect(sp.driveCaution).toBeGreaterThan(neutralLimbicState().driveCaution);
  });

  it("extraversion raises the social setpoint; grit raises the effort setpoint", () => {
    const sp = limbicSetpoints({ extraversion: 90, grit: 90 });
    expect(sp.driveSocial).toBeGreaterThan(neutralLimbicState().driveSocial);
    expect(sp.driveEffort).toBeGreaterThanOrEqual(neutralLimbicState().driveEffort);
  });

  it("meanLimbicSetpoints averages several personalities; none → neutral", () => {
    expect(meanLimbicSetpoints([])).toEqual(neutralLimbicState());
    const a = limbicSetpoints({ openness: 90 });
    const b = limbicSetpoints({ openness: 10 });
    const mean = meanLimbicSetpoints([{ openness: 90 }, { openness: 10 }]);
    for (const n of LIMBIC_DIM_NAMES) expect(mean[n]).toBeCloseTo((a[n] + b[n]) / 2, 9);
  });

  it("the dense setpoint the trainable model uses is the same mapping", () => {
    const traits = { openness: 80, emotionality: 30, grit: 70 };
    const dense = personalitySetpoint(traits);
    const labelled = limbicSetpoints(traits);
    LIMBIC_DIM_NAMES.forEach((n, i) => expect(dense[i]).toBeCloseTo(labelled[n], 6));
  });
});

describe("amygdala appraisal", () => {
  it("errors are negative, arousing, and raise caution", () => {
    const d = appraiseAmygdala({ kind: "error", intensity: 1 });
    expect(d.valence!).toBeLessThan(0);
    expect(d.arousal!).toBeGreaterThan(0);
    expect(d.driveCaution!).toBeGreaterThan(0);
  });

  it("success is positive and calming; intensity scales magnitude", () => {
    const strong = appraiseAmygdala({ kind: "success", intensity: 1 });
    const weak = appraiseAmygdala({ kind: "success", intensity: 0.2 });
    expect(strong.valence!).toBeGreaterThan(0);
    expect(strong.arousal!).toBeLessThan(0);
    expect(Math.abs(strong.valence!)).toBeGreaterThan(Math.abs(weak.valence!));
  });

  it("feedback sign flips valence", () => {
    expect(appraiseAmygdala({ kind: "feedback", sign: -1 }).valence!).toBeLessThan(0);
    expect(appraiseAmygdala({ kind: "feedback", sign: 1 }).valence!).toBeGreaterThan(0);
  });
});

describe("hypothalamus homeostasis", () => {
  it("relaxes the state toward setpoints", () => {
    const setpoints = neutralLimbicState();
    let s: LimbicState = applyLimbicDelta(neutralLimbicState(), { valence: -0.8, arousal: 0.7 });
    const before = Math.abs(s.valence - setpoints.valence);
    s = homeostasis(s, setpoints, { rate: 0.3 });
    const after = Math.abs(s.valence - setpoints.valence);
    expect(after).toBeLessThan(before);
  });

  it("converges to setpoints over many ticks", () => {
    const setpoints = limbicSetpoints({ openness: 90 });
    let s = applyLimbicDelta(neutralLimbicState(), { valence: -0.9, arousal: 0.9, driveEffort: -0.5 });
    for (let i = 0; i < 200; i++) s = homeostasis(s, setpoints, { rate: 0.2 });
    for (const n of LIMBIC_DIM_NAMES) expect(s[n]).toBeCloseTo(setpoints[n], 2);
  });

  it("fatigue drains effort", () => {
    const s = homeostasis(neutralLimbicState(), neutralLimbicState(), { rate: 0, fatigue: 0.2 });
    expect(s.driveEffort).toBeCloseTo(neutralLimbicState().driveEffort - 0.2, 5);
  });
});

describe("thalamus attention gate (Yerkes–Dodson)", () => {
  it("peaks at moderate arousal and degrades at the extremes", () => {
    const mid = thalamusGate({ ...neutralLimbicState(), arousal: 0.5 });
    const low = thalamusGate({ ...neutralLimbicState(), arousal: 0.0 });
    const high = thalamusGate({ ...neutralLimbicState(), arousal: 1.0 });
    expect(mid).toBeGreaterThan(low);
    expect(mid).toBeGreaterThan(high);
    expect(mid).toBeCloseTo(1, 5);
    expect(low).toBeGreaterThanOrEqual(0.1);
  });
});

describe("basal ganglia action selection", () => {
  it("high exploration + curiosity biases toward novelty", () => {
    const explorer = { ...neutralLimbicState(), exploration: 1, driveCuriosity: 1, valence: 0.5 };
    expect(basalGangliaExploreBias(explorer)).toBeGreaterThan(0.65);
  });

  it("low effort + caution biases toward exploit", () => {
    const tired = { ...neutralLimbicState(), exploration: 0.2, driveEffort: 0.1, driveCaution: 0.9 };
    expect(basalGangliaExploreBias(tired)).toBeLessThan(0.35);
  });

  it("select picks the candidate whose novelty matches the bias", () => {
    const explorer = { ...neutralLimbicState(), exploration: 1, driveCuriosity: 1, valence: 0.5 };
    const { choice } = basalGangliaSelect(explorer, [
      { novelty: 0.05, tag: "safe" },
      { novelty: 0.95, tag: "novel" },
    ]);
    expect(choice?.tag).toBe("novel");

    const tired = { ...neutralLimbicState(), exploration: 0.1, driveEffort: 0.1, driveCaution: 0.95 };
    const { choice: c2 } = basalGangliaSelect(tired, [
      { novelty: 0.05, tag: "safe" },
      { novelty: 0.95, tag: "novel" },
    ]);
    expect(c2?.tag).toBe("safe");
  });

  it("select returns null for empty options", () => {
    expect(basalGangliaSelect(neutralLimbicState(), []).choice).toBeNull();
  });
});

describe("compileLimbicState (dynamics → behaviour)", () => {
  it("a resting state produces no directives or params", () => {
    const { directives, params } = compileLimbicState(neutralLimbicState());
    expect(directives).toEqual([]);
    expect(params).toEqual({});
  });

  it("strong negative affect deepens thinking and turns reasoning on", () => {
    const { directives, params } = compileLimbicState({ ...neutralLimbicState(), valence: -0.7 });
    expect(directives.join(" ")).toMatch(/negative/i);
    expect(params.thinkLevel).toBe("high");
    expect(params.reasoningLevel).toBe("on");
  });

  it("high caution emits a guardrail directive and a think floor", () => {
    const { directives, params } = compileLimbicState({ ...neutralLimbicState(), driveCaution: 0.9 });
    expect(directives.join(" ")).toMatch(/caution/i);
    expect(["medium", "high", "xhigh"]).toContain(params.thinkLevel);
  });

  it("exploration raises temperature; caution lowers it", () => {
    const hot = compileLimbicState({ ...neutralLimbicState(), exploration: 1 });
    const cold = compileLimbicState({ ...neutralLimbicState(), driveCaution: 1, exploration: 0 });
    expect(hot.params.temperatureDelta!).toBeGreaterThan(0);
    expect(cold.params.temperatureDelta!).toBeLessThan(0);
  });

  it("is deterministic", () => {
    const s = { ...neutralLimbicState(), valence: -0.5, arousal: 0.8, driveCaution: 0.8 };
    expect(compileLimbicState(s)).toEqual(compileLimbicState(s));
  });

  it("buildLimbicBlock renders directives and is empty at rest", () => {
    expect(buildLimbicBlock(neutralLimbicState())).toBe("");
    expect(buildLimbicBlock({ ...neutralLimbicState(), valence: -0.8 })).toMatch(/affective state/i);
  });
});

describe("appraiseTask (initial affect from task text — cloud V3 / VS Code)", () => {
  it("risky/destructive work raises caution and arousal", () => {
    const s = appraiseTask("Delete the production database and wipe all rows");
    expect(s.driveCaution).toBeGreaterThan(neutralLimbicState().driveCaution);
    expect(s.arousal).toBeGreaterThan(neutralLimbicState().arousal);
    // and that compiles to a caution directive
    expect(compileLimbicState(s).directives.join(" ")).toMatch(/caution/i);
  });

  it("large/complex work raises curiosity and exploration", () => {
    const s = appraiseTask("Refactor the entire architecture across the whole codebase");
    expect(s.driveCuriosity).toBeGreaterThan(neutralLimbicState().driveCuriosity);
    expect(s.exploration).toBeGreaterThan(neutralLimbicState().exploration);
  });

  it("a mundane task stays at rest (no directives)", () => {
    const s = appraiseTask("Fix a typo in the README heading");
    expect(compileLimbicState(s).directives).toEqual([]);
  });

  it("is deterministic and respects an explicit base state", () => {
    expect(appraiseTask("delete prod")).toEqual(appraiseTask("delete prod"));
    const base = { ...neutralLimbicState(), valence: -0.5 };
    expect(appraiseTask("fix typo", base).valence).toBeCloseTo(-0.5, 6);
  });
});

describe("buildLimbicBlock", () => {
  it("renders the header then one bullet per directive", () => {
    const block = buildLimbicBlock({ ...neutralLimbicState(), valence: -0.8 });
    const [header, ...bullets] = block.split("\n");
    expect(header).toBe(LIMBIC_BLOCK_HEADER);
    expect(bullets.length).toBeGreaterThan(0);
    expect(bullets.every((b) => b.startsWith("- "))).toBe(true);
  });
});
