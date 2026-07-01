/**
 * tests/limbic_regions.test.ts
 * Affective-state schema: bounds, clamping, and labelled <-> dense conversions.
 */

import {
  LIMBIC_DIM,
  LIMBIC_DIM_NAMES,
  LIMBIC_STATE_DIM,
  LIMBIC_BOUNDS,
  clampDim,
  clampState,
  neutralState,
  stateToRecord,
  recordToState,
  personalitySetpoint,
} from "../src/limbic/regions.js";

test("dimension indices are unique and cover the state vector", () => {
  const indices = Object.values(LIMBIC_DIM);
  expect(new Set(indices).size).toBe(LIMBIC_STATE_DIM);
  expect(Math.max(...indices)).toBe(LIMBIC_STATE_DIM - 1);
  expect(LIMBIC_DIM_NAMES.length).toBe(LIMBIC_STATE_DIM);
  expect(LIMBIC_BOUNDS.length).toBe(LIMBIC_STATE_DIM);
});

test("valence is signed [-1,1]; all other dims are [0,1]", () => {
  expect(LIMBIC_BOUNDS[LIMBIC_DIM.valence]).toEqual([-1, 1]);
  for (let i = 0; i < LIMBIC_STATE_DIM; i++) {
    if (i === LIMBIC_DIM.valence) continue;
    expect(LIMBIC_BOUNDS[i]).toEqual([0, 1]);
  }
});

test("clampDim respects per-dim bounds and rejects NaN", () => {
  expect(clampDim(LIMBIC_DIM.valence, -5)).toBe(-1);
  expect(clampDim(LIMBIC_DIM.valence, 5)).toBe(1);
  expect(clampDim(LIMBIC_DIM.arousal, -0.3)).toBe(0);
  expect(clampDim(LIMBIC_DIM.arousal, 2)).toBe(1);
  expect(clampDim(LIMBIC_DIM.attention, Number.NaN)).toBe(0);
});

test("clampState clamps every out-of-range entry in place", () => {
  const s = Float32Array.from([9, 9, 9, -9, 9, -9, 9, 9]);
  clampState(s);
  expect(s[LIMBIC_DIM.valence]).toBe(1);
  expect(s[LIMBIC_DIM.driveCaution]).toBe(0);
  expect(s[LIMBIC_DIM.attention]).toBe(1);
});

test("neutral state is within bounds", () => {
  const s = neutralState();
  expect(s.length).toBe(LIMBIC_STATE_DIM);
  for (let i = 0; i < s.length; i++) {
    const [lo, hi] = LIMBIC_BOUNDS[i]!;
    expect(s[i]).toBeGreaterThanOrEqual(lo);
    expect(s[i]).toBeLessThanOrEqual(hi);
  }
});

test("record <-> dense round-trips and clamps on the way in", () => {
  const rec = stateToRecord(neutralState());
  expect(rec.valence).toBeCloseTo(0, 6);
  expect(rec.attention).toBeCloseTo(0.7, 6);

  const dense = recordToState({ valence: 0.8, arousal: 5 /* clamps to 1 */ });
  expect(dense[LIMBIC_DIM.valence]).toBeCloseTo(0.8, 6);
  expect(dense[LIMBIC_DIM.arousal]).toBe(1);
  // Unspecified dims fall back to neutral.
  expect(dense[LIMBIC_DIM.driveEffort]).toBeCloseTo(0.8, 6);
});

test("personalitySetpoint: undefined traits → neutral resting state", () => {
  expect(Array.from(personalitySetpoint(undefined))).toEqual(Array.from(neutralState()));
});

test("personalitySetpoint: all-neutral (50) traits leave affect/drives at neutral", () => {
  // Mirrors agent-tools deriveLimbicSetpoints: a computed setpoint centres every
  // drive at neutral EXCEPT exploration, which rests mildly exploit-leaning (0.4).
  const s = personalitySetpoint({ openness: 50, conscientiousness: 50, extraversion: 50, emotionality: 50 });
  expect(s[LIMBIC_DIM.valence]).toBeCloseTo(0, 6);
  expect(s[LIMBIC_DIM.arousal]).toBeCloseTo(0.2, 6);
  expect(s[LIMBIC_DIM.driveCuriosity]).toBeCloseTo(0.5, 6);
  expect(s[LIMBIC_DIM.driveCaution]).toBeCloseTo(0.5, 6);
  expect(s[LIMBIC_DIM.driveEffort]).toBeCloseTo(0.8, 6);
  expect(s[LIMBIC_DIM.driveSocial]).toBeCloseTo(0.5, 6);
  expect(s[LIMBIC_DIM.attention]).toBeCloseTo(0.7, 6);
  expect(s[LIMBIC_DIM.exploration]).toBeCloseTo(0.4, 6);
});

test("personalitySetpoint: high openness lifts curiosity + exploration", () => {
  const s = personalitySetpoint({ openness: 100 });
  expect(s[LIMBIC_DIM.driveCuriosity]).toBeGreaterThan(neutralState()[LIMBIC_DIM.driveCuriosity]!);
  expect(s[LIMBIC_DIM.exploration]).toBeGreaterThan(neutralState()[LIMBIC_DIM.exploration]!);
});

test("personalitySetpoint: high conscientiousness raises caution + attention", () => {
  const s = personalitySetpoint({ conscientiousness: 100 });
  expect(s[LIMBIC_DIM.driveCaution]).toBeGreaterThan(0.5);
  expect(s[LIMBIC_DIM.attention]).toBeGreaterThan(0.7);
});

test("personalitySetpoint: extraversion drives the social + arousal setpoints", () => {
  const introvert = personalitySetpoint({ extraversion: 0 });
  const extravert = personalitySetpoint({ extraversion: 100 });
  expect(extravert[LIMBIC_DIM.driveSocial]).toBeGreaterThan(introvert[LIMBIC_DIM.driveSocial]!);
  expect(extravert[LIMBIC_DIM.arousal]).toBeGreaterThan(introvert[LIMBIC_DIM.arousal]!);
});

test("personalitySetpoint: every dim stays within bounds at trait extremes", () => {
  const s = personalitySetpoint({
    openness: 100, emotionality: 100, conscientiousness: 100, extraversion: 100,
    regulatoryFocus: 100, riskTolerance: 100, grit: 100, stimulation: 100,
  });
  for (let i = 0; i < s.length; i++) {
    const [lo, hi] = LIMBIC_BOUNDS[i]!;
    expect(s[i]).toBeGreaterThanOrEqual(lo);
    expect(s[i]).toBeLessThanOrEqual(hi);
  }
});
