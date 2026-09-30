/**
 * affect.ts – the heuristic limbic regions and the directive compiler.
 *
 * The labelled (record-shaped) face of the limbic system every agent surface runs:
 * the on-prem runtime, the cloud engine in a Worker, and the VS Code agent. Where
 * the psychometric personality is *static* (a trait vector → behaviour), this layer
 * is *dynamic* (a live affective state → behaviour): personality sets the
 * homeostatic setpoints ({@link limbicSetpoints}), the regions push the live state
 * around them in response to experience.
 *
 *   • Amygdala      → fast salience/threat appraisal of an event → affect delta
 *   • Hypothalamus  → homeostasis: relax toward setpoints; effort fatigue
 *   • Thalamus      → attention gate (inverted-U of arousal — Yerkes–Dodson)
 *   • Basal ganglia → action-selection bias (explore vs. exploit)
 *   • Hippocampus   → reused (the SSM memory) — supplies the experience embedding
 *                     the trainable {@link LimbicModel} learns from.
 *
 * Pure and deterministic (no GPU, no I/O, no imports beyond the schema), so it
 * runs anywhere and is the trainable model's heuristic teacher and always-available
 * fallback. Published alone as `@seanhogg/builderforce-memory-engine/limbic`.
 */

import {
  LIMBIC_DIM,
  LIMBIC_DIM_NAMES,
  NEUTRAL_STATE,
  clampDim,
  limbicSetpoints,
  stateToRecord,
  type LimbicDimName,
  type PersonalityTraits,
} from "./regions.js";

export {
  REGION,
  LIMBIC_DIM,
  LIMBIC_DIM_NAMES,
  LIMBIC_STATE_DIM,
  LIMBIC_BOUNDS,
  NEUTRAL_STATE,
  clampDim,
  clampState,
  neutralState,
  stateToRecord,
  recordToState,
  limbicSetpoints,
  personalitySetpoint,
} from "./regions.js";
export type { Region, LimbicDimName, PersonalityTraits } from "./regions.js";

// ── Labelled state ─────────────────────────────────────────────────────────────

export type LimbicState = Record<LimbicDimName, number>;
export type LimbicSetpoints = LimbicState;
export type LimbicDelta = Partial<Record<LimbicDimName, number>>;

/** Clamp one named dim to its bounds (NaN → the lower bound). */
export function clampLimbicDim(name: LimbicDimName, v: number): number {
  return clampDim(LIMBIC_DIM[name], v);
}

/** A fresh labelled neutral state. */
export function neutralLimbicState(): LimbicState {
  return stateToRecord(NEUTRAL_STATE);
}

/**
 * The resting setpoints of several personalities active at once (an agent's own
 * personality plus its assigned personas): the mean of each one's
 * {@link limbicSetpoints}. Neutral when there are none.
 */
export function meanLimbicSetpoints(traits: ReadonlyArray<PersonalityTraits>): LimbicSetpoints {
  if (traits.length === 0) return neutralLimbicState();
  const acc = neutralLimbicState();
  for (const name of LIMBIC_DIM_NAMES) acc[name] = 0;
  for (const t of traits) {
    const sp = limbicSetpoints(t);
    for (const name of LIMBIC_DIM_NAMES) acc[name] += sp[name];
  }
  for (const name of LIMBIC_DIM_NAMES) acc[name] /= traits.length;
  return acc;
}

export function applyLimbicDelta(state: LimbicState, delta: LimbicDelta): LimbicState {
  const out = { ...state };
  for (const name of LIMBIC_DIM_NAMES) {
    const d = delta[name];
    if (typeof d === "number" && !Number.isNaN(d)) out[name] = clampLimbicDim(name, out[name] + d);
  }
  return out;
}

// ── Amygdala — fast salience/threat appraisal of an event → affect delta ─────────

export interface LimbicEvent {
  kind: "success" | "progress" | "error" | "blocked" | "risk" | "feedback" | "idle";
  intensity?: number;
  sign?: number;
  text?: string;
}

const I = (e: LimbicEvent): number => Math.max(0, Math.min(1, e.intensity ?? 0.5));

export function appraiseAmygdala(event: LimbicEvent): LimbicDelta {
  const k = I(event);
  switch (event.kind) {
    case "success":
      return { valence: +0.5 * k, arousal: -0.15 * k, driveCaution: -0.1 * k, driveEffort: -0.05 * k };
    case "progress":
      return { valence: +0.25 * k, arousal: +0.05 * k, driveEffort: -0.03 * k };
    case "error":
      return { valence: -0.45 * k, arousal: +0.4 * k, driveCaution: +0.3 * k, attention: +0.15 * k };
    case "blocked":
      return { valence: -0.35 * k, arousal: +0.3 * k, driveEffort: -0.15 * k, driveSocial: +0.2 * k };
    case "risk":
      return { arousal: +0.35 * k, driveCaution: +0.4 * k, attention: +0.2 * k, exploration: -0.15 * k };
    case "feedback": {
      const s = Math.sign(event.sign ?? 1) || 1;
      return { valence: 0.4 * k * s, arousal: 0.1 * k, driveSocial: +0.1 * k };
    }
    case "idle":
      return { arousal: -0.2 * k, driveEffort: +0.15 * k };
    default:
      return {};
  }
}

// ── Hypothalamus — homeostasis: relax toward setpoints + effort fatigue ──────────

export function homeostasis(
  state: LimbicState,
  setpoints: LimbicSetpoints,
  opts: { rate?: number; fatigue?: number } = {},
): LimbicState {
  const rate = opts.rate ?? 0.1;
  const fatigue = opts.fatigue ?? 0.0;
  const out = { ...state };
  for (const name of LIMBIC_DIM_NAMES) {
    out[name] = clampLimbicDim(name, out[name] + rate * (setpoints[name] - out[name]));
  }
  if (fatigue > 0) out.driveEffort = clampLimbicDim("driveEffort", out.driveEffort - fatigue);
  return out;
}

// ── Thalamus — attention gate (inverted-U of arousal, Yerkes–Dodson) ─────────────

export function thalamusGate(state: LimbicState, optimalArousal = 0.5): number {
  const d = state.arousal - optimalArousal;
  const gain = 1 - 2.8 * d * d;
  return Math.max(0.1, Math.min(1, gain));
}

// ── Basal ganglia — action selection (explore vs. exploit) ───────────────────────

export function basalGangliaExploreBias(state: LimbicState): number {
  let b = 0.5 * state.exploration + 0.3 * state.driveCuriosity;
  b += 0.15 * state.valence;
  b -= 0.25 * (1 - state.driveEffort);
  b -= 0.2 * (state.driveCaution - 0.5);
  return Math.max(0, Math.min(1, b));
}

export function basalGangliaSelect<T extends { novelty: number }>(
  state: LimbicState,
  options: T[],
): { choice: T | null; exploreBias: number } {
  if (options.length === 0) return { choice: null, exploreBias: basalGangliaExploreBias(state) };
  const bias = basalGangliaExploreBias(state);
  let best = options[0]!;
  let bestScore = -Infinity;
  for (const o of options) {
    const nov = Math.max(0, Math.min(1, o.novelty));
    const sc = -Math.abs(nov - bias);
    if (sc > bestScore) {
      bestScore = sc;
      best = o;
    }
  }
  return { choice: best, exploreBias: bias };
}

// ── The compiler: limbic state → { directives, params } ──────────────────────────

/**
 * Execution nudges the limbic state asks for. Structurally a subset of an agent
 * host's exec params: the compiler only ever raises thinking to a floor of
 * "medium" or "high" and turns reasoning on — the host merges these onto its own
 * params with its own think-level ladder.
 */
export type LimbicExecParams = {
  thinkLevel?: "medium" | "high";
  reasoningLevel?: "on";
  /** Signed temperature delta in roughly [-0.3, 0.3]. */
  temperatureDelta?: number;
};

export type CompiledLimbic = {
  directives: string[];
  params: LimbicExecParams;
};

const HI_F = 0.6;
const LO_F = 0.4;

export function compileLimbicState(state: LimbicState): CompiledLimbic {
  const directives: string[] = [];
  const params: LimbicExecParams = {};
  // The deeper of the floors asked for so far ("high" beats "medium").
  const floor = (level: "medium" | "high"): void => {
    if (params.thinkLevel !== "high") params.thinkLevel = level;
  };

  if (state.valence <= -0.4) {
    directives.push(
      "Affect (negative): recent steps have gone badly — slow down, avoid rash or destructive actions, and re-verify your assumptions before proceeding.",
    );
    floor("high");
    params.reasoningLevel = "on";
  } else if (state.valence >= 0.4) {
    directives.push("Affect (positive): things are going well — keep momentum, but don't get sloppy.");
  }

  if (state.arousal >= 0.7) {
    directives.push(
      "Arousal (heightened): you are highly activated — resist the urge to rush; double-check before any irreversible step.",
    );
    floor("high");
  } else if (state.arousal <= 0.15) {
    directives.push("Arousal (low): this is routine — keep it lightweight and efficient.");
  }

  if (state.driveCaution >= HI_F) {
    directives.push(
      "Drive (caution): heightened — prefer reversible steps, add guardrails, and confirm risky or destructive operations.",
    );
    floor("medium");
  } else if (state.driveCaution <= LO_F) {
    directives.push("Drive (caution): relaxed — bias to action on low-risk work.");
  }

  if (state.driveCuriosity >= HI_F) {
    directives.push("Drive (curiosity): high — it is appropriate to investigate and consider alternative approaches.");
  }

  if (state.driveEffort <= LO_F) {
    directives.push(
      "Drive (effort): energy is low — keep scope tight, avoid over-engineering, checkpoint progress, and consider escalating rather than grinding.",
    );
  }

  if (state.driveSocial >= HI_F) {
    directives.push("Drive (social): communicate proactively — surface progress and ask for input when blocked.");
  } else if (state.driveSocial <= LO_F) {
    directives.push("Drive (social): work heads-down — narrate only what matters.");
  }

  const attentionGain = thalamusGate(state);
  if (attentionGain <= 0.5) {
    directives.push(
      "Attention (degraded): your attention gate is low — re-read the task carefully and do not skip verification steps.",
    );
    floor("medium");
  }

  const explore = basalGangliaExploreBias(state);
  if (explore >= 0.7) {
    directives.push("Action selection: lean exploratory — try a novel approach before settling on the obvious one.");
  } else if (explore <= 0.3) {
    directives.push("Action selection: lean exploitative — use the proven, known approach and avoid unnecessary detours.");
  }

  let dTemp = 0;
  dTemp += 0.2 * (state.exploration - 0.5) * 2;
  dTemp += 0.06 * (state.driveCuriosity - 0.5) * 2;
  dTemp += 0.08 * state.valence;
  dTemp -= 0.16 * (state.driveCaution - 0.5) * 2;
  dTemp -= 0.08 * Math.max(0, state.arousal - 0.7);
  const clampedTemp = Math.max(-0.3, Math.min(0.3, Math.round(dTemp * 100) / 100));
  if (Math.abs(clampedTemp) >= 0.02) params.temperatureDelta = clampedTemp;

  return { directives, params };
}

/** Heading of the system-prompt block {@link buildLimbicBlock} renders. */
export const LIMBIC_BLOCK_HEADER = "Current affective state (execute accordingly):";

/** Render the limbic directives as a system-prompt sub-block. '' when at rest. */
export function buildLimbicBlock(state: LimbicState | undefined): string {
  if (!state) return "";
  const { directives } = compileLimbicState(state);
  if (directives.length === 0) return "";
  return [LIMBIC_BLOCK_HEADER, ...directives.map((d) => `- ${d}`)].join("\n");
}

// ── Task appraisal — derive an initial affect from a task description ─────────────
// For surfaces with no live event stream yet (cloud, VS Code): a quick amygdala
// read of the task text so the agent starts in a fitting affective state.

const RISK_PATTERNS =
  /\b(delete|drop|truncate|destroy|wipe|prod(uction)?|migrat|secret|credential|password|payment|billing|charge|refund|deploy|force[- ]?push|rm\s+-rf|irreversible|security|auth|breaking)\b/i;
const HARD_PATTERNS = /\b(refactor|rewrite|architecture|complex|large|entire|whole|across|end[- ]to[- ]end|overhaul)\b/i;

/**
 * Appraise a task's text into an initial affective deviation from `base`
 * (defaults to neutral): risky/destructive work raises caution + arousal;
 * large/complex work raises effort engagement + curiosity. Deterministic.
 */
export function appraiseTask(text: string, base: LimbicState = neutralLimbicState()): LimbicState {
  const t = text || "";
  let s = base;
  if (RISK_PATTERNS.test(t)) {
    s = applyLimbicDelta(s, { driveCaution: +0.3, arousal: +0.25, attention: +0.15, exploration: -0.1 });
  }
  if (HARD_PATTERNS.test(t)) {
    s = applyLimbicDelta(s, { driveCuriosity: +0.2, arousal: +0.1, exploration: +0.1 });
  }
  return s;
}
