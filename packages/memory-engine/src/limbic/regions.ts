/**
 * regions.ts – Limbic-system region map and affective state schema.
 *
 * The limbic model is the *dynamic* affective/motivational layer that rides on
 * top of the (static) psychometric personality. Where the hippocampus
 * (HybridMambaModel + MemoryStore) holds *what the agent knows*, the limbic
 * model holds *how the agent currently feels and what it is driven toward* —
 * and learns the dynamics of that, in WebGPU, from experience.
 *
 * This file is the single source of truth for the affective state vector that
 * every other limbic module (model, trainer, runtime service) indexes into.
 * Keep the dimension ids in sync with the runtime compiler in
 * `agent-runtime/src/builderforce/limbic.ts` — the two are coupled solely by
 * these string ids and the {@link LIMBIC_DIM} indices.
 *
 * Region → state mapping (mirrors the labelled diagram):
 *   • Amygdala      → salience / threat appraisal → drives valence + arousal
 *   • Hypothalamus  → homeostatic drives (curiosity, caution, effort, social)
 *   • Thalamus      → attention gate (how much incoming signal is admitted)
 *   • Basal ganglia → action selection bias (explore vs. exploit)
 *   • Hippocampus   → reused (existing SSM memory); feeds the experience input
 */

/** The five modelled limbic regions. Hippocampus is reused, not re-modelled. */
export const REGION = {
  amygdala: "amygdala",
  hypothalamus: "hypothalamus",
  thalamus: "thalamus",
  basalGanglia: "basal_ganglia",
  hippocampus: "hippocampus",
} as const;
export type Region = (typeof REGION)[keyof typeof REGION];

/**
 * Canonical indices into the affective state vector. The vector is a dense
 * Float32Array of length {@link LIMBIC_STATE_DIM}. Core affect (valence,
 * arousal) is the 2D summary; the remaining dims are the per-region drives the
 * agent's behaviour is modulated by.
 */
export const LIMBIC_DIM = {
  /** Core affect — pleasantness. Range [-1, +1] (negative .. positive). */
  valence: 0,
  /** Core affect — activation. Range [0, 1] (calm .. activated). */
  arousal: 1,
  /** Hypothalamus drive — appetite for novelty/exploration. Range [0, 1]. */
  driveCuriosity: 2,
  /** Hypothalamus drive — appetite for safety/guardrails. Range [0, 1]. */
  driveCaution: 3,
  /** Hypothalamus drive — available energy. Range [0, 1] (fatigued .. fresh). */
  driveEffort: 4,
  /** Hypothalamus drive — appetite for communication/collaboration. Range [0, 1]. */
  driveSocial: 5,
  /** Thalamus — attention gain on incoming signal. Range [0, 1]. */
  attention: 6,
  /** Basal ganglia — explore(1) vs. exploit(0) action-selection bias. Range [0, 1]. */
  exploration: 7,
} as const;
export type LimbicDimName = keyof typeof LIMBIC_DIM;

/** Length of the affective state vector. */
export const LIMBIC_STATE_DIM = 8;

/** Ordered dim names, index-aligned with {@link LIMBIC_DIM}. */
export const LIMBIC_DIM_NAMES: LimbicDimName[] = [
  "valence",
  "arousal",
  "driveCuriosity",
  "driveCaution",
  "driveEffort",
  "driveSocial",
  "attention",
  "exploration",
];

/** Inclusive [min, max] bounds per state dim, index-aligned. Valence is signed. */
export const LIMBIC_BOUNDS: ReadonlyArray<readonly [number, number]> = [
  [-1, 1], // valence
  [0, 1], // arousal
  [0, 1], // driveCuriosity
  [0, 1], // driveCaution
  [0, 1], // driveEffort
  [0, 1], // driveSocial
  [0, 1], // attention
  [0, 1], // exploration
];

/**
 * Neutral resting state — the default homeostatic setpoint before personality
 * pulls it anywhere. Calm, mildly positive, balanced drives, full attention,
 * slightly exploit-biased.
 */
export const NEUTRAL_STATE: ReadonlyArray<number> = [
  0.0, // valence
  0.2, // arousal
  0.5, // driveCuriosity
  0.5, // driveCaution
  0.8, // driveEffort
  0.5, // driveSocial
  0.7, // attention
  0.5, // exploration (centred → resting state is behaviourally inert)
];

/** Clamp a single state dim to its bounds. */
export function clampDim(index: number, value: number): number {
  const b = LIMBIC_BOUNDS[index];
  if (!b) return value;
  if (Number.isNaN(value)) return b[0];
  return Math.max(b[0], Math.min(b[1], value));
}

/** Clamp a whole state vector in place and return it. */
export function clampState(state: Float32Array): Float32Array {
  for (let i = 0; i < state.length && i < LIMBIC_STATE_DIM; i++) {
    state[i] = clampDim(i, state[i]!);
  }
  return state;
}

/** A fresh neutral state vector. */
export function neutralState(): Float32Array {
  return Float32Array.from(NEUTRAL_STATE);
}

/** Build a labelled record from a dense state vector (for logging / transport). */
export function stateToRecord(state: ArrayLike<number>): Record<LimbicDimName, number> {
  const out = {} as Record<LimbicDimName, number>;
  for (let i = 0; i < LIMBIC_DIM_NAMES.length; i++) {
    out[LIMBIC_DIM_NAMES[i]!] = state[i] ?? 0;
  }
  return out;
}

/** Build a dense state vector from a (possibly partial) labelled record. */
export function recordToState(rec: Partial<Record<LimbicDimName, number>>): Float32Array {
  const s = neutralState();
  for (let i = 0; i < LIMBIC_DIM_NAMES.length; i++) {
    const v = rec[LIMBIC_DIM_NAMES[i]!];
    if (typeof v === "number" && !Number.isNaN(v)) s[i] = clampDim(i, v);
  }
  return s;
}

/**
 * The static psychometric traits (0..100, 50 = neutral) that pull the resting
 * affective setpoint away from {@link NEUTRAL_STATE}. A subset of the full
 * personality vector — only the dimensions that map to a limbic drive. All
 * optional; an omitted trait is treated as neutral (50).
 *
 * Keep the mapping in {@link personalitySetpoint} in sync with the runtime
 * compiler `deriveLimbicSetpoints` in `@builderforce/agent-tools` so the on-prem
 * runtime, the cloud engine, and this WebGPU trainer settle on the SAME baseline
 * for the same personality. The two are coupled only by this mapping.
 */
export interface PersonalityTraits {
  openness?: number;
  emotionality?: number;
  conscientiousness?: number;
  extraversion?: number;
  /** Regulatory focus: 0 = prevention, 100 = promotion. */
  regulatoryFocus?: number;
  riskTolerance?: number;
  grit?: number;
  /** Schwartz "stimulation" value. */
  stimulation?: number;
}

/** Signed influence of a 0..100 trait around its neutral midpoint → [-1, 1]. */
function infl(s: number | undefined): number {
  const v = typeof s === "number" && !Number.isNaN(s) ? Math.max(0, Math.min(100, s)) : 50;
  return (v - 50) / 50;
}

/**
 * Derive the personality-conditioned resting SETPOINT from a trait vector — the
 * homeostatic target the limbic dynamics relax toward and that the trainable
 * affect model rides on top of ("personality = setpoints, limbic = dynamics").
 * Returns a fresh clamped 8-dim state vector. A fully-neutral trait vector yields
 * {@link NEUTRAL_STATE}. Mirrors `deriveLimbicSetpoints` in `@builderforce/agent-tools`.
 */
export function personalitySetpoint(traits: PersonalityTraits | undefined): Float32Array {
  const s = neutralState();
  if (!traits) return s;
  const open = infl(traits.openness);
  const emo = infl(traits.emotionality);
  const consc = infl(traits.conscientiousness);
  const extra = infl(traits.extraversion);
  const reg = infl(traits.regulatoryFocus);
  const risk = infl(traits.riskTolerance);
  const grit = infl(traits.grit);
  const stim = infl(traits.stimulation);

  s[LIMBIC_DIM.driveCuriosity] = clampDim(LIMBIC_DIM.driveCuriosity, 0.5 + 0.35 * open + 0.15 * stim);
  s[LIMBIC_DIM.exploration] = clampDim(LIMBIC_DIM.exploration, 0.4 + 0.3 * open + 0.25 * risk + 0.15 * reg);
  s[LIMBIC_DIM.driveCaution] = clampDim(LIMBIC_DIM.driveCaution, 0.5 + 0.3 * consc - 0.3 * risk - 0.2 * reg + 0.15 * emo);
  s[LIMBIC_DIM.arousal] = clampDim(LIMBIC_DIM.arousal, 0.2 + 0.2 * emo + 0.1 * extra);
  s[LIMBIC_DIM.driveSocial] = clampDim(LIMBIC_DIM.driveSocial, 0.5 + 0.35 * extra);
  s[LIMBIC_DIM.driveEffort] = clampDim(LIMBIC_DIM.driveEffort, 0.8 + 0.15 * grit + 0.1 * consc);
  s[LIMBIC_DIM.valence] = clampDim(LIMBIC_DIM.valence, 0.0 + 0.1 * reg - 0.1 * emo);
  s[LIMBIC_DIM.attention] = clampDim(LIMBIC_DIM.attention, 0.7 + 0.1 * consc);
  return s;
}
