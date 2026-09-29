/**
 * The model-driving half of Evermind serving: flatten a chat into a continuation
 * prompt, generate under a wall-clock budget, grade a head's fitness to serve, and
 * bind a head to the tool planner's decoder port.
 *
 * Written against STRUCTURAL model/tokenizer shapes ({@link EvermindTextModel},
 * {@link EvermindTextTokenizer}) that the engine's `EvermindLM` and `BPETokenizer`
 * satisfy, so this module imports nothing and the `/evermind` subpath stays
 * dependency-free. Where the head comes FROM (an R2 artifact, a file, a freshly merged
 * in-memory model) is the host's business; everything here takes a loaded head.
 */
import { isServableText, type CoherenceFailure } from './textCoherence.js';
import {
  planEvermindToolCall,
  type EvermindPlannedCall,
  type EvermindToolDecoder,
  type NormalizedTool,
  type ToolChoicePlan,
} from './toolCall.js';

/** What serving needs from a loaded text head (the engine's `EvermindLM`). */
export interface EvermindTextModel {
  generateText(prompt: string, tok: EvermindTextTokenizer, opts: { maxNewTokens: number; temperature: number; seed?: number }): string;
  forward(ids: number[]): { logits: ArrayLike<number>[] };
}

/** What serving needs from a head's tokenizer (the engine's `BPETokenizer`). */
export interface EvermindTextTokenizer {
  encode(text: string): number[];
}

export interface EvermindUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface EvermindGenerateOptions {
  maxTokens?: number;
  temperature?: number;
  seed?: number;
  /**
   * Wall-clock budget for the generation loop, in ms. Evermind generation is
   * SYNCHRONOUS CPU on the request path, so without this a large head on a slow host
   * runs until the host kills the request — no message and no partial answer. With
   * it, the loop stops at the budget and says so.
   */
  deadlineMs?: number;
}

export interface EvermindGeneration {
  content: string;
  usage: EvermindUsage;
  /** True when the wall-clock budget stopped generation before `maxTokens` or a stop
   *  token did — the text is a real partial answer, not a complete one. */
  truncated: boolean;
  /** How long the generation loop actually took. */
  elapsedMs: number;
}

/** Flatten chat messages into a single continuation prompt for the LM. */
export function messagesToPrompt(messages: Array<{ role?: unknown; content?: unknown }>): string {
  const lines = messages
    .map((m) => {
      const role = typeof m.role === 'string' ? m.role : 'user';
      const content = typeof m.content === 'string' ? m.content : '';
      return content ? `${role}: ${content}` : '';
    })
    .filter(Boolean);
  return `${lines.join('\n')}\nassistant:`;
}

/**
 * Tokens generated between wall-clock checks. The deadline is enforced BETWEEN slices
 * because the engine's generate is one synchronous loop with no per-token hook.
 * Slicing costs nothing: the forward pass already recomputes the whole sequence per
 * token, so re-entering with `prompt + producedSoFar` is the work the next token was
 * going to do anyway. Small enough that the overshoot is one slice.
 */
const GENERATE_SLICE_TOKENS = 16;
/** Default wall-clock budget for one generation — inside a Worker's CPU allowance, so
 *  the caller gets a partial answer plus `truncated` rather than a killed request. */
const DEFAULT_GENERATE_DEADLINE_MS = 8000;

/**
 * Generate a reply from a loaded head. Deterministic for a given (prompt, seed,
 * maxTokens): each slice derives its seed from the base seed and its index, so two
 * operators running the same probe see the same text.
 */
export function generateEvermindText(
  lm: EvermindTextModel,
  tok: EvermindTextTokenizer,
  messages: Array<{ role?: unknown; content?: unknown }>,
  opts: EvermindGenerateOptions = {},
): EvermindGeneration {
  const prompt = messagesToPrompt(messages);
  const maxTokens = opts.maxTokens ?? 256;
  const temperature = opts.temperature ?? 0.7;
  const baseSeed = opts.seed;
  const deadlineMs = opts.deadlineMs ?? DEFAULT_GENERATE_DEADLINE_MS;

  const started = Date.now();
  let content = '';
  let truncated = false;
  let produced = 0;
  let slice = 0;
  while (produced < maxTokens) {
    const want = Math.min(GENERATE_SLICE_TOKENS, maxTokens - produced);
    const chunk = lm.generateText(`${prompt}${content}`, tok, {
      maxNewTokens: want,
      temperature,
      ...(baseSeed != null ? { seed: baseSeed + slice } : {}),
    });
    slice++;
    // An empty slice means the model stopped producing; continuing would spin.
    if (!chunk) break;
    content += chunk;
    produced += want;
    if (Date.now() - started >= deadlineMs) {
      truncated = produced < maxTokens;
      break;
    }
  }

  const prompt_tokens = tok.encode(prompt).length;
  const completion_tokens = content ? tok.encode(content).length : 0;
  return {
    content,
    usage: { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens },
    truncated,
    elapsedMs: Date.now() - started,
  };
}

// ── Fitness to serve ────────────────────────────────────────────────────────

/** Neutral probe prompts a project chat head should answer coherently. Fixed and
 *  generic (not project-specific) so the probe measures GENERATION QUALITY, not
 *  recall. */
export const COHERENCE_PROBE_PROMPTS: readonly string[] = [
  'Summarize the current status of the project.',
  'What has the team been working on recently?',
  'List the main things left to do.',
];

/** One graded probe generation. */
export interface EvermindCoherenceSample {
  prompt: string;
  text: string;
  coherent: boolean;
  /** The failing signal when `coherent` is false (null when it passed) — so an
   *  operator sees WHY a head was refused, not just that it was. */
  failure: CoherenceFailure | null;
  /** Short human-readable explanation of {@link failure} (empty when coherent). */
  detail: string;
}

/** A head's fitness-to-serve verdict. */
export interface EvermindCoherenceAssessment {
  ready: boolean;
  /** Fraction of probe samples that were substantive AND coherent (0..1). */
  passRate: number;
  samples: EvermindCoherenceSample[];
}

/**
 * Score a loaded head's fitness to serve chat: generate from the neutral probe
 * prompts and grade each with the serve-time bar ({@link isServableText}).
 * Synchronous and storage-free, so the promote-to-inference gate and a learning
 * coordinator holding a freshly merged model grade it the same way. Deterministic
 * seeds keep the verdict reproducible; a majority must pass by default.
 */
export function assessLMCoherence(
  lm: EvermindTextModel,
  tok: EvermindTextTokenizer,
  opts: { minPassRate?: number } = {},
): EvermindCoherenceAssessment {
  const samples: EvermindCoherenceSample[] = COHERENCE_PROBE_PROMPTS.map((prompt, i) => {
    const text = lm.generateText(messagesToPrompt([{ role: 'user', content: prompt }]), tok, {
      maxNewTokens: 80,
      temperature: 0.7,
      seed: 1234 + i,
    });
    const verdict = isServableText(text, { context: prompt });
    return { prompt, text, coherent: verdict.coherent, failure: verdict.failure, detail: verdict.detail };
  });
  const passRate = samples.length ? samples.filter((s) => s.coherent).length / samples.length : 0;
  return { ready: passRate >= (opts.minPassRate ?? 0.5), passRate, samples };
}

/** One operator-run test-bench generation, graded with the serve-time bar. */
export interface EvermindProbeGeneration extends EvermindCoherenceSample {
  usage: EvermindUsage;
  /** True when the wall-clock budget stopped generation early — an incoherent verdict
   *  on a truncated sample is a statement about the clock, not about the model. */
  truncated: boolean;
  elapsedMs: number;
}

/**
 * Test bench: run ONE prompt through a head and grade the output with the SAME bar
 * the serve path uses, so "what will this model actually produce?" is answerable
 * before inference is switched on. Deterministic by default (`seed`).
 */
export function probeEvermindText(
  lm: EvermindTextModel,
  tok: EvermindTextTokenizer,
  prompt: string,
  opts: EvermindGenerateOptions = {},
): EvermindProbeGeneration {
  const gen = generateEvermindText(lm, tok, [{ role: 'user', content: prompt }], {
    maxTokens: opts.maxTokens ?? 120,
    temperature: opts.temperature ?? 0.7,
    seed: opts.seed ?? 1234,
    ...(opts.deadlineMs != null ? { deadlineMs: opts.deadlineMs } : {}),
  });
  const verdict = isServableText(gen.content, { context: prompt });
  return {
    prompt,
    text: gen.content,
    coherent: verdict.coherent,
    failure: verdict.failure,
    detail: verdict.detail,
    usage: gen.usage,
    truncated: gen.truncated,
    elapsedMs: gen.elapsedMs,
  };
}

// ── Tool calling ─────────────────────────────────────────────────────────────

/**
 * A tool-decision prompt is rebuilt for every candidate and every argument, and each
 * costs a full forward pass, so the conversation is capped rather than replayed whole.
 * Truncated from the LEFT: the most recent turns are what a tool choice depends on.
 */
const MAX_TOOL_PROMPT_CHARS = 6000;

/** Log-probability the model assigned to `id` at a position, from that position's raw
 *  logits (log-softmax, max-shifted so long-tail logits don't overflow). */
function logProbOf(row: ArrayLike<number>, id: number): number {
  let max = -Infinity;
  for (let i = 0; i < row.length; i++) { const v = row[i]!; if (v > max) max = v; }
  if (!Number.isFinite(max)) return -Infinity;
  let sum = 0;
  for (let i = 0; i < row.length; i++) sum += Math.exp(row[i]! - max);
  const logit = id >= 0 && id < row.length ? row[id]! : -Infinity;
  return logit - (max + Math.log(sum));
}

/** Keep the tail of an over-long prompt (see {@link MAX_TOOL_PROMPT_CHARS}). */
function clampPromptText(prompt: string): string {
  return prompt.length <= MAX_TOOL_PROMPT_CHARS ? prompt : prompt.slice(prompt.length - MAX_TOOL_PROMPT_CHARS);
}

/** A decoder plus the token usage it accumulated, so a tool-calling turn reports real
 *  numbers instead of zeros. */
export interface MeteredToolDecoder extends EvermindToolDecoder {
  usage(): EvermindUsage;
}

/**
 * Bind a loaded head to the {@link EvermindToolDecoder} port.
 *
 * `score`: teacher forcing lets one forward pass over `prompt + continuation` yield
 * the log-prob of EVERY continuation token at once (position `t` predicts `t+1`), so
 * ranking a candidate costs one pass. The mean is returned — not the sum — so a long
 * tool name is not out-voted by a short one for its length alone.
 *
 * `generate` is greedy by default: a tool ARGUMENT is a value to get right, not prose
 * to vary, and determinism keeps a replayed run reproducible.
 */
export function createEvermindToolDecoder(
  lm: EvermindTextModel,
  tok: EvermindTextTokenizer,
  opts: { temperature?: number; seed?: number } = {},
): MeteredToolDecoder {
  let promptTokens = 0;
  let completionTokens = 0;
  return {
    score(prompt: string, continuation: string): number {
      const contIds = tok.encode(continuation);
      if (contIds.length === 0) return -Infinity;
      // A leading token is required for the first continuation token to have a
      // position to be predicted FROM; an empty prompt gets the same id-0 prefix the
      // engine's own sampler uses.
      const promptIds = tok.encode(clampPromptText(prompt));
      const prefix = promptIds.length > 0 ? promptIds : [0];
      const { logits } = lm.forward([...prefix, ...contIds]);
      let total = 0;
      for (let i = 0; i < contIds.length; i++) {
        total += logProbOf(logits[prefix.length + i - 1]!, contIds[i]!);
      }
      promptTokens += prefix.length;
      completionTokens += contIds.length;
      return total / contIds.length;
    },
    generate(prompt: string, maxTokens: number): string {
      const text = lm.generateText(clampPromptText(prompt), tok, {
        maxNewTokens: maxTokens,
        temperature: opts.temperature ?? 0,
        ...(opts.seed != null ? { seed: opts.seed } : {}),
      });
      promptTokens += tok.encode(clampPromptText(prompt)).length;
      completionTokens += text ? tok.encode(text).length : 0;
      return text;
    },
    usage: () => ({ prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens }),
  };
}

/** A tool-aware generation: either planned calls, or prose when the head chose to
 *  answer directly (`tool_choice: 'auto'`). */
export interface EvermindToolGeneration {
  /** The first planned call, or null for prose. */
  call: EvermindPlannedCall | null;
  /** EVERY call this turn emits (the OpenAI shape is an array; frontier models emit
   *  parallel calls). Empty when `call` is null. */
  calls: EvermindPlannedCall[];
  /** Prose answer — populated only when `call` is null. */
  content: string;
  usage: EvermindUsage;
  /** Confidence separation behind the plan — returned unjudged: the host owns the
   *  policy of what separation is good enough, because it owns the fallback. */
  margin: number;
}

/**
 * Run a TOOL-BEARING request against a loaded head: plan a call by constrained
 * decoding, or fall through to ordinary prose when the head elected to answer directly.
 */
export function generateEvermindWithTools(
  lm: EvermindTextModel,
  tok: EvermindTextTokenizer,
  messages: Array<{ role?: unknown; content?: unknown }>,
  tools: NormalizedTool[],
  choice: ToolChoicePlan,
  opts: EvermindGenerateOptions = {},
): EvermindToolGeneration {
  const decoder = createEvermindToolDecoder(lm, tok, {
    ...(opts.temperature != null ? { temperature: opts.temperature } : {}),
    ...(opts.seed != null ? { seed: opts.seed } : {}),
  });
  // The planner is given the conversation WITHOUT the `assistant:` primer — it is
  // choosing an action, not continuing a reply.
  const conversation = messagesToPrompt(messages).replace(/\nassistant:$/, '');
  const plan = planEvermindToolCall(decoder, conversation, tools, choice);
  if (plan.call) {
    return { call: plan.call, calls: plan.calls, content: '', usage: decoder.usage(), margin: plan.margin };
  }
  const gen = generateEvermindText(lm, tok, messages, opts);
  const usage = decoder.usage();
  return {
    call: null,
    calls: [],
    content: gen.content,
    usage: {
      prompt_tokens: usage.prompt_tokens + gen.usage.prompt_tokens,
      completion_tokens: usage.completion_tokens + gen.usage.completion_tokens,
      total_tokens: usage.total_tokens + gen.usage.total_tokens,
    },
    margin: plan.margin,
  };
}
