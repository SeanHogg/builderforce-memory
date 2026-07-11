/**
 * workflow/steps.ts — the built-in step types.
 *
 * Two families, both compiling to a runnable {@link StackStep}:
 *   • Agent-stack layers (L1..L7) — the diagnostic checks (the 7-layer template).
 *   • LLM-creation pipeline (BUILD) — train-tokenizer → train-model → evaluate →
 *     package, which produces a trained `.evermind` artifact (a custom LLM) into
 *     `ctx.artifacts.evermind`.
 *
 * A custom workflow mixes these freely; new step types register the same way.
 */

import {
  EvermindLM,
  EvermindLMTrainer,
  BPETokenizer,
  EvermindModelPackage,
  exportEvermind,
  importEvermind,
  benchmarkText,
  VideoRVQCodec,
  buildVideoSequence,
  generateVideo,
  type ExportFormat,
  type BenchmarkReport,
  type ImportOptions,
  type Video,
} from "@seanhogg/builderforce-memory-engine";
import { EvermindCognition } from "../cognition/index.js";
import { hybridRetrieve, chunkText } from "../retrieval/index.js";
import { MemoryStore } from "../memory/MemoryStore.js";
import { OpenAIBridge } from "../bridges/OpenAIBridge.js";
import { AnthropicBridge } from "../bridges/AnthropicBridge.js";
import type { TransformerBridge } from "../bridges/TransformerBridge.js";
import { analyzeCode, runJsCases, type CodeCase } from "./code-eval.js";
import type { StackStep } from "../diagnostics/stack-diagnostic.js";
import type { StepFactory, StepTypeInfo, WorkflowStepConfig } from "./types.js";

const KNOWLEDGE =
  "BuilderForce orchestrates many agents through a planning loop. " +
  "The memory layer stores facts as SSM embeddings. " +
  "Deployment runs on Cloudflare Workers and Durable Objects. " +
  "Tools are gated by a capability registry.";

let dbSeq = 0;

// ── param readers (configurable per step) ──────────────────────────────────────
function pStr(cfg: WorkflowStepConfig, key: string, def: string): string {
  const v = cfg.params?.[key];
  return typeof v === "string" ? v : def;
}
function pNum(cfg: WorkflowStepConfig, key: string, def: number): number {
  const v = cfg.params?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : def;
}
function pBool(cfg: WorkflowStepConfig, key: string, def: boolean): boolean {
  const v = cfg.params?.[key];
  return typeof v === "boolean" ? v : def;
}
/** Read a string[] param (non-strings dropped). */
function pStrArr(cfg: WorkflowStepConfig, key: string): string[] {
  const v = cfg.params?.[key];
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

/** Default teacher instruction for code distillation — code only, no prose/fences. */
const CODE_TEACHER_SYSTEM =
  "You are an expert programmer. Respond with correct, idiomatic code that fulfils the request. " +
  "Output code only — no explanation, no markdown fences.";

/**
 * A (prompt → teacher completion) training exemplar produced by distillation.
 * `cases` (when present) execution-verify the completion; `teacher` records
 * which panel member produced the surviving answer (attribution/telemetry).
 */
interface DistillPair {
  prompt: string;
  completion: string;
  cases?: CodeCase[];
  teacher?: string;
}

/** One coding task to distil: a prompt plus the tests its answer must pass. */
interface DistillTask {
  prompt: string;
  cases?: CodeCase[];
}

/** A single teacher's answer to a prompt, before selection. */
interface Candidate {
  teacher: string;
  completion: string;
}

/** A configured teacher in the panel (its display label + bridge). */
interface Teacher {
  label: string;
  bridge: TransformerBridge;
}

/** One teacher's config — a panel entry, or the legacy single-teacher shape. */
interface TeacherSpec {
  provider?: string;
  apiKey?: string;
  model?: string;
  baseUrl?: string;
  systemPrompt?: string;
  maxTokens?: number;
}

/**
 * Strip a wrapping ```lang … ``` markdown fence some teachers add despite the
 * "code only, no fences" instruction — otherwise the fence fails every code
 * gate and the exemplar is wrongly dropped.
 */
export function stripCodeFences(text: string): string {
  const t = text.trim();
  const m = /^```[\w-]*\r?\n([\s\S]*?)\r?\n```$/.exec(t);
  return m ? m[1]!.trim() : t;
}

/** Read `cases` off a param blob, dropping malformed entries. */
function normalizeCases(raw: unknown): CodeCase[] {
  return Array.isArray(raw)
    ? raw.filter((c): c is CodeCase => typeof (c as { call?: unknown })?.call === "string")
    : [];
}

/**
 * Best-of-N selection across a teacher panel. When the task carries execution
 * `cases`, keep only fully-passing candidates and return the first — teachers
 * are listed strongest-first (e.g. Opus), so the strongest correct answer wins
 * and a wrong teacher answer is dropped rather than trained on. With no cases,
 * take the first non-empty candidate (teacher-order priority).
 */
export function selectBestCandidate(candidates: Candidate[], cases: CodeCase[]): Candidate | null {
  if (candidates.length === 0) return null;
  if (cases.length === 0) return candidates[0]!;
  for (const c of candidates) {
    if (runJsCases(c.completion, cases).passRate === 1) return c;
  }
  return null;
}

/**
 * Build one teacher bridge from its spec. Any external LLM can teach Evermind:
 *   • provider "openai" (default) — the BuilderForce gateway, OpenRouter, OpenAI,
 *     or any OpenAI-compatible server (Ollama / vLLM / LM Studio), by `baseUrl`.
 *   • provider "anthropic" — the Anthropic Messages API directly (Opus etc).
 * Returns null when the entry lacks the credentials it needs.
 */
function buildTeacher(spec: TeacherSpec, defSystem: string, defMaxTokens: number): Teacher | null {
  const provider = (spec.provider ?? "openai").toLowerCase();
  const apiKey = spec.apiKey ?? "";
  const model = spec.model ?? "";
  const systemPrompt = spec.systemPrompt ?? defSystem;
  const maxTokens = spec.maxTokens ?? defMaxTokens;
  if (provider === "anthropic") {
    if (!apiKey) return null;
    return { label: `anthropic:${model || "default"}`, bridge: new AnthropicBridge({ apiKey, systemPrompt, maxTokens, ...(model ? { model } : {}) }) };
  }
  // openai-compatible (gateway / OpenRouter / OpenAI / local) — needs a baseUrl.
  const baseUrl = spec.baseUrl ?? "";
  if (!baseUrl) return null;
  return { label: `${provider}:${model || "default"}`, bridge: new OpenAIBridge({ apiKey, baseUrl, systemPrompt, maxTokens, ...(model ? { model } : {}) }) };
}

/**
 * Build the teacher panel from step params. `teachers: TeacherSpec[]` runs
 * several models as a best-of-N panel (list the strongest FIRST — e.g. Opus —
 * so it wins ties); the legacy single-teacher shape (top-level
 * provider/baseUrl/apiKey/model) is honoured as a one-entry panel. Entries
 * without credentials are skipped, so an offline `pairs`-only run yields [].
 */
function makeTeachers(cfg: WorkflowStepConfig): Teacher[] {
  const defSystem = pStr(cfg, "systemPrompt", CODE_TEACHER_SYSTEM);
  const defMaxTokens = pNum(cfg, "maxTokens", 512);
  const raw = cfg.params?.teachers;
  const specs: TeacherSpec[] =
    Array.isArray(raw) && raw.length > 0
      ? (raw as TeacherSpec[])
      : [{ provider: pStr(cfg, "provider", "openai"), apiKey: pStr(cfg, "apiKey", ""), model: pStr(cfg, "model", ""), baseUrl: pStr(cfg, "baseUrl", "") }];
  const out: Teacher[] = [];
  for (const s of specs) {
    const t = buildTeacher(s ?? {}, defSystem, defMaxTokens);
    if (t) out.push(t);
  }
  return out;
}

/** Read the coding tasks to distil: rich `tasks[{prompt,cases}]` or plain `prompts[]`. */
function readTasks(cfg: WorkflowStepConfig): DistillTask[] {
  const raw = cfg.params?.tasks;
  if (Array.isArray(raw)) {
    const out: DistillTask[] = [];
    for (const t of raw) {
      const prompt = (t as DistillTask)?.prompt;
      if (typeof prompt !== "string") continue;
      const cases = normalizeCases((t as DistillTask).cases);
      out.push(cases.length ? { prompt, cases } : { prompt });
    }
    return out;
  }
  return pStrArr(cfg, "prompts").map((prompt) => ({ prompt }));
}

/** Byte length of an emitted export file (text or binary). */
function fileBytes(data: Uint8Array | string): number {
  return typeof data === "string" ? new TextEncoder().encode(data).length : data.length;
}

/** Coerce a workflow-param weight blob (bytes, ArrayBuffer, number[], or base64) to Uint8Array. */
function toBytes(value: unknown, base64?: string): Uint8Array | null {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (Array.isArray(value) && value.every((n) => typeof n === "number")) return Uint8Array.from(value as number[]);
  if (typeof base64 === "string" && base64) {
    if (typeof atob !== "function") throw new Error("import-model: base64 decoding needs atob (browser/Worker/Node 16+)");
    const bin = atob(base64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  return null;
}

function mkStep(cfg: WorkflowStepConfig, layer: string, defaultLabel: string, run: StackStep["run"]): StackStep {
  return { id: cfg.id, layer, label: cfg.label ?? defaultLabel, run };
}

/**
 * Encode a corpus into next-token training sequences. Split on sentence
 * boundaries (prose) OR blank lines (code blocks / distilled prompt→code pairs),
 * so both natural-language and code corpora yield multiple trainable sequences.
 */
function corpusToSequences(corpus: string, tok: BPETokenizer): number[][] {
  return corpus
    .split(/(?<=\.)\s+|\n\s*\n/)
    .map((s) => tok.encode(s.trim()))
    .filter((ids) => ids.length >= 2);
}

/** Build a VideoRVQCodec from step params (shared by the video build steps). */
function makeVideoCodec(cfg: WorkflowStepConfig, textVocabSize: number): VideoRVQCodec {
  return new VideoRVQCodec({
    height: pNum(cfg, "height", 8),
    width: pNum(cfg, "width", 8),
    channels: pNum(cfg, "channels", 3),
    patch: pNum(cfg, "patch", 4),
    levels: pNum(cfg, "levels", 2),
    codebookSize: pNum(cfg, "codebookSize", 16),
    keyframeInterval: pNum(cfg, "keyframeInterval", 12),
    textVocabSize,
    seed: pNum(cfg, "seed", 7),
  });
}

/**
 * Deterministic synthetic clips so the video steps are self-runnable (the way the
 * text steps default their corpus). A smooth spatial pattern that drifts a little
 * per frame — enough temporal redundancy to exercise the inter-frame path.
 */
function synthVideos(count: number, frames: number, h: number, w: number, c: number): Video[] {
  const out: Video[] = [];
  for (let v = 0; v < count; v++) {
    const clip: Video = [];
    for (let t = 0; t < frames; t++) {
      const f = new Float32Array(h * w * c);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          for (let ch = 0; ch < c; ch++) {
            f[((y * w) + x) * c + ch] = 0.5 + 0.4 * Math.sin((x + v * 1.3 + t * 0.3) * 0.7 + y * 0.5 + ch);
          }
        }
      }
      clip.push(f);
    }
    out.push(clip);
  }
  return out;
}

/** Mean per-pixel reconstruction MSE of a codec over clips (the video roundtrip metric). */
function videoReconMSE(codec: VideoRVQCodec, videos: Video[]): number {
  let se = 0;
  let n = 0;
  for (const v of videos) {
    const r = codec.decode(codec.encode(v));
    for (let t = 0; t < v.length; t++) {
      for (let k = 0; k < v[t]!.length; k++) {
        const d = v[t]![k]! - r[t]![k]!;
        se += d * d;
      }
      n += v[t]!.length;
    }
  }
  return n > 0 ? se / n : 0;
}

interface Registered {
  info: StepTypeInfo;
  factory: StepFactory;
}

export const BUILTIN_STEPS: Record<string, Registered> = {
  // ── Agent-stack layers (the diagnostic / 7-layer template) ──────────────────
  foundation: {
    info: { type: "foundation", layer: "L1", label: "Foundation Model", description: "EvermindLM + tokenizer generate text" },
    factory: (cfg) =>
      mkStep(cfg, "L1", "Foundation Model — EvermindLM + tokenizer", async (ctx) => {
        const tok = new BPETokenizer();
        tok.train(KNOWLEDGE.repeat(4), { numMerges: 60 });
        const model = new EvermindLM({ vocabSize: tok.vocabSize, dModel: 16, numLayers: 2, hiddenDim: 24, seed: 7 });
        const out = model.generateText("Deployment", tok, { maxNewTokens: 4, temperature: 0 });
        if (typeof out !== "string") throw new Error("model did not produce text");
        if (tok.decode(tok.encode("Cloudflare Workers")) !== "Cloudflare Workers") throw new Error("tokenizer did not round-trip text");
        ctx.bag.model = model;
        ctx.bag.tokenizer = tok;
        return `vocab ${tok.vocabSize}, generated ${out.length} chars`;
      }),
  },
  memory: {
    info: { type: "memory", layer: "L3", label: "Memory", description: "Write-through cognition (replace-on-write)" },
    factory: (cfg) =>
      mkStep(cfg, "L3", "Memory — write-through cognition", async (ctx) => {
        const store = new MemoryStore({ dbName: `wf-${ctx.now()}-${dbSeq++}`, idbFactory: ctx.idbFactory as never });
        const cognition = new EvermindCognition({ store });
        await cognition.commit({ subjectKey: "fact:deploy", content: "deploy target = unknown" });
        const r = await cognition.commit({ subjectKey: "fact:deploy", content: "deploy target = Cloudflare" });
        if (r.verdict !== "supersede") throw new Error(`expected supersede, got ${r.verdict}`);
        ctx.bag.cognition = cognition;
        return `verdict ${r.verdict}`;
      }),
  },
  rag: {
    info: { type: "rag", layer: "L4", label: "RAG", description: "Chunk + hybrid (BM25 + dense) retrieve" },
    factory: (cfg) =>
      mkStep(cfg, "L4", "RAG — chunk + hybrid retrieve", async (ctx) => {
        const candidates = chunkText(KNOWLEDGE, { chunkSize: 70, chunkOverlap: 0 }).map((c, i) => ({ id: `c${i}`, text: c.text }));
        const hits = hybridRetrieve({ text: "where does deployment run" }, candidates, { topK: 2 });
        if (hits.length === 0 || !hits[0]!.text.toLowerCase().includes("cloudflare")) throw new Error("retriever did not surface the deployment passage");
        ctx.bag.candidates = candidates;
        return `top hit: ${hits[0]!.text.slice(0, 40)}…`;
      }),
  },
  tools: {
    info: { type: "tools", layer: "L5", label: "Tools", description: "Capability-gated tool registry" },
    factory: (cfg) =>
      mkStep(cfg, "L5", "Tools — capability-gated registry", async (ctx) => {
        const tools = new Map<string, (a: string) => string>([["shout", (a) => a.toUpperCase()]]);
        if (tools.get("shout")?.("deploy") !== "DEPLOY") throw new Error("tool invocation failed");
        if (tools.get("missing")) throw new Error("ungated tool resolved");
        ctx.bag.tools = tools;
        return "1 tool registered + invoked";
      }),
  },
  orchestration: {
    info: { type: "orchestration", layer: "L2", label: "Orchestration", description: "retrieve→recall→act→generate loop" },
    factory: (cfg) =>
      mkStep(cfg, "L2", "Orchestration — retrieve→recall→act→generate", async (ctx) => {
        const model = ctx.bag.model as EvermindLM | undefined;
        const tokenizer = ctx.bag.tokenizer as BPETokenizer | undefined;
        const cognition = ctx.bag.cognition as EvermindCognition | undefined;
        const candidates = ctx.bag.candidates as { id: string; text: string }[] | undefined;
        const tools = ctx.bag.tools as Map<string, (a: string) => string> | undefined;
        if (!model || !tokenizer || !cognition || !candidates || !tools) throw new Error("a prerequisite layer (L1/L3/L4/L5) did not complete");
        const steps: string[] = [];
        const hits = hybridRetrieve({ text: "deployment cloudflare" }, candidates, { topK: 1 });
        steps.push("retrieve");
        await cognition.recall("deployment", 3);
        steps.push("recall");
        tools.get("shout")!(hits[0]?.text ?? "deploy");
        steps.push("tool");
        model.generateText("Deployment", tokenizer, { maxNewTokens: 3, temperature: 0 });
        steps.push("model");
        if (steps.join(",") !== "retrieve,recall,tool,model") throw new Error("loop did not complete all stages");
        return steps.join(" → ");
      }),
  },
  observability: {
    info: { type: "observability", layer: "L6", label: "Observability", description: "Trace captured every prior step" },
    factory: (cfg) =>
      mkStep(cfg, "L6", "Observability — trace captured prior steps", async (ctx) => {
        if (ctx.trace.length === 0) throw new Error("no spans captured");
        for (const s of ctx.trace) {
          if (typeof s.layer !== "string" || typeof s.status !== "string" || s.ms < 0) throw new Error(`malformed span: ${s.id}`);
        }
        return `${ctx.trace.length} spans recorded`;
      }),
  },
  deployment: {
    info: { type: "deployment", layer: "L7", label: "Deployment", description: ".evermind artifact ships + runs identically" },
    factory: (cfg) =>
      mkStep(cfg, "L7", "Deployment — .evermind ships + runs identically", async (ctx) => {
        const model = ctx.bag.model as EvermindLM | undefined;
        const tokenizer = ctx.bag.tokenizer as BPETokenizer | undefined;
        if (!model || !tokenizer) throw new Error("no model from L1 to deploy");
        const before = model.generateText("Deployment", tokenizer, { maxNewTokens: 4, temperature: 0 });
        const blob = EvermindModelPackage.fromLM(model, { name: "stack-demo", version: "1.0.0", card: { description: "diagnostic" } }).toBlob();
        const pkg = EvermindModelPackage.fromBlob(blob);
        if (!pkg.validate().ok) throw new Error("packaged artifact failed validation");
        const after = pkg.loadLM().generateText("Deployment", tokenizer, { maxNewTokens: 4, temperature: 0 });
        if (after !== before) throw new Error("deployed instance produced different output");
        return `artifact ${blob.byteLength} bytes, output matches`;
      }),
  },

  // ── LLM-creation pipeline (BUILD) — produces a custom .evermind model ────────
  "train-tokenizer": {
    info: {
      type: "train-tokenizer",
      layer: "BUILD",
      label: "Train tokenizer",
      description:
        "Build a BPE vocab — train fresh from your corpus, OR import merges from a proven code tokenizer (params: corpus, numMerges, hfTokenizer, hfTokenizerUrl)",
    },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Train tokenizer (BPE)", async (ctx) => {
        const tok = new BPETokenizer();
        // Two paths, both first-class (this is an IDE choice):
        //   • import-merges: seed from a Hugging Face tokenizer.json (code vocab on day one)
        //   • train-fresh:   learn a vocab from the corpus
        const hfTokenizer = cfg.params?.hfTokenizer as Record<string, unknown> | undefined;
        const hfTokenizerUrl = pStr(cfg, "hfTokenizerUrl", "");
        if (hfTokenizer && typeof hfTokenizer === "object") {
          tok.loadHuggingFace(hfTokenizer);
          ctx.bag.tokenizer = tok;
          return `imported vocab ${tok.vocabSize} (Hugging Face merges)`;
        }
        if (hfTokenizerUrl) {
          await tok.loadHuggingFaceUrl(hfTokenizerUrl);
          ctx.bag.tokenizer = tok;
          return `imported vocab ${tok.vocabSize} from ${hfTokenizerUrl}`;
        }
        // train-fresh — default to the distilled corpus from a prior step when present.
        const corpus = pStr(cfg, "corpus", (ctx.bag.corpus as string) ?? KNOWLEDGE.repeat(4));
        const numMerges = pNum(cfg, "numMerges", 100);
        tok.train(corpus, { numMerges });
        ctx.bag.tokenizer = tok;
        ctx.bag.corpus = corpus;
        return `trained vocab ${tok.vocabSize} (${numMerges} merges)`;
      }),
  },
  "train-model": {
    info: { type: "train-model", layer: "BUILD", label: "Train model", description: "Train an EvermindLM on the corpus (params: dModel, numLayers, hiddenDim, epochs, lr)" },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Train model (EvermindLM)", async (ctx) => {
        const tok = ctx.bag.tokenizer as BPETokenizer | undefined;
        if (!tok) throw new Error("no tokenizer — add a 'train-tokenizer' step first");
        const corpus = pStr(cfg, "corpus", (ctx.bag.corpus as string) ?? KNOWLEDGE.repeat(4));
        const sequences = corpusToSequences(corpus, tok);
        if (sequences.length === 0) throw new Error("corpus produced no trainable sequences");
        const model = new EvermindLM({
          vocabSize: tok.vocabSize,
          dModel: pNum(cfg, "dModel", 16),
          numLayers: pNum(cfg, "numLayers", 2),
          hiddenDim: pNum(cfg, "hiddenDim", 24),
          seed: pNum(cfg, "seed", 7),
        });
        const epochs = pNum(cfg, "epochs", 40);
        const history = new EvermindLMTrainer(model, { lr: pNum(cfg, "lr", 0.03), epochs }).fit(sequences);
        ctx.bag.model = model;
        // Surface the loss curve so a `convergence` diagnostic / observability can
        // assert training actually worked (it was previously discarded).
        ctx.bag.trainingHistory = history;
        const last = history.at(-1) ?? 0;
        return `trained ${epochs} epochs over ${sequences.length} seqs, loss ${last.toFixed(3)}`;
      }),
  },
  evaluate: {
    info: { type: "evaluate", layer: "BUILD", label: "Evaluate", description: "Sanity-check the model generates text (params: prompt)" },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Evaluate model", async (ctx) => {
        const model = ctx.bag.model as EvermindLM | undefined;
        const tok = ctx.bag.tokenizer as BPETokenizer | undefined;
        if (!model || !tok) throw new Error("no trained model — add 'train-model' first");
        const prompt = pStr(cfg, "prompt", "The");
        const out = model.generateText(prompt, tok, { maxNewTokens: pNum(cfg, "maxNewTokens", 6), temperature: 0 });
        if (typeof out !== "string") throw new Error("evaluation generation failed");
        ctx.bag.sample = out;
        return `sample: "${out.slice(0, 40)}"`;
      }),
  },
  benchmark: {
    info: {
      type: "benchmark",
      layer: "BUILD",
      label: "Benchmark",
      description:
        "Score the trained model on held-out text — perplexity, bits/token, top-1/top-k accuracy, throughput (params: evalCorpus, topK, maxPerplexity, minTop1)",
    },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Benchmark model (held-out)", async (ctx) => {
        const model = ctx.bag.model as EvermindLM | undefined;
        const tok = ctx.bag.tokenizer as BPETokenizer | undefined;
        if (!model || !tok) throw new Error("no trained model — add 'train-model' first");
        // Score on held-out text: the eval corpus must be text the model did not
        // train on, otherwise perplexity is meaningless. Default to a small
        // held-out probe distinct from the training KNOWLEDGE corpus.
        const evalCorpus = pStr(
          cfg,
          "evalCorpus",
          "Agents recall facts and act on them. The planning loop retrieves context before generating.",
        );
        const topK = pNum(cfg, "topK", 5);
        const report: BenchmarkReport = benchmarkText(model, tok, evalCorpus, { topK });
        if (report.tokens === 0) {
          throw new Error("eval corpus produced no scorable tokens — provide longer held-out text");
        }
        ctx.bag.benchmark = report;

        // Optional quality gates — only enforced when a threshold is configured.
        const maxPerplexity = pNum(cfg, "maxPerplexity", 0);
        if (maxPerplexity > 0 && report.perplexity > maxPerplexity) {
          throw new Error(
            `perplexity ${report.perplexity.toFixed(2)} exceeds max ${maxPerplexity} (model underfit — raise epochs/lr or enlarge the corpus)`,
          );
        }
        const minTop1 = pNum(cfg, "minTop1", 0);
        if (minTop1 > 0 && report.top1Accuracy < minTop1) {
          throw new Error(
            `top-1 accuracy ${(report.top1Accuracy * 100).toFixed(1)}% below min ${(minTop1 * 100).toFixed(0)}%`,
          );
        }
        return `ppl ${report.perplexity.toFixed(2)}, ${report.bitsPerToken.toFixed(2)} bits/tok, top1 ${(report.top1Accuracy * 100).toFixed(0)}%, top${report.topK} ${(report.topKAccuracy * 100).toFixed(0)}% over ${report.tokens} tok`;
      }),
  },
  "dataset-quality": {
    info: { type: "dataset-quality", layer: "BUILD", label: "Dataset quality", description: "Validate the corpus is trainable before spending epochs (params: minSequences, minWords, maxDuplicateRatio)" },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Dataset quality gate", async (ctx) => {
        const tok = ctx.bag.tokenizer as BPETokenizer | undefined;
        if (!tok) throw new Error("no tokenizer — add a 'train-tokenizer' step first");
        const corpus = pStr(cfg, "corpus", (ctx.bag.corpus as string) ?? "");
        const words = corpus.trim().split(/\s+/).filter(Boolean).length;
        const sequences = corpusToSequences(corpus, tok);
        const seqLens = sequences.map((s) => s.length);
        const avgSeqLen = seqLens.length ? seqLens.reduce((a, b) => a + b, 0) / seqLens.length : 0;
        const uniqueSeqs = new Set(sequences.map((s) => s.join(","))).size;
        const duplicateRatio = sequences.length ? 1 - uniqueSeqs / sequences.length : 1;

        const minSequences = pNum(cfg, "minSequences", 3);
        const minWords = pNum(cfg, "minWords", 20);
        const maxDuplicateRatio = pNum(cfg, "maxDuplicateRatio", 0.7);

        const metrics = { words, sequences: sequences.length, avgSeqLen: Number(avgSeqLen.toFixed(1)), duplicateRatio: Number(duplicateRatio.toFixed(2)) };
        ctx.bag.datasetMetrics = metrics;

        if (words < minWords) throw new Error(`corpus too small: ${words} words < ${minWords}`);
        if (sequences.length < minSequences) throw new Error(`too few trainable sequences: ${sequences.length} < ${minSequences} (need more sentences)`);
        if (duplicateRatio > maxDuplicateRatio) throw new Error(`corpus too repetitive: ${(duplicateRatio * 100) | 0}% duplicate sequences > ${(maxDuplicateRatio * 100) | 0}%`);
        return `${words} words, ${sequences.length} seqs (avg ${metrics.avgSeqLen} tok), ${(duplicateRatio * 100) | 0}% dup`;
      }),
  },
  convergence: {
    info: { type: "convergence", layer: "BUILD", label: "Convergence", description: "Assert the model actually learned — training loss decreased (params: minDrop)" },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Training convergence", async (ctx) => {
        const history = ctx.bag.trainingHistory as number[] | undefined;
        if (!history || history.length === 0) throw new Error("no training history — run 'train-model' first");
        const first = history[0]!;
        const last = history.at(-1)!;
        if (!Number.isFinite(first) || !Number.isFinite(last)) throw new Error(`loss is not finite (first=${first}, last=${last}) — training diverged`);
        const minDrop = pNum(cfg, "minDrop", 0); // require at least this absolute loss drop
        if (last >= first - minDrop) throw new Error(`loss did not decrease: ${first.toFixed(3)} → ${last.toFixed(3)} (model did not learn — raise epochs/lr)`);
        ctx.bag.converged = true;
        return `loss ${first.toFixed(3)} → ${last.toFixed(3)} (−${(first - last).toFixed(3)})`;
      }),
  },
  "generate-check": {
    info: { type: "generate-check", layer: "BUILD", label: "Generation check", description: "Validate the model generates non-empty, reproducible text (params: prompt, seed, temperature)" },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Generation quality + determinism", async (ctx) => {
        const model = ctx.bag.model as EvermindLM | undefined;
        const tok = ctx.bag.tokenizer as BPETokenizer | undefined;
        if (!model || !tok) throw new Error("no trained model — add 'train-model' first");
        const prompt = pStr(cfg, "prompt", "The");
        const maxNewTokens = pNum(cfg, "maxNewTokens", 8);
        const seed = pNum(cfg, "seed", 1234);
        const temperature = pNum(cfg, "temperature", 0.8);
        // Same seed + temperature MUST reproduce — serving relies on it.
        const a = model.generateText(prompt, tok, { maxNewTokens, temperature, seed });
        const b = model.generateText(prompt, tok, { maxNewTokens, temperature, seed });
        if (a !== b) throw new Error("sampling is non-deterministic for a fixed seed — serving would be unreproducible");
        const greedy = model.generateText(prompt, tok, { maxNewTokens, temperature: 0 });
        if (typeof greedy !== "string" || greedy.trim().length === 0) throw new Error("model produced empty output");
        ctx.bag.sample = greedy;
        return `deterministic@seed ${seed}; sample "${greedy.slice(0, 40)}"`;
      }),
  },
  roundtrip: {
    info: { type: "roundtrip", layer: "BUILD", label: "Deploy round-trip", description: "Package the TRAINED model → load → generate; output must match (the real serve smoke test)" },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Deploy round-trip (.evermind)", async (ctx) => {
        const model = ctx.bag.model as EvermindLM | undefined;
        const tok = ctx.bag.tokenizer as BPETokenizer | undefined;
        if (!model || !tok) throw new Error("no trained model — add 'train-model' first");
        const prompt = pStr(cfg, "prompt", "The");
        const maxNewTokens = pNum(cfg, "maxNewTokens", 6);
        const before = model.generateText(prompt, tok, { maxNewTokens, temperature: 0 });
        const blob = EvermindModelPackage.fromLM(model, {
          name: pStr(cfg, "name", "custom-evermind"),
          version: pStr(cfg, "version", "1.0.0"),
          card: { description: pStr(cfg, "description", "round-trip validated model") },
        }).toBlob();
        const pkg = EvermindModelPackage.fromBlob(blob);
        const v = pkg.validate();
        if (!v.ok) throw new Error(`packaged artifact failed validation: ${v.errors.join("; ")}`);
        const after = pkg.loadLM().generateText(prompt, tok, { maxNewTokens, temperature: 0 });
        if (after !== before) throw new Error("served model produced different output than the trained model");
        ctx.artifacts.evermind = blob;
        ctx.artifacts.tokenizer = { vocab: Object.fromEntries(tok.vocab), merges: [...tok.merges.keys()] };
        return `artifact ${blob.byteLength} bytes; trained vs served output matches`;
      }),
  },
  "video-train": {
    info: {
      type: "video-train",
      layer: "BUILD",
      label: "Train video model",
      description:
        "Fit a temporal residual-VQ codec on clips, then train an EvermindLM on the video token streams — the model that generates video (params: height, width, channels, patch, levels, codebookSize, clips, frames, epochs, lr)",
    },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Train video model (codec + EvermindLM)", async (ctx) => {
        const videos =
          (ctx.bag.videos as Video[] | undefined) ??
          synthVideos(
            pNum(cfg, "clips", 3),
            pNum(cfg, "frames", 4),
            pNum(cfg, "height", 8),
            pNum(cfg, "width", 8),
            pNum(cfg, "channels", 3),
          );
        if (videos.length === 0) throw new Error("no clips to train on — provide ctx.bag.videos or a positive 'clips' param");

        // Learn the codec's codebooks, then autoregress the (unchanged) EvermindLM
        // over the resulting token streams. Text region is 0 for a pure-video model.
        const codec = makeVideoCodec(cfg, 0);
        const codecMSE = codec.fit(videos, { iterations: pNum(cfg, "codecIterations", 10) });
        const sequences = videos.map((v) => buildVideoSequence(codec, [], v));
        const model = new EvermindLM({
          vocabSize: codec.vocabSize,
          dModel: pNum(cfg, "dModel", 32),
          numLayers: pNum(cfg, "numLayers", 2),
          hiddenDim: pNum(cfg, "hiddenDim", 48),
          seed: pNum(cfg, "seed", 11),
        });
        const epochs = pNum(cfg, "epochs", 60);
        const history = new EvermindLMTrainer(model, { lr: pNum(cfg, "lr", 0.03), epochs }).fit(sequences);

        ctx.bag.videoCodec = codec;
        ctx.bag.videoModel = model;
        ctx.bag.videoTrainingHistory = history;
        const last = history.at(-1) ?? 0;
        return `codec MSE ${codecMSE.toFixed(4)}; trained ${epochs} epochs over ${sequences.length} clip(s), loss ${last.toFixed(3)}`;
      }),
  },
  "video-roundtrip": {
    info: {
      type: "video-roundtrip",
      layer: "BUILD",
      label: "Video round-trip",
      description:
        "Fit the video codec and gate reconstruction quality (encode→decode MSE), then smoke-test generate→decode yields valid frames (params: maxMSE, height, width, levels, codebookSize)",
    },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Video codec round-trip (encode→decode)", async (ctx) => {
        const videos =
          (ctx.bag.videos as Video[] | undefined) ??
          synthVideos(
            pNum(cfg, "clips", 3),
            pNum(cfg, "frames", 4),
            pNum(cfg, "height", 8),
            pNum(cfg, "width", 8),
            pNum(cfg, "channels", 3),
          );
        // Reuse a codec trained earlier if present; otherwise fit a fresh one.
        let codec = ctx.bag.videoCodec as VideoRVQCodec | undefined;
        const before = codec ? videoReconMSE(codec, videos) : Infinity;
        if (!codec) {
          codec = makeVideoCodec(cfg, 0);
          codec.fit(videos, { iterations: pNum(cfg, "codecIterations", 10) });
        }
        const mse = videoReconMSE(codec, videos);

        // Structural check: encode is a self-delimiting stream that decodes to the
        // same frame count and shape — the real serve smoke test for video.
        const toks = codec.encode(videos[0]!);
        const recon = codec.decode(toks);
        if (recon.length !== videos[0]!.length) throw new Error(`round-trip frame count ${recon.length} ≠ ${videos[0]!.length}`);
        for (const f of recon) {
          if (f.length !== videos[0]![0]!.length) throw new Error("round-trip frame shape mismatch");
        }

        // A trained model (if present) → the real deploy round-trip: package the
        // model + codec into a self-contained .evermind, reload it, and generate.
        // Served output must match the in-memory model, exactly like text roundtrip.
        const model = ctx.bag.videoModel as EvermindLM | undefined;
        if (model) {
          const before = generateVideo(model, codec, [], { maxNewTokens: toks.length });
          const modality = pNum(cfg, "frames", 4) === 1 ? "image" : "video";
          const blob = EvermindModelPackage.fromMediaLM(model, codec, {
            name: pStr(cfg, "name", "custom-evermind-video"),
            version: pStr(cfg, "version", "1.0.0"),
            modality,
            card: { description: pStr(cfg, "description", "round-trip validated media model") },
          }).toBlob();
          const pkg = EvermindModelPackage.fromBlob(blob);
          const val = pkg.validate();
          if (!val.ok) throw new Error(`packaged media artifact failed validation: ${val.errors.join("; ")}`);
          const served = pkg.loadMediaLM();
          const after = generateVideo(served.lm, served.codec, [], { maxNewTokens: toks.length });
          if (after.tokens.join(",") !== before.tokens.join(",")) {
            throw new Error("served media model produced different tokens than the trained model");
          }
          for (const f of after.video) {
            if (f.length !== videos[0]![0]!.length) throw new Error("generated frame shape mismatch");
          }
          ctx.artifacts.evermind = blob;
        }

        // Quality gate — only enforced when configured.
        const maxMSE = pNum(cfg, "maxMSE", 0);
        if (maxMSE > 0 && mse > maxMSE) {
          throw new Error(`video reconstruction MSE ${mse.toFixed(4)} exceeds max ${maxMSE} (raise levels/codebookSize or fit longer)`);
        }
        ctx.bag.videoReconMSE = mse;
        const improved = Number.isFinite(before) ? ` (was ${before.toFixed(4)} unfit)` : "";
        return `recon MSE ${mse.toFixed(4)}${improved}; ${recon.length} frames round-trip clean`;
      }),
  },
  package: {
    info: { type: "package", layer: "BUILD", label: "Package", description: "Bundle the trained model into a portable .evermind artifact (params: name, version)" },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Package .evermind artifact", async (ctx) => {
        const model = ctx.bag.model as EvermindLM | undefined;
        if (!model) throw new Error("no trained model — add 'train-model' first");
        const blob = EvermindModelPackage.fromLM(model, {
          name: pStr(cfg, "name", "custom-evermind"),
          version: pStr(cfg, "version", "1.0.0"),
          card: { description: pStr(cfg, "description", "workflow-built model") },
        }).toBlob();
        ctx.artifacts.evermind = blob;
        // Ship the tokenizer alongside so the deployed model has text I/O.
        const tok = ctx.bag.tokenizer as BPETokenizer | undefined;
        if (tok) {
          ctx.artifacts.tokenizer = { vocab: Object.fromEntries(tok.vocab), merges: [...tok.merges.keys()] };
        }
        return `packaged ${blob.byteLength} bytes${tok ? " + tokenizer" : ""}`;
      }),
  },
  export: {
    info: {
      type: "export",
      layer: "BUILD",
      label: "Export",
      description:
        "Export the trained model to a publishable format (params: format=huggingface|onnx|safetensors|gguf, name, version, license, fp16)",
    },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Export model (publishable)", async (ctx) => {
        const model = ctx.bag.model as EvermindLM | undefined;
        if (!model) throw new Error("no trained model — add 'train-model' first");
        const tok = ctx.bag.tokenizer as BPETokenizer | undefined;
        const format = pStr(cfg, "format", "huggingface") as ExportFormat;
        if (format === "huggingface" && !tok) {
          throw new Error("huggingface export needs a tokenizer — add 'train-tokenizer' first");
        }
        const result = exportEvermind(
          model,
          format,
          {
            name: pStr(cfg, "name", "Evermind"),
            version: pStr(cfg, "version", "1.0.0"),
            license: pStr(cfg, "license", "mit"),
            author: pStr(cfg, "author", ""),
            description: pStr(cfg, "description", "workflow-built Evermind model"),
            fp16: pBool(cfg, "fp16", false),
          },
          tok,
        );
        const total = result.files.reduce((n, f) => n + fileBytes(f.data), 0);
        // Land the file set as a workflow OUTPUT so a publish step / portal can ship it.
        ctx.artifacts.export = { format, files: result.files, paramCount: result.paramCount };
        return `exported ${format}: ${result.files.length} file(s), ${total} bytes, ${result.paramCount} params`;
      }),
  },

  "import-model": {
    info: {
      type: "import-model",
      layer: "BUILD",
      label: "Import model (warm-start)",
      description:
        "Warm-start an EvermindLM from a .safetensors checkpoint instead of random init — round-trips our own exports, or maps a foreign SSM checkpoint via renameMap (params: safetensors, safetensorsBase64, topK, renameMap)",
    },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Import model (warm-start)", async (ctx) => {
        const bytes = toBytes(cfg.params?.safetensors, pStr(cfg, "safetensorsBase64", ""));
        if (!bytes) {
          throw new Error("import-model: provide 'safetensors' (bytes/ArrayBuffer/number[]) or 'safetensorsBase64'");
        }
        const opts: ImportOptions = {};
        const topK = pNum(cfg, "topK", 0);
        if (topK > 0) opts.topK = topK;
        // renameMap is the JSON-friendly form of ImportOptions.rename — used to
        // translate a foreign SSM checkpoint's tensor names to our canonical names.
        const renameMap = cfg.params?.renameMap as Record<string, string> | undefined;
        if (renameMap && typeof renameMap === "object") {
          opts.rename = (name) => (name in renameMap ? renameMap[name]! : name);
        }
        const model = importEvermind(bytes, opts);
        ctx.bag.model = model;
        const params = model.parameters().reduce((n, p) => n + p.data.length, 0);
        const c = model.config;
        return `imported ${params} params (dModel ${c.dModel}, ${c.numLayers}L, ${c.numExperts}E) from ${bytes.length} bytes`;
      }),
  },

  // ── Teaching Evermind to code — multi-teacher distillation + exec gates ───────
  "distill-corpus": {
    info: {
      type: "distill-corpus",
      layer: "BUILD",
      label: "Distill from teacher panel",
      description:
        "Build a training corpus from a panel of teacher LLMs (Opus + any other model — the gateway, OpenRouter, OpenAI, a local server, Anthropic). Per task, queries every teacher, keeps only completions that PASS the task's execution cases (best-of-N, strongest teacher first), tags each with its teacher, and ACCUMULATES so stacked steps add up. Feeds the corpus to train-model (params: teachers[{provider,baseUrl,apiKey,model}], tasks[{prompt,cases}], prompts, provider, baseUrl, apiKey, model, systemPrompt, maxTokens, temperature, pairs)",
    },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Distill corpus from teacher panel", async (ctx) => {
        // Accumulate across steps so multiple teacher panels stack instead of
        // overwriting a prior distill step's exemplars.
        const pairs: DistillPair[] = Array.isArray(ctx.bag.distillPairs) ? (ctx.bag.distillPairs as DistillPair[]) : [];
        const startCount = pairs.length;
        let offlineKept = 0;
        let verified = 0;
        let dropped = 0;

        // Offline exemplars (pre-captured) — a single trusted candidate each,
        // execution-verified when the pair carries `cases`.
        const offline = cfg.params?.pairs;
        if (Array.isArray(offline)) {
          for (const p of offline) {
            const prompt = (p as DistillPair)?.prompt;
            const completion = (p as DistillPair)?.completion;
            if (typeof prompt !== "string" || typeof completion !== "string") continue;
            const cases = normalizeCases((p as DistillPair).cases);
            const chosen = selectBestCandidate([{ teacher: "offline", completion: stripCodeFences(completion) }], cases);
            if (!chosen) { dropped++; continue; }
            const pair: DistillPair = { prompt, completion: chosen.completion, teacher: "offline" };
            if (cases.length) pair.cases = cases;
            pairs.push(pair);
            offlineKept++;
          }
        }

        // Live teacher panel — Opus and/or any other models, best-of-N + exec-filtered.
        const teachers = makeTeachers(cfg);
        const tasks = readTasks(cfg);
        if (teachers.length > 0 && tasks.length > 0) {
          const temperature = pNum(cfg, "temperature", 0.2);
          // Dedupe identical (teacher, prompt) calls within this run so a repeated
          // prompt is not billed twice.
          const memo = new Map<string, string>();
          for (const task of tasks) {
            const candidates: Candidate[] = [];
            for (const t of teachers) {
              const key = `${t.label} ${task.prompt}`;
              let completion = memo.get(key);
              if (completion === undefined) {
                completion = stripCodeFences(await t.bridge.generate(task.prompt, { temperature }));
                memo.set(key, completion);
              }
              if (completion.trim()) candidates.push({ teacher: t.label, completion });
            }
            const chosen = selectBestCandidate(candidates, task.cases ?? []);
            if (!chosen) { dropped++; continue; }
            const pair: DistillPair = { prompt: task.prompt, completion: chosen.completion, teacher: chosen.teacher };
            if (task.cases?.length) pair.cases = task.cases;
            pairs.push(pair);
            verified++;
          }
        }

        if (pairs.length === 0) {
          throw new Error(
            "distill-corpus produced no exemplars — set teachers (provider + apiKey + prompts/tasks; baseUrl for openai-compatible) or supply offline 'pairs'",
          );
        }

        const corpus = pairs.map((p) => `${p.prompt}\n${p.completion}`).join("\n\n");
        ctx.bag.corpus = corpus;
        ctx.bag.distillPairs = pairs;

        const added = pairs.length - startCount;
        const parts: string[] = [];
        if (offlineKept) parts.push(`${offlineKept} offline`);
        if (verified) parts.push(`${verified} verified via ${teachers.length} teacher(s): ${teachers.map((t) => t.label).join(", ")}`);
        if (dropped) parts.push(`${dropped} dropped`);
        return `distilled ${added} exemplar(s) (${pairs.length} total, ${corpus.length} chars)${parts.length ? ` — ${parts.join(", ")}` : ""}`;
      }),
  },
  "code-parse-check": {
    info: {
      type: "code-parse-check",
      layer: "BUILD",
      label: "Code parse check",
      description:
        "Validate generated output is structurally valid code — balanced delimiters/strings, optional JS parse (params: code, language, minScore)",
    },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Code structure / parse gate", async (ctx) => {
        const code = pStr(cfg, "code", (ctx.bag.sample as string) ?? "");
        if (!code.trim()) throw new Error("no code to check — run 'evaluate'/'generate-check' first or pass a 'code' param");
        const language = pStr(cfg, "language", "js");
        const analysis = analyzeCode(code, language);
        ctx.bag.codeParse = analysis;
        const minScore = pNum(cfg, "minScore", 0); // only enforced when configured
        if (minScore > 0 && analysis.score < minScore) {
          throw new Error(
            `code structure score ${analysis.score.toFixed(2)} < min ${minScore} (unbalanced delimiters/strings — output is not valid ${language})`,
          );
        }
        return `structure ${(analysis.score * 100) | 0}%${language === "js" ? `, js-parse ${analysis.jsParse ? "ok" : "fail"}` : ""}`;
      }),
  },
  "code-eval": {
    info: {
      type: "code-eval",
      layer: "BUILD",
      label: "Code eval (reward)",
      description:
        "Execution-grounded reward: run the generated JS against test cases and score the pass-rate (params: code, cases, minPassRate)",
    },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Code execution reward", async (ctx) => {
        const code = pStr(cfg, "code", (ctx.bag.sample as string) ?? "");
        if (!code.trim()) throw new Error("no code to evaluate — run 'evaluate'/'generate-check' first or pass a 'code' param");
        const rawCases = cfg.params?.cases;
        const cases = Array.isArray(rawCases)
          ? rawCases.filter((c): c is { call: string; expect: unknown } => typeof (c as { call?: unknown })?.call === "string")
          : [];
        const result = runJsCases(code, cases);
        ctx.bag.codeEval = result;
        const minPassRate = pNum(cfg, "minPassRate", 0); // only enforced when configured
        if (minPassRate > 0 && result.passRate < minPassRate) {
          throw new Error(
            `code-eval pass-rate ${(result.passRate * 100) | 0}% < min ${(minPassRate * 100) | 0}%${result.error ? ` (${result.error})` : ""}`,
          );
        }
        return result.total > 0
          ? `${result.passed}/${result.total} cases passed (${(result.passRate * 100) | 0}%)`
          : "no cases — execution reward skipped";
      }),
  },
  "code-benchmark": {
    info: {
      type: "code-benchmark",
      layer: "BUILD",
      label: "Code benchmark (pass@1)",
      description:
        "Held-out coding benchmark: prompt the TRAINED model on N unseen tasks, execute each generated solution against its cases, and score pass@1 (a task passes only if ALL its cases pass). The symmetric coding counterpart to `benchmark`'s held-out perplexity (params: tasks:[{prompt,cases,maxNewTokens?}], maxNewTokens, temperature, minPass1)",
    },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Code benchmark (held-out pass@1)", async (ctx) => {
        const model = ctx.bag.model as EvermindLM | undefined;
        const tok = ctx.bag.tokenizer as BPETokenizer | undefined;
        if (!model || !tok) throw new Error("no trained model — add 'train-model' first");

        // Held-out tasks: each is a prompt the model did NOT train on plus the
        // execution cases its output must satisfy. A task counts as passed only
        // when EVERY case passes (strict pass@1), so this measures whether the
        // student learned to GENERATE working code, not just clean corpus tokens.
        const rawTasks = cfg.params?.tasks;
        const tasks = Array.isArray(rawTasks)
          ? rawTasks.filter(
              (t): t is { prompt: string; cases?: unknown; maxNewTokens?: unknown } =>
                typeof (t as { prompt?: unknown })?.prompt === "string",
            )
          : [];
        if (tasks.length === 0) {
          throw new Error("no held-out tasks — pass params.tasks: [{ prompt, cases: [{ call, expect }] }]");
        }

        const defaultMaxNewTokens = pNum(cfg, "maxNewTokens", 24);
        const temperature = pNum(cfg, "temperature", 0);
        const perTask = tasks.map((t) => {
          const cases: CodeCase[] = Array.isArray(t.cases)
            ? (t.cases as unknown[]).filter(
                (c): c is CodeCase => typeof (c as { call?: unknown })?.call === "string",
              )
            : [];
          const maxNewTokens = typeof t.maxNewTokens === "number" ? t.maxNewTokens : defaultMaxNewTokens;
          const generated = model.generateText(t.prompt, tok, { maxNewTokens, temperature });
          const code = stripCodeFences(typeof generated === "string" ? generated : "");
          const evalResult = runJsCases(code, cases);
          // Strict pass@1: the task passes only when it HAS cases and they ALL pass.
          const passed = evalResult.total > 0 && evalResult.passRate === 1;
          return { prompt: t.prompt, passed, cases: evalResult.total, casesPassed: evalResult.passed, error: evalResult.error };
        });

        const passedTasks = perTask.filter((r) => r.passed).length;
        const pass1 = perTask.length > 0 ? passedTasks / perTask.length : 0;
        ctx.bag.codeBenchmark = { pass1, passedTasks, totalTasks: perTask.length, perTask };

        // Optional quality gate — symmetric with `benchmark`'s maxPerplexity/minTop1.
        const minPass1 = pNum(cfg, "minPass1", 0); // only enforced when configured
        if (minPass1 > 0 && pass1 < minPass1) {
          throw new Error(
            `code pass@1 ${(pass1 * 100) | 0}% below min ${(minPass1 * 100) | 0}% (student did not learn to generate passing code — more/better TEACH_CODE corpus or epochs)`,
          );
        }
        return `pass@1 ${passedTasks}/${perTask.length} (${(pass1 * 100) | 0}%) over held-out coding tasks`;
      }),
  },
};
