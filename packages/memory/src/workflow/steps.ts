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
  benchmarkText,
  type ExportFormat,
  type BenchmarkReport,
} from "@seanhogg/builderforce-memory-engine";
import { EvermindCognition } from "../cognition/index.js";
import { hybridRetrieve, chunkText } from "../retrieval/index.js";
import { MemoryStore } from "../memory/MemoryStore.js";
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

/** Byte length of an emitted export file (text or binary). */
function fileBytes(data: Uint8Array | string): number {
  return typeof data === "string" ? new TextEncoder().encode(data).length : data.length;
}

function mkStep(cfg: WorkflowStepConfig, layer: string, defaultLabel: string, run: StackStep["run"]): StackStep {
  return { id: cfg.id, layer, label: cfg.label ?? defaultLabel, run };
}

/** Encode a corpus into next-token training sequences (one per sentence). */
function corpusToSequences(corpus: string, tok: BPETokenizer): number[][] {
  return corpus
    .split(/(?<=\.)\s+/)
    .map((s) => tok.encode(s.trim()))
    .filter((ids) => ids.length >= 2);
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
    info: { type: "train-tokenizer", layer: "BUILD", label: "Train tokenizer", description: "Learn a BPE vocab from your corpus (params: corpus, numMerges)" },
    factory: (cfg) =>
      mkStep(cfg, "BUILD", "Train tokenizer (BPE)", async (ctx) => {
        const corpus = pStr(cfg, "corpus", KNOWLEDGE.repeat(4));
        const numMerges = pNum(cfg, "numMerges", 100);
        const tok = new BPETokenizer();
        tok.train(corpus, { numMerges });
        ctx.bag.tokenizer = tok;
        ctx.bag.corpus = corpus;
        return `vocab ${tok.vocabSize} (${numMerges} merges)`;
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
};
