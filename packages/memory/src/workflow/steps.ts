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
};
